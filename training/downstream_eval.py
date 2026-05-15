# -*- coding: utf-8 -*-
"""
Downstream Evaluation (Classification, Regression, ADMET proxies)
 - Φορτώνει specialists (manifest ή glob) από το νέο training
 - Blending: anchor-based + (optional) perplexity-aware από manifest["val_ppl_best"]
 - Εξάγει ανά ακολουθία:
     * blended scalars: mean NLL/token, perplexity, mean entropy, mean logit margin
     * blended embedding (PROJ_DIM): projected thought vectors από κάθε specialist
     * AA composition (20-D) & length
 - Εκπαιδεύει με cross-validation (auto feature-view selection per fold):
     * Classification: LogisticRegressionCV (+optional isotonic calibration)
       (AUROC, AUPRC, F1, Brier, ECE, ROC/PR/Calibration curves)
     * Regression: RidgeCV/LassoCV/ElasticNetCV (RMSE/MAE/R²/Spearman/Pearson, scatter)
 - Αποθηκεύει predictions, metrics (JSON), figures (PNG), embeddings (Parquet)
"""

import os, re, json, time, glob, pickle, hashlib, warnings
from typing import List, Dict, Any, Optional, Tuple

# ---- JAX memory safety (set BEFORE importing jax) ----
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_curve, roc_curve, mean_squared_error, mean_absolute_error, r2_score
)

import jax
import jax.numpy as jnp
from jax import lax
warnings.filterwarnings("ignore", category=UserWarning)

# ===================== CONFIG =====================
# Specialists manifest από το multi-specialist build
SPECIALISTS_MANIFEST = os.environ.get("SPECIALISTS_MANIFEST", "/kaggle/input/latestspecialists/pytorch/default/1/specialists_manifest.json")
SPECIALIST_PKLS_GLOB = os.environ.get("SPECIALISTS_GLOB", "/kaggle/input/latestspecialists/pytorch/default/1/specialist_*.pkl")  # fallback

# Δώσε datasets εδώ (ένα entry ανά task)
# Κάθε CSV πρέπει να έχει τουλάχιστον: "sequence" και label/value στήλη
# task: "classification" | "regression"
# transform (optional για regression): "log10" | "log1p" | None
DATASETS = [
    {
        "name": "YAP1_DMS_regression",
        "csv": "/kaggle/working/datasets/yap1_dms_regression.csv",
        "seq_col": "sequence",
        "label_col": "DMS_score",
        "task": "regression",
        "transform": None,          # π.χ. "log10" για MIC
        "higher_is_better": True    # γύρνα σε False αν το metric είναι αντίστροφο (π.χ. MIC)
    },
    {
        "name": "YAP1_DMS_classification_q33",
        "csv": "/kaggle/working/datasets/yap1_dms_classification_q33.csv",
        "seq_col": "sequence",
        "label_col": "label",
        "task": "classification"
    },

    # Παραδείγματα για τα δικά σου datasets:
    # {"name":"AMP_activity", "csv":"/kaggle/input/amp/amp_binary.csv",
    #  "seq_col":"sequence", "label_col":"label", "task":"classification"},
    # {"name":"MIC", "csv":"/kaggle/input/mic/mic_values.csv",
    #  "seq_col":"sequence", "label_col":"MIC", "task":"regression", "transform":"log10",
    #  "higher_is_better": False},
    # {"name":"Hemolysis", "csv":"/kaggle/input/hemo/hemo.csv",
    #  "seq_col":"sequence", "label_col":"hemolysis", "task":"regression"},
    # {"name":"Cytotox", "csv":"/kaggle/input/cytotox/cytotox.csv",
    #  "seq_col":"sequence", "label_col":"viability", "task":"regression", "higher_is_better": True},
    # {"name":"Solubility", "csv":"/kaggle/input/sol/sol.csv",
    #  "seq_col":"sequence", "label_col":"sol", "task":"regression"}
]

OUT_DIR = os.environ.get("DOWNSTREAM_OUT", "/kaggle/working/eval_downstream")
PROJ_DIM = int(os.environ.get("PROJ_DIM", "128"))       # embedding dimensionality (fixed across specialists)     256
BLEND_POLICY = os.environ.get("BLEND_POLICY", "triangular")  # "triangular" | "inverse_dist" | "hard"

# Perplexity-aware reweight (από manifest["val_ppl_best"])
USE_VALPPL_WEIGHT = True
VALPPL_BETA = float(os.environ.get("VALPPL_BETA", "1.2"))   # μεγαλύτερο => πιο “κοφτερό” reweight

CV_FOLDS = int(os.environ.get("CV_FOLDS", "5"))
RNG_SEED = 42
#FEAT_MODES = ["embed", "scalars", "fuse"]

FEAT_MODES = ["embed", "fuse"]


# Classification calibration
CALIBRATE_PROBS = True    # isotonic στο outer fold
ECE_NBINS = 15

# ===================== HELPERS =====================
def ensure_dir(p): os.makedirs(p, exist_ok=True)
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a:i for i,a in enumerate(AA20)}

def aa_composition(seq: str) -> np.ndarray:
    v = np.zeros((20,), np.float32)
    L = max(1, len(seq))
    for ch in seq:
        i = AA2IDX.get(ch)
        if i is not None: v[i] += 1.0
    return v / float(L)

def normalize_labels_reg(y: np.ndarray, transform: Optional[str]):
    y = np.asarray(y, np.float64)
    if transform == "log10":
        return np.log10(y + 1e-12)
    if transform == "log1p":
        return np.log1p(y)
    return y

def parse_class_labels(y):
    if isinstance(y, pd.Series): y = y.to_numpy()
    if y.dtype.kind in "biu":
        return y.astype(int)
    y_s = pd.Series(y).astype(str).str.strip().str.lower()
    pos = {"1","true","yes","active","pos","positive"}
    return y_s.apply(lambda s: 1 if s in pos else 0).to_numpy(int)

def compute_ece(y_true, probs, n_bins=15):
    y_true = np.asarray(y_true, int)
    probs = np.asarray(probs, np.float64)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    inds = np.digitize(probs, bins) - 1
    ece = 0.0; total = len(probs)
    for b in range(n_bins):
        m = inds == b
        if not np.any(m): continue
        conf = probs[m].mean()
        acc  = y_true[m].mean()
        ece += np.abs(acc - conf) * (m.sum() / total)
    return float(ece)

# ===================== MODEL IO =====================
def is_compact(params):
    try:
        return ('encoder' in params and 'decoder' in params
                and 'mps' in params['encoder'] and 'isos' in params['encoder']
                and 'mps' in params['decoder'] and 'projection_matrix' in params
                and 'output_projection' in params)
    except Exception:
        return False

class ProcShim:
    def sequence_to_indices(self, seq):
        vocab = getattr(self, "vocab", {}) or {}
        unk = vocab.get("<UNK>", 0)
        return np.asarray([vocab.get(tok, unk) for tok in seq], dtype=np.int32)

def load_manifest(path: str) -> List[Dict[str,Any]]:
    with open(path, "r", encoding="utf-8") as f:
        man = json.load(f)
    return list(man.get("specialists", []))

def load_specialist_pkl(pkl_path: str):
    payload = pickle.load(open(pkl_path, "rb"))
    if isinstance(payload, tuple) and len(payload) >= 3:
        params, processor, max_len = payload[:3]
    elif isinstance(payload, dict) and "params" in payload:
        params = payload["params"]; processor = payload.get("processor", None)
        max_len = payload.get("max_len", params["encoder"]["mps"].shape[0])
    else:
        raise RuntimeError(f"Unsupported pickle format: {pkl_path}")
    if processor is None:
        processor = ProcShim()
    compact = is_compact(params)
    chi_mps = int(params['encoder']['mps'].shape[-1])
    out = {
        "path": pkl_path,
        "params": params,
        "processor": processor,
        "max_len": int(max_len),
        "anchor_len": int(max_len),  # override later if manifest έχει explicit anchor
        "window": (1, max_len-2),
        "compact": bool(compact),
        "bond_dim_mps": chi_mps,
        "val_ppl_best": None,
    }
    return out

def load_all_specialists() -> List[Dict[str,Any]]:
    specs = []
    if os.path.exists(SPECIALISTS_MANIFEST):
        try:
            meta = load_manifest(SPECIALISTS_MANIFEST)
            for it in meta:
                pth = it["path"]
                if os.path.exists(pth):
                    s = load_specialist_pkl(pth)
                    s["anchor_len"] = int(it.get("anchor_len", s["max_len"]))
                    s["window"] = tuple(it.get("window", (1, s["max_len"]-2)))
                    s["val_ppl_best"] = float(it.get("val_ppl_best", np.nan)) if ("val_ppl_best" in it) else None
                    specs.append(s)
        except Exception as e:
            print(f"⚠️ manifest read failed: {e}")
    if not specs:
        for p in sorted(glob.glob(SPECIALIST_PKLS_GLOB)):
            try:
                s = load_specialist_pkl(p)
                specs.append(s)
            except Exception as e:
                print(f"⚠️ skip {p}: {e}")
    if not specs:
        raise RuntimeError("No specialists found.")
    return specs

# ===================== ENCODE/DECODE PRIMS (COMPACT) =====================
def pad_and_pack(processor, seq_chars, max_len):
    sos=processor.vocab['<SOS>']; eos=processor.vocab['<EOS>']; pad=processor.vocab['<PAD>']
    seq = ['<SOS>'] + list(seq_chars)[:max_len-2] + ['<EOS>']
    idx = processor.sequence_to_indices(seq)
    if len(idx) < max_len:
        idx = np.concatenate([idx, np.full((max_len-len(idx),), pad, np.int32)])
    return idx

def _encode_step(carry, token_and_pos):
    state, mps = carry
    tok, pos = token_and_pos
    W = mps[pos, tok]
    state = state @ W
    return (state, mps), state

@jax.jit
def encode_compact(enc_params, seq_indices):
    L = enc_params['mps'].shape[0]
    chi = enc_params['mps'].shape[-1]
    state0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L, dtype=jnp.int32)
    (_, _), states = lax.scan(_encode_step, (state0, enc_params['mps']), (seq_indices, positions))
    def pair_reduce(_, i):
        l = states[2*i]; r = states[2*i+1]
        iso = enc_params['isos'][i]
        hi  = jnp.einsum('i,j,ijk->k', l, r, iso)
        return None, hi
    _, hi_list = lax.scan(pair_reduce, None, jnp.arange(L//2))
    tv = hi_list.reshape(-1)
    return tv / (jnp.linalg.norm(tv) + 1e-9)

def _decode_step(carry, tok_and_pos):
    vec, ctx, mps_dec, out_proj, first_flag = carry
    tok, pos = tok_and_pos
    W = mps_dec[pos, tok]
    vec = vec @ W
    vec = jnp.where(first_flag, vec + ctx, vec)
    logits = vec @ out_proj
    return (vec, ctx, mps_dec, out_proj, False), logits

@jax.jit
def decode_teacher_forcing_compact(params, thought_vector, decoder_input):
    L = params['decoder']['mps'].shape[0]
    chi = params['decoder']['mps'].shape[-1]
    ctx = thought_vector @ params['projection_matrix']
    vec0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L-1, dtype=jnp.int32)
    mps_dec = params['decoder']['mps'][:-1]
    (_, _, _, _, _), logits = lax.scan(
        _decode_step,
        (vec0, ctx, mps_dec, params['output_projection'], True),
        (decoder_input, positions)
    )
    return logits  # (L-1, V)

# ===================== BLENDING / PICK =====================
def pick_applicable_specialists(specs: List[Dict[str,Any]], L: int) -> List[Dict[str,Any]]:
    out = []
    for s in specs:
        lo, hi = s.get("window", (1, s["max_len"]-2))
        if lo <= L <= hi and s["max_len"] >= (L + 2):
            out.append(s)
    if not out:
        # fallback: closest by anchor_len
        out = [min(specs, key=lambda ss: abs(ss["anchor_len"]-L))]
    return out

def _valppl_weights(specs: List[Dict[str,Any]]):
    vals = []
    for s in specs:
        v = s.get("val_ppl_best")
        vals.append(v if (v is not None and np.isfinite(v)) else np.nan)
    vals = np.asarray(vals, np.float64)
    if np.all(np.isnan(vals)):  # no val ppl info
        return np.ones((len(specs),), np.float64)
    m = np.nanmin(vals)
    # higher ppl => smaller weight; best gets 1.0
    w = np.exp(-VALPPL_BETA * (vals - m) / (m + 1e-9))
    w[~np.isfinite(w)] = 0.0
    return w

def blend_weights(L: int, specs: List[Dict[str,Any]], policy="triangular") -> np.ndarray:
    if len(specs) == 1 or policy == "hard":
        w = np.zeros((len(specs),), np.float64); w[np.argmin([abs(s["anchor_len"]-L) for s in specs])] = 1.0
        return w
    if policy == "inverse_dist":
        dist = np.array([abs(s["anchor_len"] - L) for s in specs], dtype=np.float64)
        w = 1.0 / (dist + 1e-6)
    else:
        # triangular μέσα στο window
        ws=[]
        for s in specs:
            lo, hi = s.get("window", (1, s["max_len"]-2))
            half = (max(2, hi-lo+1))/2.0
            a = float(s["anchor_len"])
            ws.append(max(0.0, 1.0 - abs(L - a) / (half + 1e-6)))
        w = np.array(ws, np.float64)
        if w.sum() <= 0: w = np.ones_like(w)
    if USE_VALPPL_WEIGHT:
        w_val = _valppl_weights(specs)
        w = w * w_val
    w = w / w.sum()
    return w

# ===================== FEATURE EXTRACTION =====================
_proj_cache: Dict[str, np.ndarray] = {}

def _proj_for_specialist(spec_path: str, in_dim: int, out_dim: int=PROJ_DIM) -> np.ndarray:
    key = f"{spec_path}|{in_dim}|{out_dim}"
    if key in _proj_cache: return _proj_cache[key]
    h = hashlib.sha256(key.encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2**32-1)
    rng = np.random.default_rng(seed)
    P = rng.normal(loc=0.0, scale=1.0/np.sqrt(max(1,in_dim)), size=(in_dim, out_dim)).astype(np.float32)
    _proj_cache[key] = P
    return P

def _sequence_logits_and_tv(spec, seq: str):
    par, proc, ML, chi = spec["params"], spec["processor"], spec["max_len"], spec["bond_dim_mps"]
    arr = pad_and_pack(proc, list(seq), ML)
    inp = jnp.asarray(arr)
    tv = encode_compact(par['encoder'], inp)
    logits = decode_teacher_forcing_compact(par, tv, inp[:-1])  # (L-1,V)
    return logits, tv, inp  # return inp for masking

def _scalars_from_logits(logits, inp, pad_idx: int):
    # true targets (teacher forcing)
    tgt = inp[1:]
    mask = (tgt != pad_idx)
    logp = jax.nn.log_softmax(logits, axis=-1)
    tok_logp = logp[jnp.arange(logp.shape[0]), tgt]
    nll_full = -float((tok_logp * mask).sum())
    T = int(mask.sum())
    mean_nll = nll_full / max(1, T)

    # entropy & top-2 margin on predictive distribution
    p = jnp.exp(logp)
    ent = -jnp.sum(p * logp, axis=-1)                # (T,)
    top2 = jax.lax.top_k(logits, 2)[0]               # (T,2)
    margin = top2[:,0] - top2[:,1]

    ent_mean    = float(jnp.sum(ent * mask) / jnp.maximum(1, jnp.sum(mask)))
    margin_mean = float(jnp.sum(margin * mask) / jnp.maximum(1, jnp.sum(mask)))
    return mean_nll, ent_mean, margin_mean

def features_for_sequence(seq: str, specs_all: List[Dict[str,Any]]) -> Dict[str, Any]:
    L = len(seq)
    specs = pick_applicable_specialists(specs_all, L)
    w = blend_weights(L, specs, BLEND_POLICY)

    scalars = {"mean_nll":[], "perplexity":[], "entropy":[], "margin":[]}
    embeds = []

    for wi, s in zip(w, specs):
        logits, tv, inp = _sequence_logits_and_tv(s, seq)
        mean_nll, ent, mar = _scalars_from_logits(logits, inp, s["processor"].vocab['<PAD>'])
        scalars["mean_nll"].append(wi * mean_nll)
        scalars["perplexity"].append(wi * float(np.exp(mean_nll)))
        scalars["entropy"].append(wi * ent)
        scalars["margin"].append(wi * mar)

        # Project tv -> PROJ_DIM and weight-average
        tv_np = np.array(tv, dtype=np.float32)
        P = _proj_for_specialist(s["path"], tv_np.shape[0], PROJ_DIM)
        emb = tv_np @ P
        nrm = np.linalg.norm(emb); 
        if nrm > 0: emb = emb / nrm
        embeds.append(wi * emb)

    embed = np.sum(np.stack(embeds, axis=0), axis=0) if len(embeds)>0 else np.zeros((PROJ_DIM,), np.float32)

    # AA composition & length
    comp20 = aa_composition(seq)
    out = {
        "len": float(L),
        "embed": embed.astype(np.float32),
        "mean_nll": float(np.sum(scalars["mean_nll"])) if scalars["mean_nll"] else np.nan,
        "perplexity": float(np.sum(scalars["perplexity"])) if scalars["perplexity"] else np.nan,
        "entropy": float(np.sum(scalars["entropy"])) if scalars["entropy"] else np.nan,
        "margin": float(np.sum(scalars["margin"])) if scalars["margin"] else np.nan,
        "comp20": comp20.astype(np.float32),
    }
    return out

def row_to_feature_vector(feat: Dict[str,Any]) -> np.ndarray:
    # [embed(PROJ_DIM) | 20-D comp | 5 scalars (mean_nll, perplexity, entropy, margin, len)]
    scalars = np.array([feat["mean_nll"], feat["perplexity"], feat["entropy"], feat["margin"], feat["len"]], dtype=np.float32)
    return np.concatenate([feat["embed"], feat["comp20"], scalars], axis=0)

def _split_feature_views(X: np.ndarray, proj_dim: int):
    """
    Επιστρέφει τρεις 'όψεις':
      - base: embed + comp20 + scalars
      - fuse: ίδια με base (κρατάμε τη 'fusion' ονομασία για μελλοντική επέκταση)
      - lite: comp20 + scalars (χωρίς embedding) — χρήσιμο για πολύ μικρά datasets
    """
    base_end = proj_dim + 20 + 5
    X_base = X[:, :base_end]
    X_fuse = X[:, :base_end]  # reserved if αργότερα προσθέσουμε extra features
    X_lite = X[:, proj_dim:base_end]  # μόνο comp20 + scalars
    return X_base, X_fuse, X_lite

# ===================== EVAL ROUTINES =====================
def eval_classification_task(df: pd.DataFrame, name: str, out_dir: str) -> Dict[str, Any]:
    y = parse_class_labels(df["label"].to_numpy())
    X_all = np.stack(df["feat_vec"].to_numpy(), axis=0)
    X_base, X_fuse, X_lite = _split_feature_views(X_all, proj_dim=PROJ_DIM)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG_SEED)
    probs_all = np.zeros_like(y, dtype=np.float64)
    preds_all = np.zeros_like(y, dtype=int)
    fold_metrics = []

    for fold,(tr, te) in enumerate(skf.split(X_all, y), 1):
        # Επιλογή feature-view με το καλύτερο AUROC στο training (inner 3-fold)
        best_auc, best_view, best_sc, best_clf = -1.0, None, None, None
        for tag, Xv in [("fuse", X_fuse), ("base", X_base), ("lite", X_lite)]:
            sc = StandardScaler(with_mean=True, with_std=True)
            Xtr_sc = sc.fit_transform(Xv[tr])
            base = LogisticRegressionCV(Cs=np.logspace(-2,2,12), cv=3, solver="liblinear",
                                        penalty="l2", class_weight="balanced", max_iter=5000, random_state=RNG_SEED+fold)
            base.fit(Xtr_sc, y[tr])
            p_tr = base.predict_proba(Xtr_sc)[:,1]
            auc = roc_auc_score(y[tr], p_tr)
            if auc > best_auc:
                best_auc, best_view, best_sc, best_clf = auc, tag, sc, base

        Xtr_view = {"fuse":X_fuse, "base":X_base, "lite":X_lite}[best_view][tr]
        Xte_view = {"fuse":X_fuse, "base":X_base, "lite":X_lite}[best_view][te]
        Xtr_sc = best_sc.fit_transform(Xtr_view)
        Xte_sc = best_sc.transform(Xte_view)

        if CALIBRATE_PROBS:
            cal = CalibratedClassifierCV(best_clf, method="isotonic", cv=3)
            cal.fit(Xtr_sc, y[tr])
            p = cal.predict_proba(Xte_sc)[:,1]
        else:
            p = best_clf.predict_proba(Xte_sc)[:,1]

        probs_all[te] = p
        preds_all[te] = (p >= 0.5).astype(int)

        auroc = roc_auc_score(y[te], p)
        auprc = average_precision_score(y[te], p)
        f1    = f1_score(y[te], preds_all[te])
        brier = brier_score_loss(y[te], p)
        fold_metrics.append({"fold":fold, "AUROC":auroc, "AUPRC":auprc, "F1":f1, "Brier":brier, "feat_set":best_view})

    ece = compute_ece(y, probs_all, n_bins=ECE_NBINS)
    metrics = {
        "task": name, "type":"classification",
        "AUROC": float(np.mean([m["AUROC"] for m in fold_metrics])),
        "AUPRC": float(np.mean([m["AUPRC"] for m in fold_metrics])),
        "F1":    float(np.mean([m["F1"] for m in fold_metrics])),
        "Brier": float(np.mean([m["Brier"] for m in fold_metrics])),
        "ECE":   float(ece),
        "folds": fold_metrics,
    }

    # Curves
    fpr, tpr, _ = roc_curve(y, probs_all)
    prec, rec, _ = precision_recall_curve(y, probs_all)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{name} — ROC (AUROC={metrics['AUROC']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_roc.png")); plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{name} — PR (AUPRC={metrics['AUPRC']:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_pr.png")); plt.close()

    # Reliability diagram
    bins = np.linspace(0,1,ECE_NBINS+1); centers = 0.5*(bins[1:]+bins[:-1])
    inds = np.digitize(probs_all, bins)-1
    accs = [(y[inds==b].mean() if np.any(inds==b) else np.nan) for b in range(ECE_NBINS)]
    conf = [(probs_all[inds==b].mean() if np.any(inds==b) else np.nan) for b in range(ECE_NBINS)]
    plt.figure(figsize=(6,6))
    plt.plot([0,1],[0,1],'--', linewidth=1)
    plt.plot(np.array(conf,float), np.array(accs,float), marker='o')
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.title(f"{name} — Reliability (ECE={ece:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_reliability.png")); plt.close()

    # Save predictions
    out_pred = df[["sequence","label"]].copy()
    out_pred["prob_active"] = probs_all
    out_pred["pred_label"]  = preds_all
    out_pred.to_csv(os.path.join(out_dir, f"{name}_predictions.csv"), index=False)

    return metrics

def eval_regression_task(df: pd.DataFrame, name: str, out_dir: str, transform: Optional[str]) -> Dict[str, Any]:
    y = normalize_labels_reg(df["value"].to_numpy(), transform)
    X_all = np.stack(df["feat_vec"].to_numpy(), axis=0)
    X_base, X_fuse, X_lite = _split_feature_views(X_all, proj_dim=PROJ_DIM)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RNG_SEED)
    yhat_all = np.zeros_like(y, dtype=np.float64)
    fold_metrics = []

    for fold,(tr, te) in enumerate(kf.split(X_all), 1):
        # μικρό model selection στο training (inner)
        candidates = []
        for tag, Xv in [("fuse", X_fuse), ("base", X_base), ("lite", X_lite)]:
            pipe = Pipeline([("sc", StandardScaler()),
                             ("rg", RidgeCV(alphas=np.logspace(-4, 3, 25)))])
            pipe.fit(Xv[tr], y[tr])
            yh_tr = pipe.predict(Xv[tr])
            rmse_tr = mean_squared_error(y[tr], yh_tr, squared=False)
            candidates.append((rmse_tr, tag, pipe))
            # LassoCV
            pipe2 = Pipeline([("sc", StandardScaler()),
                              ("rg", LassoCV(alphas=np.logspace(-4, 1, 30), max_iter=30000, random_state=RNG_SEED+fold))])
            pipe2.fit(Xv[tr], y[tr])
            yh_tr2 = pipe2.predict(Xv[tr]); rmse_tr2 = mean_squared_error(y[tr], yh_tr2, squared=False)
            candidates.append((rmse_tr2, tag, pipe2))
            # ElasticNetCV
            pipe3 = Pipeline([("sc", StandardScaler()),
                              ("rg", ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=np.logspace(-4, 2, 25),
                                                  max_iter=30000, random_state=RNG_SEED+fold))])
            pipe3.fit(Xv[tr], y[tr])
            yh_tr3 = pipe3.predict(Xv[tr]); rmse_tr3 = mean_squared_error(y[tr], yh_tr3, squared=False)
            candidates.append((rmse_tr3, tag, pipe3))

        candidates.sort(key=lambda t: t[0])
        _, chosen_tag, best_model = candidates[0]
        Xte = {"fuse":X_fuse, "base":X_base, "lite":X_lite}[chosen_tag][te]
        yhat = best_model.predict(Xte)
        yhat_all[te] = yhat

        rmse = mean_squared_error(y[te], yhat, squared=False)
        mae  = mean_absolute_error(y[te], yhat)
        r2   = r2_score(y[te], yhat)
        sp   = spearmanr(y[te], yhat, nan_policy="omit")[0]
        pe   = pearsonr(y[te], yhat)[0]
        fold_metrics.append({"fold":fold, "RMSE":rmse, "MAE":mae, "R2":r2,
                             "Spearman":sp, "Pearson":pe, "feat_set":chosen_tag})

    metrics = {
        "task": name, "type":"regression", "transform": transform,
        "RMSE": float(np.mean([m["RMSE"] for m in fold_metrics])),
        "MAE":  float(np.mean([m["MAE"]  for m in fold_metrics])),
        "R2":   float(np.mean([m["R2"]   for m in fold_metrics])),
        "Spearman": float(np.nanmean([m["Spearman"] for m in fold_metrics])),
        "Pearson":  float(np.nanmean([m["Pearson"]  for m in fold_metrics])),
        "folds": fold_metrics,
    }

    # Scatter plot
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat_all, s=18, alpha=0.6, edgecolors="k")
    mn, mx = np.nanmin(y), np.nanmax(y)
    plt.plot([mn,mx],[mn,mx],'--', linewidth=1)
    plt.xlabel("Ground truth" + (" (transformed)" if transform else ""))
    plt.ylabel("Prediction")
    plt.title(f"{name} — RMSE={metrics['RMSE']:.3f}, R²={metrics['R2']:.3f}, ρ={metrics['Spearman']:.3f}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{name}_scatter.png")); plt.close()

    # Save predictions
    out_pred = df[["sequence"]].copy()
    out_pred["y_true"] = y
    out_pred["y_pred"] = yhat_all
    out_pred.to_csv(os.path.join(out_dir, f"{name}_predictions.csv"), index=False)

    return metrics

# ===================== MAIN =====================
def main():
    ensure_dir(OUT_DIR)
    print(f"[{now()}] ▶ Loading specialists…")
    specialists = load_all_specialists()
    print(f"Found {len(specialists)} specialists.")
    for s in specialists:
        vp = s.get("val_ppl_best", None)
        print(f" - {os.path.basename(s['path'])} | anchor={s['anchor_len']} | window={s.get('window')} | MAX_LEN={s['max_len']} | val_ppl_best={vp}")

    # Quick vocab sanity
    base_vocab = specialists[0]["processor"].vocab
    for s in specialists[1:]:
        if s["processor"].vocab != base_vocab:
            print("⚠️ Vocab mismatch across specialists — results may be inconsistent.")

    all_metrics = {"timestamp": now(), "specialists_used": [
        {"path": s["path"], "anchor_len": s["anchor_len"], "window": list(s.get("window",(1,s["max_len"]-2))), "max_len": s["max_len"]}
        for s in specialists
    ], "proj_dim": PROJ_DIM, "blend_policy": BLEND_POLICY, "cv_folds": CV_FOLDS, "tasks": []}

    # ===== Feature extraction per task =====
    for task in DATASETS:
        name = task["name"]
        csv  = task["csv"]
        seq_col = task.get("seq_col", "sequence")
        label_col = task.get("label_col")
        task_type = task["task"]
        transform = task.get("transform")

        print(f"\n[{now()}] ▶ Task: {name} ({task_type}) — reading {csv}")
        if not os.path.exists(csv):
            print(f"❌ Missing file: {csv} — skipping.")
            continue
        df = pd.read_csv(csv)
        if seq_col not in df.columns:
            print(f"❌ '{seq_col}' column not in {csv} — skipping.")
            continue
        # Clean sequences
        df[seq_col] = df[seq_col].astype(str).str.strip().str.upper().str.replace(r"[^A-Z]", "", regex=True)
        df = df[df[seq_col].str.fullmatch(rf"[{AA20}]+", na=False)].copy()
        df = df.drop_duplicates(subset=[seq_col]).reset_index(drop=True)

        print(f"[{name}] sequences after cleaning/dedup: {len(df)}")
        if len(df) == 0:
            print(f"⚠️ No sequences for {name}.")
            continue

        # Extract features
        feats = []
        seqs = df[seq_col].tolist()
        for i, seq in enumerate(seqs):
            if (i+1) % 200 == 0 or i==len(seqs)-1:
                print(f"  → {i+1}/{len(seqs)}")
            f = features_for_sequence(seq, specialists)
            feats.append(f)

        feat_mat = np.stack([row_to_feature_vector(f) for f in feats], axis=0)
        feat_cols = [f"emb_{i}" for i in range(PROJ_DIM)] + [f"aa_{a}" for a in AA20] + ["mean_nll","perplexity","entropy","margin","length"]

        # Build a dataframe με features + sequence
        feat_df = pd.DataFrame(feat_mat, columns=feat_cols)
        feat_df.insert(0, "sequence", df[seq_col].values)

        task_dir = os.path.join(OUT_DIR, name)
        ensure_dir(task_dir)
        feat_df.to_parquet(os.path.join(task_dir, f"{name}_features.parquet"), index=False)

        # Supervised evaluation
        metrics = {"name": name, "type":task_type}
        if task_type == "classification":
            if label_col is None or label_col not in df.columns:
                print(f"⚠️ No '{label_col}' column — saving features only.")
                feat_df.to_csv(os.path.join(task_dir, f"{name}_features_only.csv"), index=False)
                continue
            y = parse_class_labels(df[label_col].to_numpy())
            use_df = pd.DataFrame({"sequence":feat_df["sequence"], "label":y, "feat_vec":list(feat_mat)})
            m = eval_classification_task(use_df, name, task_dir)
            metrics.update(m)
        elif task_type == "regression":
            if label_col is None or label_col not in df.columns:
                print(f"⚠️ No '{label_col}' column — saving features only.")
                feat_df.to_csv(os.path.join(task_dir, f"{name}_features_only.csv"), index=False)
                continue
            y = normalize_labels_reg(df[label_col].to_numpy(), transform=None)  # raw για export
            use_df = pd.DataFrame({"sequence":feat_df["sequence"], "value":df[label_col].to_numpy(), "feat_vec":list(feat_mat)})
            m = eval_regression_task(use_df, name, task_dir, transform=transform)
            metrics.update(m)
        else:
            print(f"⚠️ Unknown task type: {task_type}")
            continue

        # Save per-task metrics
        with open(os.path.join(task_dir, f"{name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        all_metrics["tasks"].append(metrics)

    # Save global summary
    with open(os.path.join(OUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n[{now()}] ✅ Done.")
    print(f"↳ Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()
