# -*- coding: utf-8 -*-
"""
pep_product.py — Robust peptide AMP + MIC-ready (classification + regression) + Δ-MutScan

What you get
------------
• Specialist routing + projection bagging (n=5) για σταθερά embeddings
• GroupKFold by 3-mer signature (leakage-safe folds)
• Nested calibration + precision-targeted threshold (precision≥0.80 by default)
• Split-conformal margin για principled abstention
• Inference-time Δ-MutScan (fast/full) για single-aa robustness
• Νέα features: helical hydrophobic moment (μH), amphipathic index, Cys-topology/motifs
• Guardrails (CPP/hydrophobic traps, length), deterministic free tier
• CSV/FASTA batch prediction, καθαρό CLI, JSON training summaries
• Classification heads (π.χ. AMP/Toxicity/Stability/CPP) + Regression heads (π.χ. MIC)

"""

import os, re, json, time, glob, pickle, hashlib, warnings, argparse, math, joblib
from typing import List, Dict, Any, Tuple

# ---- JAX memory safety (set BEFORE importing jax) ----
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax import lax

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, HuberRegressor
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_recall_curve, mean_absolute_error, r2_score
)

warnings.filterwarnings("ignore", category=UserWarning)


class DataProcessor:
    """Minimal shim: αρκεί για unpickle + vocab + packing."""
    def sequence_to_indices(self, seq):
        vocab = getattr(self, "vocab", {}) or {}
        unk = vocab.get("<UNK>", 0)
        return np.asarray([vocab.get(tok, unk) for tok in seq], dtype=np.int32)


# ==== Adapter helpers (Platt-style) ====
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _safe_logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p/(1-p))

def apply_adapter_to_df(df: pd.DataFrame, adapter_cfg: dict, head_name: str):
    """
    Προσάρμοσε p με p_cal = sigmoid(a * logit(p) + b).
    - Αν adapter_cfg έχει 'head', εφαρμόζεται ΜΟΝΟ σ' αυτό το head.
    - Αλλιώς εφαρμόζεται σε ΟΛΕΣ τις στήλες που τελειώνουν σε '_prob'.
    - Αν υπάρχει 'thr' στο adapter, φτιάχνει *_prob_cal, *_label_cal, *_decision_cal.
    """
    a = float(adapter_cfg["a"]); b = float(adapter_cfg["b"])
    target_head = adapter_cfg.get("head", None)
    thr = adapter_cfg.get("thr", None)

    def _one(colprob, base_name):
        p = df[colprob].astype(float).to_numpy()
        p_cal = _sigmoid(a * _safe_logit(p) + b)
        df[f"{base_name}_prob_cal"] = p_cal
        if thr is not None:
            df[f"{base_name}_label_cal"] = (p_cal >= float(thr)).astype(int)
            if f"{base_name}_decision" in df.columns:
                dec = df[f"{base_name}_decision"].astype(str).to_numpy()
                dec = np.where(dec=="abstain", "abstain",
                               np.where(p_cal >= float(thr), "positive", "negative"))
                df[f"{base_name}_decision_cal"] = dec
        return df

    if target_head:
        base = target_head
        colprob = f"{base}_prob"
        if colprob in df.columns:
            df = _one(colprob, base)
    else:
        for col in list(df.columns):
            if col.endswith("_prob"):
                base = col[:-5]
                df = _one(col, base)
    return df


# ===================== DEFAULT CONFIG =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SPECIALISTS_MANIFEST = os.environ.get(
    "SPECIALISTS_MANIFEST", os.path.join(BASE_DIR, "specialists", "specialists_manifest.json")
)
SPECIALIST_PKLS_GLOB = os.environ.get(
    "SPECIALIST_PKLS_GLOB", os.path.join(BASE_DIR, "specialists", "specialist_*.pkl")
)
HEADS_DIR = os.environ.get("HEADS_DIR", os.path.join(BASE_DIR, "heads"))

os.makedirs(HEADS_DIR, exist_ok=True)

PROJ_DIM = int(os.environ.get("PROJ_DIM", "128"))
BLEND_POLICY = os.environ.get("BLEND_POLICY", "triangular")
RNG_SEED = 42
CV_FOLDS = int(os.environ.get("CV_FOLDS", "5"))
PRECISION_TARGET = float(os.environ.get("PRECISION_TARGET", "0.80"))
CONFORMAL_ALPHA = float(os.environ.get("CONFORMAL_ALPHA", "0.02"))
PROJ_BAG = int(os.environ.get("PROJ_BAG", "5"))
MIN_TRAIN_SAMPLES = int(os.environ.get("MIN_TRAIN_SAMPLES", "30"))

# ===================== AA maps & heuristics =====================
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i for i, a in enumerate(AA20)}

# Kyte-Doolittle hydrophobicities
KD = {
    'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,'S':-0.8,
    'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,'N':-3.5,'K':-3.9,'R':-4.5
}
HYDRO = set("AFILMVWYGC")

# For Δ-MutScan (fast mode) — biochemical neighbors
AA_NEIGHBORS = {
    "D":["E","N"], "E":["D","Q"], "K":["R","H"], "R":["K","H"], "H":["K","R"],
    "N":["D","Q","S"], "Q":["E","N","T"], "S":["T","A","N"], "T":["S","A","Q"],
    "A":["S","T","G","V"], "G":["A","S"], "V":["L","I","A"], "L":["I","V","M"],
    "I":["L","V","M"], "M":["L","I"], "F":["Y","W","L"], "Y":["F","W","H"],
    "W":["F","Y"], "C":["A","S"], "P":["A","S"]
}

# ===================== Basic feature helpers =====================
def aa_composition(seq: str) -> np.ndarray:
    v = np.zeros((20,), np.float32); L = max(1, len(seq))
    for ch in seq:
        i = AA2IDX.get(ch, None)
        if i is not None: v[i] += 1.0
    return v / float(L)

def gravy(seq: str) -> float:
    if not seq: return 0.0
    return float(np.mean([KD.get(ch, 0.0) for ch in seq]))

def net_charge_pH7(seq: str) -> float:
    pos = sum(ch in "KR" for ch in seq) + 0.1 * sum(ch == 'H' for ch in seq)
    neg = sum(ch in "DE" for ch in seq)
    return float(pos - neg)

def aromatic_fraction(seq: str) -> float:
    if not seq: return 0.0
    return sum(ch in "FWY" for ch in seq) / len(seq)

def hydro_fraction(seq: str) -> float:
    if not seq: return 0.0
    return sum(ch in HYDRO for ch in seq) / len(seq)

def max_hydrophobic_run(seq: str) -> int:
    m = 0; c = 0
    for ch in seq:
        if ch in HYDRO: c += 1; m = max(m, c)
        else: c = 0
    return m

# NEW: Helical hydrophobic moment (μH) and amphipathic index
_HELIX_RAD = math.radians(100.0)
def helical_moment(seq: str) -> float:
    x = y = 0.0
    for i, ch in enumerate(seq):
        k = KD.get(ch, 0.0)
        angle = i * _HELIX_RAD
        x += k * math.cos(angle); y += k * math.sin(angle)
    L = max(len(seq), 1)
    return float(math.sqrt(x*x + y*y) / L)

def amphipathic_index(seq: str) -> float:
    if not seq: return 0.0
    vals = []
    for i, ch in enumerate(seq):
        k = KD.get(ch, 0.0)
        ang = (i * _HELIX_RAD) % (2*math.pi)
        side = 1 if (ang < math.pi) else -1
        vals.append((side, k))
    s1 = [k for side, k in vals if side==1]
    s2 = [k for side, k in vals if side==-1]
    if not s1 or not s2:
        return 0.0
    return float(abs(np.mean(s1) - np.mean(s2)))

# NEW: motifs & Cys topology
MOTIFS = ["W..W", "KWK", "KLAK", "LRLR", "GLGF", "G...G"]
def motif_hits(seq: str) -> int:
    cnt = 0
    for m in MOTIFS:
        pattern = m.replace(".", "[A-Z]")
        cnt += len(re.findall(pattern, seq))
    return cnt

def cysteine_features(seq: str) -> Tuple[int,int,float]:
    idx = [i for i,ch in enumerate(seq) if ch=="C"]
    nC = len(idx)
    pairs = nC//2
    mean_gap = float(np.mean(np.diff(idx))) if len(idx)>=2 else 0.0
    return nC, pairs, mean_gap

def amp_activity_proxy_from_heuristics(seq: str) -> float:
    L = len(seq)
    charge = max(0.0, min(12.0, net_charge_pH7(seq)))
    hydro = hydro_fraction(seq)
    arom = aromatic_fraction(seq)
    length_ok = 1.0 if 10 <= L <= 40 else max(0.0, 1.0 - abs(L - 25.0) / 25.0)
    raw = 0.50 * (charge / 12.0) + 0.35 * hydro + 0.05 * (1.0 - arom) + 0.10 * helical_moment(seq)
    return float(np.clip(raw * length_ok, 0.0, 1.0))

def admet_scores_from_heuristics(seq: str) -> Tuple[float, float, float]:
    g = gravy(seq); h = hydro_fraction(seq); a = aromatic_fraction(seq)
    r = float(max_hydrophobic_run(seq)); q = abs(net_charge_pH7(seq))
    sol = (-g) + (-0.2 * r) + (1.0 - h)
    stab = 0.4 * h + 0.1 * a - 0.2 * q
    agg = 0.6 * h + 0.3 * r + 0.1 * a
    return (float(sol), float(stab), float(agg))

# ===================== Specialists I/O =====================
class ProcShim:
    def sequence_to_indices(self, seq):
        vocab = getattr(self, "vocab", {}) or {}
        unk = vocab.get("<UNK>", 0)
        return np.asarray([vocab.get(tok, unk) for tok in seq], dtype=np.int32)

def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        man = json.load(f)
    return list(man.get("specialists", []))

def load_specialist_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, tuple) and len(payload) >= 3:
        params, processor, max_len = payload[:3]
    else:
        raise RuntimeError(f"Unsupported pickle format: {pkl_path}")
    if not hasattr(processor, 'vocab'):
        processor = ProcShim()
    chi_mps = int(params['encoder']['mps'].shape[-1])
    return {
        "path": pkl_path, "params": params, "processor": processor,
        "max_len": int(max_len), "anchor_len": int(max_len), "window": (1, max_len - 2),
        "bond_dim_mps": chi_mps,
    }

def load_all_specialists() -> List[Dict[str, Any]]:
    specs = []
    if os.path.exists(SPECIALISTS_MANIFEST):
        try:
            meta = load_manifest(SPECIALISTS_MANIFEST)
            for it in meta:
                path = it.get("path", "")
                if os.path.exists(path):
                    s = load_specialist_pkl(path)
                    s.update(it)
                    specs.append(s)
        except Exception as e:
            print(f"⚠️ manifest read failed: {e}")
    if not specs:
        for p in sorted(glob.glob(SPECIALIST_PKLS_GLOB)):
            try:
                specs.append(load_specialist_pkl(p))
            except Exception as e:
                print(f"⚠️ skip {p}: {e}")
    if not specs:
        raise RuntimeError("No specialists found.")
    return specs

# ===================== Encode/Decode primitives (COMPACT) =====================
def pad_and_pack(processor, seq_chars, max_len):
    sos=processor.vocab['<SOS>']; eos=processor.vocab['<EOS>']; pad=processor.vocab['<PAD>']
    seq = ['<SOS>'] + list(seq_chars)[:max_len-2] + ['<EOS>']
    idx = processor.sequence_to_indices(seq)
    if len(idx) < max_len:
        idx = np.concatenate([idx, np.full((max_len-len(idx),), pad, np.int32)])
    return idx

@jax.jit
def encode_compact(enc_params, seq_indices):
    L, chi = enc_params['mps'].shape[0], enc_params['mps'].shape[-1]
    state0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    def scan_body(carry, x):
        state, mps = carry
        tok, pos = x
        return (state @ mps[pos, tok], mps), state @ mps[pos, tok]
    (_, _), states = lax.scan(scan_body, (state0, enc_params['mps']), (seq_indices, jnp.arange(L)))
    def pair_reduce(_, i):
        return None, jnp.einsum('i,j,ijk->k', states[2*i], states[2*i+1], enc_params['isos'][i])
    _, hi_list = lax.scan(pair_reduce, None, jnp.arange(L//2))
    tv = hi_list.reshape(-1)
    return tv / (jnp.linalg.norm(tv) + 1e-9)

@jax.jit
def decode_teacher_forcing_compact(params, thought_vector, decoder_input):
    L, chi = params['decoder']['mps'].shape[0], params['decoder']['mps'].shape[-1]
    ctx = thought_vector @ params['projection_matrix']
    vec0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    def scan_body(carry, x):
        vec, first_flag = carry
        tok, pos = x
        vec = vec @ params['decoder']['mps'][pos, tok]
        vec = jnp.where(first_flag, vec + ctx, vec)
        return (vec, False), vec @ params['output_projection']
    (_, _), logits = lax.scan(scan_body, (vec0, True), (decoder_input, jnp.arange(L-1)))
    return logits

# ===================== Routing & Projection bagging =====================
def pick_applicable_specialists(specs: List[Dict[str, Any]], L: int) -> List[Dict[str, Any]]:
    out = [s for s in specs if s.get("window",(1,1e9))[0] <= L <= s.get("window",(0,0))[1] and s["max_len"] >= (L + 2)]
    return out if out else [min(specs, key=lambda s: abs(s.get("anchor_len", s["max_len"]) - L))]

def blend_weights(L: int, specs: List[Dict[str, Any]], policy="triangular") -> np.ndarray:
    if len(specs) == 1 or policy == "hard":
        w = np.zeros(len(specs)); w[np.argmin([abs(s.get("anchor_len", s["max_len"]) - L) for s in specs])] = 1.0; return w
    if policy == "inverse_dist":
        dist = np.array([abs(s.get("anchor_len", s["max_len"]) - L) for s in specs], dtype=np.float64); w = 1.0 / (dist + 1e-6); return w / w.sum()
    ws = [max(0.0, 1.0 - abs(L - float(s.get("anchor_len", s["max_len"]))) / (max(2.0, (s.get("window",(0,0))[1] - s.get("window",(0,0))[0])) / 2.0 + 1e-6)) for s in specs]
    w = np.array(ws, np.float64); return w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)

_proj_cache: Dict[str, np.ndarray] = {}
def _projs_for_specialist(spec_path: str, in_dim: int, out_dim: int, n: int) -> List[np.ndarray]:
    Ps = []
    for k in range(n):
        key = f"{spec_path}|{in_dim}|{out_dim}|{k}"
        if key in _proj_cache: Ps.append(_proj_cache[key]); continue
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        P = rng.normal(0.0, 1.0/np.sqrt(max(1,in_dim)), size=(in_dim,out_dim)).astype(np.float32)
        _proj_cache[key] = P; Ps.append(P)
    return Ps

# ===================== Feature extraction =====================
def features_for_sequence(seq: str, specialists: List[Dict[str, Any]], proj_dim: int, blend: str, proj_bag: int) -> Dict[str, Any]:
    L = len(seq)
    specs = pick_applicable_specialists(specialists, L)
    w = blend_weights(L, specs, blend) if specs else []

    scalars = {"mean_nll": [], "perplexity": [], "entropy": [], "margin": []}
    embeds = []
    for wi, s in zip(w, specs):
        arr = pad_and_pack(s["processor"], list(seq), s["max_len"])
        inp = jnp.asarray(arr)
        tv = encode_compact(s['params']['encoder'], inp)
        logits = decode_teacher_forcing_compact(s['params'], tv, inp[:-1])

        tgt, mask = inp[1:], (inp[1:] != s["processor"].vocab['<PAD>'])
        logp = jax.nn.log_softmax(logits, axis=-1)
        mean_nll = -float((logp[jnp.arange(len(tgt)), tgt] * mask).sum() / mask.sum())

        p = jnp.exp(logp); ent = -jnp.sum(p * logp, axis=-1)
        top2 = jax.lax.top_k(logits, 2)[0]

        scalars["mean_nll"].append(wi * mean_nll)
        scalars["perplexity"].append(wi * np.exp(mean_nll))
        scalars["entropy"].append(wi * float(ent[:L].mean()))
        scalars["margin"].append(wi * float((top2[:L,0]-top2[:L,1]).mean()))

        tv_np = np.array(tv, dtype=np.float32)
        for P in _projs_for_specialist(s["path"], tv_np.shape[0], proj_dim, n=proj_bag):
            emb = tv_np @ P; emb /= max(1e-9, float(np.linalg.norm(emb)))
            embeds.append(wi * emb)

    embed_vec = np.mean(np.stack(embeds, axis=0), axis=0).astype(np.float32) if embeds else np.zeros((proj_dim,), np.float32)

    sol_sc, stab_sc, agg_sc = admet_scores_from_heuristics(seq)
    muH = helical_moment(seq)
    amph = amphipathic_index(seq)
    nC, nC_pairs, c_gap = cysteine_features(seq)
    motifs = motif_hits(seq)

    return {
        "len": float(L), "embed": embed_vec,
        "mean_nll": float(np.sum(scalars["mean_nll"])),
        "perplexity": float(np.sum(scalars["perplexity"])),
        "entropy": float(np.sum(scalars["entropy"])),
        "margin": float(np.sum(scalars["margin"])),
        "comp20": aa_composition(seq),
        "gravy": gravy(seq),
        "net_charge": net_charge_pH7(seq),
        "aromatic_frac": aromatic_fraction(seq),
        "hydro_frac": hydro_fraction(seq),
        "max_hydrophobic_run": float(max_hydrophobic_run(seq)),
        "solubility_score": sol_sc,
        "protease_stability_score": stab_sc,
        "aggregation_score": agg_sc,
        "muH": float(muH),
        "amphipathic_idx": float(amph),
        "cys_count": int(nC),
        "cys_pairs": int(nC_pairs),
        "cys_mean_gap": float(c_gap),
        "motif_hits": int(motifs),
    }

def feat_vector(feat: Dict[str, Any], mode: str = "fuse") -> np.ndarray:
    scalars = np.array([
        feat["mean_nll"], feat["perplexity"], feat["entropy"], feat["margin"], feat["len"],
        feat["muH"], feat["amphipathic_idx"], feat["cys_count"], feat["cys_pairs"], feat["cys_mean_gap"], feat["motif_hits"]
    ], dtype=np.float32)
    return {
        "embed": feat["embed"],
        "scalars": np.concatenate([scalars, feat["comp20"]]),
        "fuse": np.concatenate([feat["embed"], feat["comp20"], scalars]),
    }[mode]

# ===================== Heads (load/save/utils) =====================
def head_path(task_name, heads_dir): return os.path.join(heads_dir, f"{task_name}.joblib")
def have_head(task_name, heads_dir): return os.path.exists(head_path(task_name, heads_dir))
def load_head(task_name, heads_dir): return joblib.load(head_path(task_name, heads_dir))
def save_head(task_name, payload, heads_dir): os.makedirs(heads_dir, exist_ok=True); joblib.dump(payload, head_path(task_name, heads_dir))

# ===================== Guards, OOD, ECE =====================
def add_guard_heuristics(out_df: pd.DataFrame) -> pd.DataFrame:
    L = out_df["length"].to_numpy(float)
    charge = out_df["net_charge_heur"].to_numpy(float)
    hydro  = out_df["hydro_frac_heur"].to_numpy(float)
    arom   = out_df["aromatic_frac_heur"].to_numpy(float)
    maxrun = out_df["max_hydro_run_heur"].to_numpy(float)

    g_len  = (L < 8) | (L > 60)
    rho    = charge / np.maximum(L, 1.0)
    g_cpp  = (rho >= 0.60) & (hydro <= 0.20) & (arom <= 0.20)
    g_hyd  = (hydro >= 0.90) & (maxrun >= 8)

    guard  = g_len | g_cpp | g_hyd
    reason = np.where(g_len, "length",
              np.where(g_cpp, "cpp_trap",
              np.where(g_hyd, "hydro_trap","")))
    out_df["GUARD_flag"] = guard
    out_df["GUARD_reason"] = reason
    out_df["charge_density"] = rho
    return out_df

def add_ood_flags(out_df: pd.DataFrame) -> pd.DataFrame:
    L = out_df["length"].to_numpy(float)
    ood = (L < 8) | (L > 60)
    out_df["OOD_flag"] = ood
    out_df["OOD_reason"] = np.where(ood, "length", "")
    return out_df

def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0,1,n_bins+1); ece=0.0; N=len(y_true)
    for i in range(n_bins):
        idx = (p >= bins[i]) & (p < bins[i+1])
        if np.any(idx):
            conf = np.mean(p[idx])
            acc = np.mean(y_true[idx])
            ece += (np.sum(idx) / N) * abs(acc - conf)
    return float(ece)

# ===================== Grouping =====================
def peptide_groups(seqs: List[str]) -> np.ndarray:
    def sig(s):
        if len(s) < 3: return hashlib.sha1(("*"+s).encode()).hexdigest()[:8]
        kmers = {s[i:i+3] for i in range(len(s)-2)}
        return hashlib.sha1("|".join(sorted(kmers)).encode()).hexdigest()[:8]
    return np.array([sig(s) for s in seqs])

# ===================== Thresholding & Conformal =====================
def tune_threshold_for_precision(y_true, p, target_prec=0.80) -> float:
    prec, rec, thr = precision_recall_curve(y_true, p)
    thr = np.r_[thr, 1.0]
    ok = np.where(prec >= target_prec)[0]
    return float(thr[ok[0]]) if len(ok) else 0.5

def conformal_margin(p_cal, alpha=0.10) -> float:
    m = np.abs(p_cal - 0.5)
    return float(np.quantile(m, alpha))

# ===================== Δ-MutScan helpers =====================
def single_mutants(seq: str, mode: str = "fast", budget: int = 60):
    muts=[]
    letters=list(seq)
    if mode=="none":
        return muts
    if mode=="full":
        for i,ch in enumerate(letters):
            for aa in AA20:
                if aa!=ch:
                    muts.append(("{}{}{}".format(seq[:i],aa,seq[i+1:]), i, ch, aa))
        return muts
    # fast
    for i,ch in enumerate(letters):
        cand = AA_NEIGHBORS.get(ch, [])
        for aa in cand:
            if aa!=ch: muts.append(("{}{}{}".format(seq[:i],aa,seq[i+1:]), i, ch, aa))
    if len(muts)>budget:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(muts), size=budget, replace=False)
        muts = [muts[j] for j in idx]
    return muts

def probs_for_sequences(seqs, specialists, proj_dim, blend, proj_bag, head_pipe, feat_mode):
    feats = [features_for_sequence(s, specialists, proj_dim, blend, proj_bag) for s in seqs]
    X = np.stack([feat_vector(f, feat_mode) for f in feats], axis=0)
    return head_pipe.predict_proba(X)[:,1]

# ===================== PREDICT =====================
def run_predict(args):
    specs = load_all_specialists(); print(f"Loaded {len(specs)} specialists.")

    if args.sequence:
        seqs = [args.sequence]
    elif args.input_csv:
        df_in = pd.read_csv(args.input_csv)
        if args.seq_col not in df_in.columns:
            raise RuntimeError(f"seq_col '{args.seq_col}' not in CSV.")
        seqs = df_in[args.seq_col].tolist()
    else:
        raise RuntimeError("Provide --sequence or --input_csv.")

    clean_seqs = [s.strip().upper() for s in seqs if isinstance(s, str) and re.fullmatch(rf"[{AA20}]+", s.strip().upper())]
    if not clean_seqs: raise RuntimeError("No valid AA20 sequences found.")

    feats = [features_for_sequence(s, specs, args.proj_dim, args.blend_policy, args.proj_bag) for s in clean_seqs]
    X_by_mode = {m: np.stack([feat_vector(f, m) for f in feats], axis=0) for m in ["embed", "scalars", "fuse"]}

    # Core dataframe
    out = pd.DataFrame([{k: v for k, v in f.items() if k not in ("embed","comp20")} for f in feats],
                       index=clean_seqs).reset_index().rename(columns={"index":"sequence"})
    out = out.rename(columns={
        "len":"length",
        "gravy":"gravy_heur",
        "net_charge":"net_charge_heur",
        "aromatic_frac":"aromatic_frac_heur",
        "hydro_frac":"hydro_frac_heur",
        "max_hydrophobic_run":"max_hydro_run_heur",
        "perplexity":"LM_perplexity",
        "entropy":"LM_entropy",
        "margin":"LM_margin",
        "solubility_score":"Solubility_score",
        "protease_stability_score":"ProteaseStability_score",
        "aggregation_score":"Aggregation_score",
        "muH":"muH_helix",
        "amphipathic_idx":"amphipathic_idx",
        "cys_count":"Cys_count",
        "cys_pairs":"Cys_pairs",
        "cys_mean_gap":"Cys_mean_gap",
        "motif_hits":"motif_hits",
    })

    out["AMP_proxy"] = [amp_activity_proxy_from_heuristics(s) for s in out["sequence"]]
    if len(out) > 1:
        for sn in ["Solubility_score","ProteaseStability_score","Aggregation_score"]:
            raw = out[sn].to_numpy(np.float64)
            out[sn.replace("_score","_proxy")] = (raw - raw.mean()) / (raw.std() + 1e-9)

    out = add_guard_heuristics(out)
    out = add_ood_flags(out)

    # ===== HEADS (classification + regression) =====
    if args.tier == "premium":
        for fname in sorted(glob.glob(os.path.join(args.heads_dir, "*.joblib"))):
            head_name = os.path.splitext(os.path.basename(fname))[0]
            try:
                head = joblib.load(fname)
            except Exception as e:
                print(f"⚠️ Failed loading head {head_name}: {e}")
                continue

            ttype = head.get("task_type", "classification")
            feat_mode = head.get("feat_mode", "fuse")
            X = X_by_mode.get(feat_mode, X_by_mode["fuse"])
            pipe = head["pipeline"]

            if ttype == "classification":
                metrics = head.get("metrics_cv", {})
                ece_cv = metrics.get("ECE", None)
                min_n = metrics.get("n_samples", None)
                if not args.force_unreliable_heads:
                    if (ece_cv is not None) and (ece_cv > args.max_ece):
                        print(f"⚠️ Skipping head {head_name}: ECE={ece_cv:.3f} > {args.max_ece:.2f}.")
                        continue
                    if (min_n is not None) and (min_n < max(args.min_train_samples, 30)):
                        print(f"⚠️ Skipping head {head_name}: n={min_n} too small.")
                        continue

                prob = pipe.predict_proba(X)[:,1]
                thr_auto = head.get("threshold", 0.5)
                thr = args.thr_override if args.thr_override is not None else thr_auto
                out[f"{head_name}_prob"] = prob

                if not args.no_labels:
                    if args.no_borderline:
                        dec = np.where(prob >= thr, "positive","negative")
                    else:
                        dec = np.where(np.abs(prob - thr) <= args.label_margin, "borderline",
                                       np.where(prob >= thr, "positive","negative"))
                    if args.decision_policy == "conformal":
                        qhat = head.get("conformal_margin", None)
                        if (qhat is not None) and (not args.ignore_conformal):
                            band = max(qhat * args.qhat_mult, args.label_margin)
                            undecided = np.abs(prob - thr) < band
                            dec = np.where(undecided, "abstain", dec)
                    elif args.decision_policy == "margin_only":
                        undecided = np.abs(prob - thr) < max(1e-9, args.label_margin)
                        dec = np.where(undecided, "abstain", dec)
                    if getattr(args, "abstain_on_guard", False) and not getattr(args, "ignore_guard", False):
                        dec = np.where(out["GUARD_flag"].to_numpy(bool), "abstain", dec)

                    out[f"{head_name}_label"] = (prob >= thr).astype(int)
                    out[f"{head_name}_decision"] = dec

                # ========= Δ-MutScan (robustness) =========
                if args.mutscan != "none":
                    robust_cols = ["Robust_min_drop","Robust_worst_prob","Robust_flip_count",
                                   "Robust_worst_mut","Robust_worst_pos","Robust_worst_from","Robust_worst_to"]
                    for c in robust_cols:
                        if c not in out.columns:
                            if c in {"Robust_worst_mut", "Robust_worst_from", "Robust_worst_to"}:
                                out[c] = pd.Series([""] * len(out), dtype="object")
                            else:
                                out[c] = np.nan

                    for idx, seq in enumerate(out["sequence"].tolist()):
                        muts = single_mutants(seq, mode=args.mutscan, budget=args.mutscan_budget)
                        if not muts: continue
                        mut_seqs = [m[0] for m in muts]
                        mut_probs = probs_for_sequences(mut_seqs, specs, args.proj_dim, args.blend_policy,
                                                        args.proj_bag, pipe, feat_mode)
                        base_p = prob[idx]
                        drops = base_p - mut_probs
                        worst_j = int(np.argmax(drops))
                        out.at[idx, "Robust_min_drop"]   = float(np.max(drops))
                        out.at[idx, "Robust_worst_prob"] = float(mut_probs[worst_j])
                        out.at[idx, "Robust_flip_count"] = int(np.sum(mut_probs < thr))
                        wmut, pos, fr, to = muts[worst_j]
                        out.at[idx, "Robust_worst_mut"]  = wmut
                        out.at[idx, "Robust_worst_pos"]  = int(pos)
                        out.at[idx, "Robust_worst_from"] = fr
                        out.at[idx, "Robust_worst_to"]   = to

                        if not args.no_labels and (args.decision_policy != "none"):
                            cur = out.at[idx, f"{head_name}_decision"]
                            if cur != "abstain":
                                if (float(np.max(drops)) >= args.mutscan_abstain_drop) or (int(np.sum(mut_probs < thr)) >= 1):
                                    out.at[idx, f"{head_name}_decision"] = "abstain"

            elif ttype == "regression":
                try:
                    yhat = pipe.predict(X)
                    out[f"{head_name}_pred"] = yhat.astype(float)
                    # Safety clip (units e.g., µM) — adjust if needed:
                    out[f"{head_name}_pred"] = np.clip(out[f"{head_name}_pred"], 0.0, 1e9)
                except Exception as e:
                    print(f"⚠️ Regression head {head_name} failed: {e}")
            else:
                print(f"⏭️  Unknown head type for {head_name}: {ttype}")

    # ----- Optional: apply adapter calibration -----
    if getattr(args, "adapter_json", None):
        try:
            with open(args.adapter_json, "r") as f:
                adapter_cfg = json.load(f)
            any_prob_cols = [c for c in out.columns if c.endswith("_prob")]
            head_hint = any_prob_cols[0][:-5] if any_prob_cols else None
            out = apply_adapter_to_df(out, adapter_cfg, head_hint or "")
            print("✅ Applied adapter calibration from", args.adapter_json)
        except Exception as e:
            print(f"⚠️ Adapter load/apply failed: {e}")

    # ----- Optional: meta-head (mini-stacking) -----
    if getattr(args, "meta_joblib", None):
        try:
            meta = joblib.load(args.meta_joblib)
            feats = [s.strip() for s in (args.meta_features or "").split(",") if s.strip()]
            if not feats:
                prob_cols = [c for c in out.columns if c.endswith("_prob_cal")] or [c for c in out.columns if c.endswith("_prob")]
                base_prob = prob_cols[0] if prob_cols else None
                candidate_feats = [base_prob, "length", "net_charge_heur", "hydro_frac_heur", "AMP_proxy"]
                feats = [c for c in candidate_feats if (c is not None) and (c in out.columns)]
                print("ℹ️ meta_features (auto):", feats)
            X_meta = out[feats].astype(float).to_numpy()
            p_meta = meta.predict_proba(X_meta)[:,1]
            name = args.meta_outname
            out[f"{name}_prob"] = p_meta
            thrm = float(args.meta_threshold)
            out[f"{name}_label"] = (p_meta >= thrm).astype(int)
            dec = np.where(p_meta >= thrm, "positive", "negative")
            out[f"{name}_decision"] = dec
            print(f"✅ Applied meta-head from {args.meta_joblib} with feats={feats} thr={thrm}")
        except Exception as e:
            print(f"⚠️ Meta-head failed: {e}")

    # Save/print
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        out.to_csv(args.out_csv, index=False)
        print(f"✅ Saved predictions to {args.out_csv}")
    else:
        print("Preview:\n", out.head(min(12, len(out))).to_string(index=False))


# ===================== TRAIN (classification only) =====================
def _parse_class_labels(y):
    y = pd.Series(y)
    try:
        yn = pd.to_numeric(y, errors="coerce")
        unique = np.unique(yn.dropna())
        if set(unique).issubset({0,1}):
            return yn.fillna(0).astype(int).to_numpy()
    except Exception:
        pass
    y = y.astype(str).str.strip().str.lower()
    return y.isin({"1", "true", "yes", "active", "pos", "positive"}).astype(int).to_numpy()

def run_train(args):
    specs = load_all_specialists(); print(f"Loaded {len(specs)} specialists.")
    df = pd.read_csv(args.dataset_csv)

    # --- Clean sequences ---
    if args.seq_col not in df.columns or args.label_col not in df.columns:
        raise RuntimeError(f"Missing columns: need '{args.seq_col}' and '{args.label_col}'.")
    df = df.dropna(subset=[args.seq_col, args.label_col]).copy()
    df[args.seq_col] = df[args.seq_col].astype(str).str.strip().str.upper()
    df = df[df[args.seq_col].str.fullmatch(rf"[{AA20}]+", na=False)].copy()
    df = df.drop_duplicates(subset=[args.seq_col]).reset_index(drop=True)

    if len(df) < args.min_train_samples:
        raise RuntimeError(f"Too few valid samples to train (found {len(df)}, need ≥{args.min_train_samples}).")

    # --- Feature extraction ---
    feats = [features_for_sequence(s, specs, args.proj_dim, args.blend_policy, args.proj_bag) for s in df[args.seq_col]]
    X = np.stack([feat_vector(f, args.feat_mode) for f in feats], axis=0)

    # --- Groups for leakage-safe CV ---
    groups = peptide_groups(df[args.seq_col].tolist())

    summary = {
        "task": args.task_name, "task_type": "classification", "feat_mode": args.feat_mode,
        "proj_dim": args.proj_dim, "blend_policy": args.blend_policy, "proj_bag": args.proj_bag,
        "n_samples": int(len(df)), "timestamp": time.time(), "precision_target": args.precision_target,
        "conformal_alpha": args.conformal_alpha, "cv_folds": args.cv_folds
    }

    y = _parse_class_labels(df[args.label_col])

    gkf = GroupKFold(n_splits=args.cv_folds)
    fold_stats, thr_list, eces = [], [], []

    base = LogisticRegression(
        penalty="l2", solver="saga", max_iter=5000, C=1.0,
        class_weight="balanced", random_state=RNG_SEED
    )

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        # nested: split train into train_in / train_cal (80/20)
        rng = np.random.default_rng(RNG_SEED + fold)
        idx = rng.permutation(tr)
        cut = int(0.8 * len(idx))
        tr_in, tr_cal = idx[:cut], idx[cut:] if cut < len(idx) else (idx, idx)

        calibrator = "sigmoid" if args.calibration in ("platt", "sigmoid", "isotonic") else None
        if args.calibration == "isotonic" and len(tr_cal) < 200:
            calibrator = "sigmoid"
        if calibrator is not None:
            clf = CalibratedClassifierCV(estimator=base, method=calibrator, cv=3)
        else:
            clf = base
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X[tr_in], y[tr_in])

        # tune threshold on tr_cal for target precision
        p_cal = pipe.predict_proba(X[tr_cal])[:,1] if len(tr_cal) > 0 else pipe.predict_proba(X[tr_in])[:,1]
        thr_fold = tune_threshold_for_precision(y[tr_cal] if len(tr_cal)>0 else y[tr_in], p_cal, args.precision_target)
        thr_list.append(thr_fold)
        qhat = conformal_margin(p_cal, args.conformal_alpha)

        p_te = pipe.predict_proba(X[te])[:,1]
        f1 = f1_score(y[te], (p_te >= thr_fold).astype(int))
        auroc = roc_auc_score(y[te], p_te)
        auprc = average_precision_score(y[te], p_te)
        brier = brier_score_loss(y[te], p_te)
        ece = ece_score(y[te], p_te, n_bins=10)
        eces.append(ece)

        fold_stats.append({
            "fold": fold, "AUROC": float(auroc), "AUPRC": float(auprc), "F1@thr": float(f1),
            "Brier": float(brier), "best_thr_prec_target": float(thr_fold), "ECE": float(ece),
            "qhat": float(qhat), "n_tr_in": int(len(tr_in)), "n_tr_cal": int(len(tr_cal)), "n_te": int(len(te))
        })

    metrics = {
        "task": args.task_name, "type": "classification", "feat_set": args.feat_mode,
        "calibration": args.calibration,
        "AUROC": float(np.mean([m["AUROC"] for m in fold_stats])),
        "AUPRC": float(np.mean([m["AUPRC"] for m in fold_stats])),
        "F1@thr": float(np.mean([m["F1@thr"] for m in fold_stats])),
        "Brier": float(np.mean([m["Brier"] for m in fold_stats])),
        "best_thr_prec_target": float(np.mean(thr_list)),
        "ECE": float(np.mean(eces)),
        "folds": fold_stats, "n_samples": int(len(df))
    }
    print("CV metrics:", json.dumps(metrics, indent=2))

    # full fit + calibration
    calibrator = "sigmoid" if args.calibration in ("platt", "sigmoid", "isotonic") else None
    if args.calibration == "isotonic" and len(df) < 500:
        calibrator = "sigmoid"
    if calibrator is not None:
        clf_full = CalibratedClassifierCV(estimator=base, method=calibrator, cv=5)
    else:
        clf_full = base
    pipe_full = Pipeline([("scaler", StandardScaler()), ("clf", clf_full)]).fit(X, y)
    p_full = pipe_full.predict_proba(X)[:,1]
    thr_final = tune_threshold_for_precision(y, p_full, args.precision_target)
    qhat_final = conformal_margin(p_full, args.conformal_alpha)

    payload = {
        "task_type": "classification",
        "pipeline": pipe_full,
        "feat_mode": args.feat_mode,
        "threshold": float(thr_final),
        "conformal_margin": float(qhat_final),
        "metrics_cv": metrics
    }
    save_head(args.task_name, payload, args.heads_dir)

    with open(os.path.join(args.heads_dir, f"{args.task_name}.summary.json"), "w") as f:
        json.dump({"task": args.task_name, **metrics, "threshold": float(thr_final), "qhat": float(qhat_final)}, f, indent=2)

    print(f"✅ Saved head → {head_path(args.task_name, args.heads_dir)}")


# ===================== TRAIN (regression; e.g., MIC) =====================
def run_train_regression(args):
    specs = load_all_specialists(); print(f"Loaded {len(specs)} specialists.")
    df = pd.read_csv(args.dataset_csv)
    if args.seq_col not in df.columns or args.target_col not in df.columns:
        raise RuntimeError(f"Missing columns: need '{args.seq_col}' and '{args.target_col}'.")

    # Clean
    df = df.dropna(subset=[args.seq_col, args.target_col]).copy()
    df[args.seq_col] = df[args.seq_col].astype(str).str.strip().str.upper()
    df = df[df[args.seq_col].str.fullmatch(rf"[{AA20}]+", na=False)].copy()
    df = df.drop_duplicates(subset=[args.seq_col]).reset_index(drop=True)

    # Features
    feats = [features_for_sequence(s, specs, args.proj_dim, args.blend_policy, args.proj_bag) for s in df[args.seq_col]]
    X = np.stack([feat_vector(f, args.feat_mode) for f in feats], axis=0)
    y = pd.to_numeric(df[args.target_col], errors="coerce").to_numpy(float)

    groups = peptide_groups(df[args.seq_col].tolist())
    gkf = GroupKFold(n_splits=args.cv_folds)

    maes, r2s = [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([("s", StandardScaler()), ("m", HuberRegressor())])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        maes.append(mean_absolute_error(y[te], pred))
        r2s.append(r2_score(y[te], pred))
    metrics = {"MAE_cv": float(np.mean(maes)), "R2_cv": float(np.mean(r2s)), "n_samples": int(len(df))}
    print("CV metrics (regression):", json.dumps(metrics, indent=2))

    pipe_full = Pipeline([("s", StandardScaler()), ("m", HuberRegressor())]).fit(X, y)
    payload = {"task_type": "regression", "pipeline": pipe_full, "feat_mode": args.feat_mode, "metrics_cv": metrics}
    save_head(args.task_name, payload, args.heads_dir)
    with open(os.path.join(args.heads_dir, f"{args.task_name}.summary.json"), "w") as f:
        json.dump({"task": args.task_name, **metrics, "timestamp": time.time()}, f, indent=2)
    print(f"✅ Saved regression head → {head_path(args.task_name, args.heads_dir)}")


# ===================== main =====================
def main():
    ap = argparse.ArgumentParser(description="EvoTensor Peptide Product — AMP + MIC (classification + regression) with Δ-MutScan")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Predict CLI
    p_pred = sub.add_parser("predict", help="Predict on a sequence or CSV")
    p_pred.add_argument("--tier", default=os.environ.get("TIER", "premium"), choices=["free", "premium"])
    p_pred.add_argument("--input_csv", help="CSV with sequences")
    p_pred.add_argument("--sequence", help="Single peptide (AA20)")
    p_pred.add_argument("--seq_col", default="sequence")
    p_pred.add_argument("--out_csv")
    p_pred.add_argument("--heads_dir", default=HEADS_DIR)
    p_pred.add_argument("--proj_dim", type=int, default=PROJ_DIM)
    p_pred.add_argument("--blend_policy", default=BLEND_POLICY, choices=["triangular", "inverse_dist", "hard"])
    p_pred.add_argument("--proj_bag", type=int, default=PROJ_BAG)
    # label/threshold controls
    p_pred.add_argument("--no_labels", action="store_true",
                        help="Only output probabilities; do not emit *_label/decision.")
    p_pred.add_argument("--thr_override", type=float, default=None,
                        help="Override classification threshold in [0,1]. If unset, uses tuned threshold.")
    p_pred.add_argument("--label_margin", type=float, default=0.15,
                        help="±band γύρω από το threshold που χαρακτηρίζεται 'borderline'.")
    # reliability-gating controls
    p_pred.add_argument("--force_unreliable_heads", action="store_true",
                        help="Bypass ECE/n gating and run heads anyway (for benchmarking).")
    p_pred.add_argument("--max_ece", type=float, default=0.15,
                        help="ECE threshold for reliability gate.")
    p_pred.add_argument("--min_train_samples", type=int, default=MIN_TRAIN_SAMPLES,
                        help="Minimum samples required to accept a head (gating).")
    # decision policy
    p_pred.add_argument("--decision_policy", default="conformal",
                        choices=["conformal","margin_only","none"],
                        help="Πολιτική abstention: conformal | margin_only | none.")
    p_pred.add_argument("--qhat_mult", type=float, default=1.0,
                        help="Multiplier στο conformal qhat πριν συγκριθεί με το label_margin.")
    p_pred.add_argument("--no_borderline", action="store_true",
                        help="Απενεργοποίηση ετικέτας 'borderline' (μένει μόνο positive/negative/abstain).")
    # Guard/Conformal ignores
    p_pred.add_argument("--ignore_conformal", action="store_true",
                        help="Απενεργοποιεί abstention από conformal margin (χρήσιμο για dry-runs).")
    p_pred.add_argument("--ignore_guard", action="store_true",
                        help="Απενεργοποιεί abstention από guard rules (length/CPP/hydrophobic).")
    # Δ-MutScan
    p_pred.add_argument("--mutscan", default="fast", choices=["none","fast","full"],
                        help="Robustness scan over single-aa mutants.")
    p_pred.add_argument("--mutscan_budget", type=int, default=60,
                        help="Max mutants per sequence for fast scan.")
    p_pred.add_argument("--mutscan_abstain_drop", type=float, default=0.20,
                        help="Abstain αν worst drop≥this ή υπάρχει οποιοδήποτε flip κάτω από threshold.")
    p_pred.add_argument("--abstain_on_guard", action="store_true",
                        help="Αν είναι on, κάνε abstain όταν ενεργοποιούνται guard rules (length/CPP/hydrophobic).")
    # --- Optional adapters / meta-head (post-process) ---
    p_pred.add_argument("--adapter_json", type=str, default=None,
                        help="Μικρό Platt adapter JSON με κλειδιά a,b[,head,thr]. Εφαρμόζεται στα *_prob.")
    p_pred.add_argument("--meta_joblib", type=str, default=None,
                        help="Joblib αρχείο ενός ελαφρού meta-classifier που δέχεται features από το output.")
    p_pred.add_argument("--meta_features", type=str, default="",
                        help="Comma-separated λίστα στηλών (features) από το output CSV για το meta-head.")
    p_pred.add_argument("--meta_outname", type=str, default="meta",
                        help="Prefix για τα output columns του meta-head (π.χ. 'meta').")
    p_pred.add_argument("--meta_threshold", type=float, default=0.5,
                        help="Threshold για το meta-head.")
    p_pred.set_defaults(func=run_predict)

    # Train CLI (CLASSIFICATION ONLY)
    p_train = sub.add_parser("train", help="Train a classification head from CSV")
    p_train.add_argument("--task_name", required=True)
    p_train.add_argument("--dataset_csv", required=True)
    p_train.add_argument("--seq_col", default="sequence")
    p_train.add_argument("--label_col", required=True)
    p_train.add_argument("--feat_mode", default="fuse", choices=["embed", "scalars", "fuse"])
    p_train.add_argument("--calibration", default="sigmoid", choices=["none", "platt", "sigmoid", "isotonic"])
    p_train.add_argument("--heads_dir", default=HEADS_DIR)
    p_train.add_argument("--proj_dim", type=int, default=PROJ_DIM)
    p_train.add_argument("--blend_policy", default=BLEND_POLICY, choices=["triangular", "inverse_dist", "hard"])
    p_train.add_argument("--proj_bag", type=int, default=PROJ_BAG)
    p_train.add_argument("--min_train_samples", type=int, default=MIN_TRAIN_SAMPLES)
    p_train.add_argument("--precision_target", type=float, default=PRECISION_TARGET)
    p_train.add_argument("--conformal_alpha", type=float, default=CONFORMAL_ALPHA)
    p_train.add_argument("--cv_folds", type=int, default=CV_FOLDS)
    p_train.set_defaults(func=run_train)

    # Train CLI (REGRESSION: e.g., MIC)
    p_train_r = sub.add_parser("train_reg", help="Train a regression head (e.g., MIC) from CSV")
    p_train_r.add_argument("--task_name", required=True)
    p_train_r.add_argument("--dataset_csv", required=True)
    p_train_r.add_argument("--seq_col", default="sequence")
    p_train_r.add_argument("--target_col", required=True, help="Numeric target column (e.g., MIC_uM)")
    p_train_r.add_argument("--feat_mode", default="fuse", choices=["embed", "scalars", "fuse"])
    p_train_r.add_argument("--heads_dir", default=HEADS_DIR)
    p_train_r.add_argument("--proj_dim", type=int, default=PROJ_DIM)
    p_train_r.add_argument("--blend_policy", default=BLEND_POLICY, choices=["triangular", "inverse_dist", "hard"])
    p_train_r.add_argument("--proj_bag", type=int, default=PROJ_BAG)
    p_train_r.add_argument("--cv_folds", type=int, default=CV_FOLDS)
    p_train_r.set_defaults(func=run_train_regression)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
