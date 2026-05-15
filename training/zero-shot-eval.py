%%writefile eval_yap1_specialist_ensemble.py

# -*- coding: utf-8 -*-
"""
YAP1 WW (Araya 2012) — Evaluation for Evotensor specialist ensemble.

This script:
  - Loads multiple generative specialist models (len30/40/52, etc.)
  - Scores each mutant with every specialist:
        * -ΔlogP (full-sequence negative delta log-prob)
        * -site log-odds (local, WT context)
  - Builds an ensemble by averaging predictors across specialists.
  - Computes:
        * Spearman / Pearson + 95% bootstrap CI
        * Macro Spearman across positions (n >= 5)
        * Within-position z-scored correlations
  - Produces:
        * predictions_yap1_specialists.csv
        * metrics_yap1_specialists.json
        * scatter / histogram / z-scatter plots for the ensemble

Important:
  - Only spans up to length 52 are supported. If the inferred WT span
    is longer, we center-crop to 52 residues around the mutated site.
"""

import os, re, json, time, pickle, argparse
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# ===================== GLOBAL CONFIG =====================

class DataProcessor:
    """Minimal shim: αρκεί για unpickle + vocab + packing."""
    def sequence_to_indices(self, seq):
        vocab = getattr(self, "vocab", {}) or {}
        unk = vocab.get("<UNK>", 0)
        return np.asarray([vocab.get(tok, unk) for tok in seq], dtype=np.int32)

FITNESS_HIGHER_IS_BETTER = True
BOOTSTRAP_N = 2000
RNG_SEED = 0
MIN_PERPOS_N = 5
MAX_SUPPORTED_LEN = 52          # you said: trained up to 52 aa

WT_SEQUENCE_OVERRIDE: Optional[str] = None  # can stay None for YAP1


# ===================== UTILS =====================

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

def infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None

def bootstrap_ci(x: np.ndarray, y: np.ndarray, stat_fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3:
        return np.nan, (np.nan, np.nan)
    base = stat_fn(x, y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        vals.append(stat_fn(x[idx], y[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(base), (float(lo), float(hi))

def zscore(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64)
    return (v - v.mean()) / (v.std(ddof=0) + 1e-9)


# ===================== MODEL FORMAT DETECTION =====================

def is_compact(params):
    try:
        return ("encoder" in params
                and isinstance(params["encoder"], dict)
                and "mps"  in params["encoder"]
                and "isos" in params["encoder"])
    except Exception:
        return False

def is_legacy(params):
    try:
        return ("encoder" in params
                and isinstance(params["encoder"], dict)
                and "mps_tensors"     in params["encoder"]
                and "mera_isometries" in params["encoder"])
    except Exception:
        return False


# ===================== COMPACT ENCODE/DECODE =====================

def _encode_step_compact(carry, token_and_pos):
    state, mps = carry
    tok, pos = token_and_pos
    W = mps[pos, tok]    # (chi, chi)
    state = state @ W
    return (state, mps), state

@jax.jit
def encode_compact(enc_params, seq_indices):
    L = enc_params["mps"].shape[0]
    chi = enc_params["mps"].shape[-1]
    state0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L, dtype=jnp.int32)
    (_, _), states = lax.scan(_encode_step_compact,
                              (state0, enc_params["mps"]),
                              (seq_indices, positions))

    def pair_reduce(_, i):
        l = states[2*i]
        r = states[2*i+1]
        iso = enc_params["isos"][i]               # (chi, chi, chi_m)
        hi = jnp.einsum("i,j,ijk->k", l, r, iso)  # (chi_m,)
        return None, hi

    _, hi_list = lax.scan(pair_reduce, None, jnp.arange(L//2))
    tv = hi_list.reshape(-1)
    return tv / (jnp.linalg.norm(tv) + 1e-9)

def _decode_step_compact(carry, tok_and_pos):
    vec, ctx, mps_dec, out_proj, first_flag = carry
    tok, pos = tok_and_pos
    W = mps_dec[pos, tok]
    vec = vec @ W
    vec = jnp.where(first_flag, vec + ctx, vec)
    logits = vec @ out_proj
    return (vec, ctx, mps_dec, out_proj, False), logits

@jax.jit
def decode_teacher_forcing_compact(params, thought_vector, decoder_input):
    L = params["decoder"]["mps"].shape[0]
    chi = params["decoder"]["mps"].shape[-1]
    ctx = thought_vector @ params["projection_matrix"]
    vec0 = jnp.zeros((chi,), dtype=jnp.float32).at[0].set(1.0)
    positions = jnp.arange(L-1, dtype=jnp.int32)
    mps_dec = params["decoder"]["mps"][:-1]
    (_, _, _, _, _), logits = lax.scan(
        _decode_step_compact,
        (vec0, ctx, mps_dec, params["output_projection"], True),
        (decoder_input, positions)
    )
    return logits


# ===================== LEGACY ENCODE/DECODE =====================

@partial(jax.jit, static_argnames=("bond_dim_mps","max_len"))
def encode_legacy(enc_params, seq_indices, bond_dim_mps, max_len):
    vec = jnp.zeros((bond_dim_mps,), dtype=jnp.float32).at[0].set(1.0)
    bonds = []
    for i in range(max_len):
        W = enc_params["mps_tensors"][i][seq_indices[i], :, :]
        vec = vec @ W
        bonds.append(vec)
    hi = []
    for i in range(0, max_len-1, 2):
        l, r = bonds[i], bonds[i+1]
        iso = enc_params["mera_isometries"][i//2]
        hi.append(jnp.einsum("i,j,ijk->k", l, r, iso))
    tv = jnp.concatenate(hi)
    return tv / (jnp.linalg.norm(tv)+1e-9)

@partial(jax.jit, static_argnames=("bond_dim_mps","max_len"))
def decode_teacher_forcing_legacy(params, thought_vector, decoder_input, bond_dim_mps, max_len):
    ctx = thought_vector @ params["projection_matrix"]
    vec = jnp.zeros((bond_dim_mps,), dtype=jnp.float32).at[0].set(1.0)
    logits_list = []
    for t in range(max_len-1):
        W = params["decoder"]["mps_tensors"][t][decoder_input[t], :, :]
        vec = vec @ W
        vec = jnp.where(t == 0, vec + ctx, vec)
        logits_list.append(vec @ params["output_projection"])
    return jnp.stack(logits_list, axis=0)


# ===================== TOKENIZATION / SCORING =====================

def pad_and_pack(processor, seq_chars, max_len):
    sos = processor.vocab["<SOS>"]
    eos = processor.vocab["<EOS>"]
    pad = processor.vocab["<PAD>"]
    seq = ["<SOS>"] + list(seq_chars)[:max_len-2] + ["<EOS>"]
    idx = processor.sequence_to_indices(seq)
    if len(idx) < max_len:
        idx = np.concatenate([idx, np.full((max_len-len(idx),), pad, np.int32)])
    return idx

def log_prob_sequence(params, processor, seq_chars, max_len, bond_dim_mps, compact):
    arr = pad_and_pack(processor, seq_chars, max_len)
    inp = jnp.asarray(arr)
    if compact:
        tv = encode_compact(params["encoder"], inp)
        dec_in = inp[:-1]
        logits = decode_teacher_forcing_compact(params, tv, dec_in)
    else:
        tv = encode_legacy(params["encoder"], inp, bond_dim_mps, max_len)
        dec_in = inp[:-1]
        logits = decode_teacher_forcing_legacy(params, tv, dec_in, bond_dim_mps, max_len)
    targets = inp[1:]
    pad_idx = processor.vocab["<PAD>"]
    mask = (targets != pad_idx)
    logp = jax.nn.log_softmax(logits, axis=-1)
    tok_logp = logp[jnp.arange(logp.shape[0]), targets]
    return float((tok_logp * mask).sum())

def site_log_odds(params, processor, wt_seq_str, pos_local_1b, mutant_aa,
                  max_len, bond_dim_mps, compact):
    seq = list(wt_seq_str.upper())
    if not (1 <= pos_local_1b <= len(seq)):
        return None
    arr = pad_and_pack(processor, seq, max_len)
    inp = jnp.asarray(arr)
    if compact:
        tv = encode_compact(params["encoder"], inp)
        logits = decode_teacher_forcing_compact(params, tv, inp[:-1])
    else:
        tv = encode_legacy(params["encoder"], inp, bond_dim_mps, max_len)
        logits = decode_teacher_forcing_legacy(params, tv, inp[:-1], bond_dim_mps, max_len)
    t = pos_local_1b
    logp = jax.nn.log_softmax(logits, axis=-1)
    wt_idx  = processor.vocab.get(seq[pos_local_1b-1], processor.vocab["<UNK>"])
    mut_idx = processor.vocab.get(mutant_aa.upper(),      processor.vocab["<UNK>"])
    return float(logp[t-1, mut_idx] - logp[t-1, wt_idx])


# ===================== METRICS & PLOTS =====================

def compute_block_metrics(df_block: pd.DataFrame, pred_col: str, tag: str,
                          out_dir: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "DMS_score" not in df_block.columns or len(df_block) < 3:
        return out

    x = df_block["DMS_score"].to_numpy(np.float64)
    y = df_block[pred_col].to_numpy(np.float64)
    if not FITNESS_HIGHER_IS_BETTER:
        x = -x

    sp, (sp_lo, sp_hi) = bootstrap_ci(
        x, y, lambda a, b: spearmanr(a, b)[0],
        n=BOOTSTRAP_N, seed=RNG_SEED
    )
    pe, (pe_lo, pe_hi) = bootstrap_ci(
        x, y, lambda a, b: pearsonr(a, b)[0],
        n=BOOTSTRAP_N, seed=RNG_SEED
    )
    out[f"{tag}_spearman"] = sp
    out[f"{tag}_spearman_ci95"] = [sp_lo, sp_hi]
    out[f"{tag}_pearson"] = pe
    out[f"{tag}_pearson_ci95"] = [pe_lo, pe_hi]

    # per-position Spearman
    macro = []
    for pos, g in df_block.groupby("pos"):
        if len(g) >= MIN_PERPOS_N:
            gx = g["DMS_score"].to_numpy(np.float64)
            gy = g[pred_col].to_numpy(np.float64)
            if not FITNESS_HIGHER_IS_BETTER:
                gx = -gx
            macro.append((pos, float(spearmanr(gx, gy)[0]), int(len(g))))
    if macro:
        macro = sorted(macro, key=lambda t: t[0])
        r_vals = np.array([r for _, r, _ in macro], dtype=np.float64)
        w      = np.array([n for *_, n in macro], dtype=np.float64)
        out[f"{tag}_macro_spearman_mean"]     = float(np.nanmean(r_vals))
        out[f"{tag}_macro_spearman_weighted"] = float(np.nansum(r_vals*w) /
                                                      max(np.nansum(w), 1.0))
        perpos_df = pd.DataFrame({
            "pos":      [p for p, _, _ in macro],
            "spearman": [r for _, r, _ in macro],
            "n":        [n for *_, n in macro],
        })
        perpos_df.to_csv(
            os.path.join(out_dir, f"per_position_spearman_{tag}.csv"),
            index=False
        )

    # within-position z
    dfz = df_block.copy()
    dfz["DMS_z"]  = dfz.groupby("pos")["DMS_score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0)+1e-9)
    )
    dfz["pred_z"] = dfz.groupby("pos")[pred_col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0)+1e-9)
    )
    out[f"{tag}_global_spearman_z"] = float(spearmanr(dfz["DMS_z"], dfz["pred_z"])[0])
    out[f"{tag}_global_pearson_z"]  = float(pearsonr(dfz["DMS_z"],  dfz["pred_z"])[0])
    return out


def scatter_plot(df_block, pred_col, title, fname, out_dir, annotate=None,
                 xlab_extra=" (higher=better)"):
    fig = plt.figure(figsize=(7.6, 7.6))
    plt.scatter(df_block["DMS_score"], df_block[pred_col],
                alpha=0.6, edgecolors="k", s=35)
    plt.xlabel("DMS score" + (xlab_extra if FITNESS_HIGHER_IS_BETTER else " (higher=worse)"))
    plt.ylabel(f"Model prediction ({pred_col})")
    plt.title(title)
    if annotate:
        plt.text(0.02, 0.98, annotate, transform=plt.gca().transAxes, va="top",
                 bbox=dict(boxstyle="round,pad=0.4", fc="wheat", alpha=0.85))
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(os.path.join(out_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def hist_plot(series, title, fname, out_dir, xlabel):
    fig = plt.figure(figsize=(7.6, 5.0))
    plt.hist(series, bins=40)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(os.path.join(out_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def zscatter_plot(df_block, pred_col, title, fname, out_dir):
    dfz = df_block.copy()
    dfz["DMS_z"]  = dfz.groupby("pos")["DMS_score"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0)+1e-9)
    )
    dfz["pred_z"] = dfz.groupby("pos")[pred_col].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0)+1e-9)
    )
    fig = plt.figure(figsize=(7.6, 7.6))
    plt.scatter(dfz["DMS_z"], dfz["pred_z"],
                alpha=0.6, edgecolors="k", s=30)
    plt.axhline(0, color="k", linewidth=1)
    plt.axvline(0, color="k", linewidth=1)
    plt.xlabel("DMS (per-position z-score)")
    plt.ylabel(f"{pred_col} (per-position z-score)")
    plt.title(title)
    plt.tight_layout()
    for ext in ("png", "svg"):
        plt.savefig(os.path.join(out_dir, f"{fname}.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ===================== LOAD SPECIALIST MODEL =====================

def load_specialist(path: str):
    """Load a generative specialist model pickle."""
    payload = pickle.load(open(path, "rb"))
    if isinstance(payload, tuple) and len(payload) >= 3:
        params, processor, max_len = payload[:3]
    elif isinstance(payload, dict):
        params = payload.get("params") or payload.get("base_params")
        processor = payload.get("processor")
        max_len = payload.get("model_len") or payload.get("max_len") \
                  or params["encoder"]["mps"].shape[0]
    else:
        raise RuntimeError(f"Unexpected pickle structure in {path}")

    if processor is None:
        raise RuntimeError(f"Processor missing from {path}")

    if is_compact(params):
        compact = True
        bond_dim_mps = params["encoder"]["mps"].shape[-1]
    elif is_legacy(params):
        compact = False
        bond_dim_mps = params["encoder"]["mps_tensors"][0].shape[-1]
    else:
        raise RuntimeError(f"Unknown encoder format in {path}")

    return {
        "path": path,
        "params": params,
        "processor": processor,
        "max_len": int(max_len),
        "bond_dim_mps": int(bond_dim_mps),
        "compact": bool(compact),
    }


# ===================== MAIN EVAL =====================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate Evotensor specialist ensemble on YAP1 WW DMS."
    )
    ap.add_argument(
        "--model", action="append", required=True,
        help="Specialist model path. Pass 3 times, e.g.: "
             "--model /kaggle/input/alltrainedmodels/.../specialist_win24-36_len30.pkl "
             "--model /kaggle/input/alltrainedmodels/.../specialist_win30-48_len40.pkl "
             "--model /kaggle/input/alltrainedmodels/.../specialist_win44-60_len52.pkl"
    )
    ap.add_argument(
        "--input_csv", required=True,
        help="YAP1 DMS CSV (must contain single mutants like A42G)."
    )
    ap.add_argument(
        "--out_dir", required=True,
        help="Output directory."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    # ---- load specialists ----
    specialists = [load_specialist(p) for p in args.model]
    print(f"[{now()}] Loaded {len(specialists)} specialist models:")
    for s in specialists:
        print(f"  - {os.path.basename(s['path'])}: max_len={s['max_len']} bond_dim={s['bond_dim_mps']}")

    max_model_len = max(s["max_len"] for s in specialists)
    if max_model_len > MAX_SUPPORTED_LEN:
        print(f"⚠️ Max model_len={max_model_len} > MAX_SUPPORTED_LEN={MAX_SUPPORTED_LEN}. "
              f"Sequences will still be cropped to {MAX_SUPPORTED_LEN}.")

    # ---- load DMS CSV ----
    print(f"[{now()}] ▶ Reading CSV: {args.input_csv}")
    df_raw = pd.read_csv(args.input_csv)
    print(f"Rows in file: {len(df_raw)} | Columns: {list(df_raw.columns)}")

    mut_col = infer_col(df_raw, ["mutant","mutation","variant","Mutant"])
    score_col = infer_col(df_raw, ["DMS_score","fitness","score","y"])
    if mut_col is None:
        raise RuntimeError(f"No mutation column found. Available: {list(df_raw.columns)}")

    AA20 = "ACDEFGHIKLMNPQRSTVWY"
    pat_single = re.compile(rf"^[{AA20}]\d+[{AA20}]$")

    df = df_raw.copy()
    df[mut_col] = df[mut_col].astype(str).str.strip().str.upper()
    df = df[df[mut_col].str.match(pat_single, na=False)].drop_duplicates(subset=[mut_col]).reset_index(drop=True)
    if score_col:
        df = df[[mut_col, score_col]].rename(columns={mut_col: "mutant", score_col: "DMS_score"})
    else:
        df = df[[mut_col]].rename(columns={mut_col: "mutant"})
    print(f"[filter] usable single substitutions: {len(df)}")

    # ---- build WT span (majority per position) ----
    wt_count: Dict[int, Dict[str, int]] = {}
    min_pos, max_pos = 10**9, -1
    for m in df["mutant"]:
        wt_aa = m[0]
        pos   = int(m[1:-1])
        min_pos = min(min_pos, pos)
        max_pos = max(max_pos, pos)
        wt_count.setdefault(pos, {}).setdefault(wt_aa, 0)
        wt_count[pos][wt_aa] += 1

    L = max_pos - min_pos + 1
    wt_seq_list = []
    for p in range(min_pos, max_pos+1):
        if p in wt_count:
            wt_aa = max(wt_count[p].items(), key=lambda kv: kv[1])[0]
            wt_seq_list.append(wt_aa)
        else:
            wt_seq_list.append("G")

    WT_SEQ = "".join(wt_seq_list)
    if WT_SEQUENCE_OVERRIDE is not None and len(WT_SEQUENCE_OVERRIDE) == len(WT_SEQ):
        WT_SEQ = WT_SEQUENCE_OVERRIDE
        print(f"[{now()}] ⚙️ Using WT override (len={len(WT_SEQ)}).")
    else:
        print(f"[{now()}] ✅ WT span constructed: {min_pos}-{max_pos} (len={L}).")

    if L > MAX_SUPPORTED_LEN:
        print(f"⚠️ WT span length {L} > MAX_SUPPORTED_LEN={MAX_SUPPORTED_LEN}. "
              f"Will center-crop windows of {MAX_SUPPORTED_LEN} around each mutation.")

    # ---- scoring loop ----
    print(f"[{now()}] ▶ Scoring mutants with specialist ensemble…")
    rows = []
    for _, row in df.iterrows():
        m = row["mutant"]
        pos_global = int(m[1:-1])
        pos_local_full = pos_global - min_pos + 1  # 1-based index in full WT_SEQ
        mut_aa = m[-1]

        # possibly crop WT_SEQ to a window of <= MAX_SUPPORTED_LEN around the site
        if len(WT_SEQ) <= MAX_SUPPORTED_LEN:
            wt_span = WT_SEQ
            pos_local = pos_local_full
        else:
            half = MAX_SUPPORTED_LEN // 2
            center_idx = pos_local_full - 1
            start = max(0, center_idx - half)
            end = min(len(WT_SEQ), start + MAX_SUPPORTED_LEN)
            # adjust start if we hit end
            start = max(0, end - MAX_SUPPORTED_LEN)
            wt_span = WT_SEQ[start:end]
            pos_local = pos_global - (min_pos + start) + 1  # 1-based in wt_span

        # ensemble accumulators
        dlogp_list = []
        slo_list   = []

        for s in specialists:
            params = s["params"]
            processor = s["processor"]
            max_len = s["max_len"]
            bond_dim = s["bond_dim_mps"]
            compact = s["compact"]

            # -ΔlogP with this specialist
            dlp_wt  = log_prob_sequence(params, processor, list(wt_span),
                                        max_len, bond_dim, compact)
            mut_seq = list(wt_span)
            mut_seq[pos_local-1] = mut_aa
            dlp_mut = log_prob_sequence(params, processor, mut_seq,
                                        max_len, bond_dim, compact)
            dlogp = -(dlp_mut - dlp_wt)

            # -site log-odds with this specialist
            slo = site_log_odds(params, processor,
                                wt_span, pos_local, mut_aa,
                                max_len, bond_dim, compact)
            if slo is not None and np.isfinite(slo):
                slo_val = -slo
            else:
                slo_val = np.nan

            dlogp_list.append(dlogp)
            slo_list.append(slo_val)

        # simple average ensemble across specialists
        neg_delta_logp_ens = float(np.nanmean(dlogp_list))
        neg_site_lo_ens    = float(np.nanmean(slo_list))

        out = {
            "mutant": m,
            "pos": pos_global,
            "wt_aa": m[0],
            "mut_aa": mut_aa,
            "pred_neg_delta_logp_ens": neg_delta_logp_ens,
            "pred_neg_site_log_odds_ens": neg_site_lo_ens,
        }
        if "DMS_score" in df.columns:
            out["DMS_score"] = float(row["DMS_score"])
        rows.append(out)

    pred_df = pd.DataFrame(rows)
    print(f"[{now()}] ✅ Ensemble predictions computed: {len(pred_df)} rows")

    # ensemble z-mean
    pred_df["pred_combo_zmean_ens"] = (
        0.5 * zscore(pred_df["pred_neg_delta_logp_ens"].to_numpy(np.float64))
        + 0.5 * zscore(pred_df["pred_neg_site_log_odds_ens"].to_numpy(np.float64))
    )

    # ---- metrics ----
    metrics = {
        "timestamp": now(),
        "n_total": int(len(df)),
        "n_scored": int(len(pred_df)),
        "span_min_pos": int(min_pos),
        "span_max_pos": int(max_pos),
        "span_len": int(L),
        "max_supported_len": int(MAX_SUPPORTED_LEN),
        "fitness_higher_is_better": bool(FITNESS_HIGHER_IS_BETTER),
        "specialist_paths": [s["path"] for s in specialists],
    }

    for col, tag in [
        ("pred_neg_delta_logp_ens", "dlogp_ens"),
        ("pred_neg_site_log_odds_ens", "sitelogodds_ens"),
        ("pred_combo_zmean_ens", "combo_zmean_ens"),
    ]:
        metrics.update(compute_block_metrics(pred_df, col, tag, args.out_dir))

    # ---- plots ----
    ann_d = None
    if "dlogp_ens_spearman" in metrics:
        ann_d = f"Spearman ρ={metrics['dlogp_ens_spearman']:.3f}\nPearson r={metrics['dlogp_ens_pearson']:.3f}"
    scatter_plot(pred_df, "pred_neg_delta_logp_ens",
                 "YAP1 WW — DMS vs −ΔlogP (specialist ensemble)",
                 "scatter_dms_vs_neg_dlogp_ens",
                 args.out_dir, ann_d)
    hist_plot(pred_df["pred_neg_delta_logp_ens"],
              "Distribution of predictions (−ΔlogP, ensemble)",
              "hist_neg_dlogp_ens",
              args.out_dir, "−Δ logP (ensemble)")
    zscatter_plot(pred_df, "pred_neg_delta_logp_ens",
                  "YAP1 WW — z-scored for −ΔlogP (ensemble)",
                  "scatter_zscore_neg_dlogp_ens",
                  args.out_dir)

    ann_s = None
    if "sitelogodds_ens_spearman" in metrics:
        ann_s = f"Spearman ρ={metrics['sitelogodds_ens_spearman']:.3f}\nPearson r={metrics['sitelogodds_ens_pearson']:.3f}"
    scatter_plot(pred_df, "pred_neg_site_log_odds_ens",
                 "YAP1 WW — DMS vs −site log-odds (specialist ensemble)",
                 "scatter_dms_vs_neg_slo_ens",
                 args.out_dir, ann_s)
    hist_plot(pred_df["pred_neg_site_log_odds_ens"],
              "Distribution of predictions (−site log-odds, ensemble)",
              "hist_neg_slo_ens",
              args.out_dir, "−site log-odds (ensemble)")
    zscatter_plot(pred_df, "pred_neg_site_log_odds_ens",
                  "YAP1 WW — z-scored for −site log-odds (ensemble)",
                  "scatter_zscore_neg_slo_ens",
                  args.out_dir)

    ann_c = None
    if "combo_zmean_ens_spearman" in metrics:
        ann_c = f"Spearman ρ={metrics['combo_zmean_ens_spearman']:.3f}\nPearson r={metrics['combo_zmean_ens_pearson']:.3f}"
    scatter_plot(pred_df, "pred_combo_zmean_ens",
                 "YAP1 WW — DMS vs ensemble (z-mean, specialists)",
                 "scatter_dms_vs_combo_zmean_ens",
                 args.out_dir, ann_c)
    hist_plot(pred_df["pred_combo_zmean_ens"],
              "Distribution (ensemble z-mean, specialists)",
              "hist_combo_zmean_ens",
              args.out_dir, "z-mean score (ensemble)")
    zscatter_plot(pred_df, "pred_combo_zmean_ens",
                  "YAP1 WW — z-scored for ensemble (specialists)",
                  "scatter_zscore_combo_zmean_ens",
                  args.out_dir)

    # ---- save outputs ----
    pred_csv = os.path.join(args.out_dir, "predictions_yap1_specialists.csv")
    pred_df.to_csv(pred_csv, index=False)

    metrics_json = os.path.join(args.out_dir, "metrics_yap1_specialists.json")
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[{now()}] ✅ Done.")
    print(f" ↳ Predictions: {pred_csv}")
    print(f" ↳ Metrics:     {metrics_json}")
    print(f" ↳ Figures:     {args.out_dir}/*.png, *.svg")


if __name__ == "__main__":
    main()
