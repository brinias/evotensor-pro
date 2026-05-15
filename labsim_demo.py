#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labsim_demo.py — Lightweight PK/PD "disease plugin" simulator for peptides.

Inputs:
  - peptides CSV (at least: sequence; ideally potency/AMP prob/toxicity proxies)
  - builtin plugin key (mrsa_demo, ecoli_uti_demo, cancer_invitro_demo)
    or a JSON path with params (schema documented below)
  - dosing schedule (dose_uM / interval_h / n_doses / duration_h)

Outputs:
  - results CSV sorted by efficacy (with simple safety penalty)
  - optional time-series CSV (per peptide concentration & effect state over time)

Plugin JSON (schema v2; all keys optional except 'type'):
{
  "name": "...",
  "type": "bacteria" | "cancer",
  "readout": "CFU_over_time" | "Viability_vs_time",
  "growth_model": "logistic",
  "params": { "r_per_h": 0.1, "K": 1e9, "N0": 1e6, "Viab0": 1.0 },
  "pk": { "model": "one_compartment", "kel_per_h": 0.05 },
  "pd": {
    "ec50": {
      "from": ["MIC_pred_uM","MIC_*","my_activity_prob"],
      "min_uM": 0.25, "max_uM": 256.0, "mic_divisor": 4.0
    },
    "kmax": {
      "base": 0.025, "scale_prob": 0.045,
      "from": ["AMP_activity_prob","amp_head_prob","prob","my_prob"]
    },
    "hill": { "base": 1.3, "from": ["my_hill"] },
    "emax": { "base": 0.9, "from": ["my_emax"] }   # used for cancer
  },
  "safety": { "tox_from": ["cpp_binary_is_toxic","Aggregation_*","gravy_heur"], "alpha": 0.2, "beta": 1.0 },
  "endpoint_hours": 24
}

This is intentionally simple (proxy-based) to support demos & what-if.
"""

import argparse, json, math, sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ----------------------- Built-in disease plugins ----------------------------

def builtin_plugin(key: str) -> Dict[str, Any]:
    key = (key or "").lower().strip()
    if key == "mrsa_demo":
        return {
            "name": "MRSA (demo)",
            "type": "bacteria",
            "readout": "CFU_over_time",
            "growth_model": "logistic",
            "params": {"r_per_h": 0.08, "K": 1e9, "N0": 1e6},
            "pk": {"model": "one_compartment", "kel_per_h": 0.04},
            "pd": {
                "ec50": {"from": ["MIC_ultimate_pred","MIC_static_pred","MIC_pred_uM","AMP_proxy","AMP_activity_prob"]},
                "kmax": {"base": 0.03, "scale_prob": 0.04},   # per hour
                "hill":  {"base": 1.3}
            },
            "safety": {"tox_from": ["Aggregation_*","gravy_heur"], "alpha": 0.2, "beta": 1.0},
            "endpoint_hours": 24
        }
    if key == "cancer_invitro_demo":
        return {
            "name": "Cancer in vitro (demo)",
            "type": "cancer",
            "readout": "Viability_vs_time",
            "params": {"Viab0": 1.0},
            "pk": {"model": "one_compartment", "kel_per_h": 0.03},
            "pd": {
                "ec50": {"from": ["IC50_uM","MIC_ultimate_pred","AMP_proxy","AMP_activity_prob"]},
                "emax": {"base": 0.9},   # max fractional effect
                "hill": {"base": 1.4}
            },
            "safety": {"tox_from": ["Aggregation_*","gravy_heur"], "alpha": 0.25, "beta": 1.0},
            "endpoint_hours": 72
        }
    if key == "ecoli_uti_demo":
        return {
            "name": "E.coli UTI (demo)",
            "type": "bacteria",
            "readout": "CFU_over_time",
            "growth_model": "logistic",
            "params": {"r_per_h": 0.1, "K": 1e9, "N0": 1e6},
            "pk": {"model": "one_compartment", "kel_per_h": 0.05},
            "pd": {
                "ec50": {"from": ["MIC_pred_uM","AMP_proxy","AMP_activity_prob"]},
                "kmax": {"base": 0.025, "scale_prob": 0.045},
                "hill": {"base": 1.3}
            },
            "safety": {"tox_from": ["Aggregation_*","gravy_heur"], "alpha": 0.2, "beta": 1.0},
            "endpoint_hours": 24
        }
    raise ValueError(f"Unknown builtin plugin: {key}")

# ----------------------- Utilities ----------------------------

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c.endswith("*"):
            prefix = c[:-1]
            hits = [col for col in df.columns if col.startswith(prefix)]
            if hits:
                return hits[0]
        else:
            if c in df.columns:
                return c
    return None

def sigmoid(x: float) -> float:
    return 1.0/(1.0 + math.exp(-x))

def soft_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

# ----------------------- PK/PD Mappings (proxy) ----------------------------

def map_ec50_uM(row: pd.Series, pd_cfg: Dict[str, Any]) -> float:
    """
    EC50 mapping from CSV -> μM (lower is better).
    Priority:
      - If a column from pd.ec50.from looks like MIC/IC50 in μM: EC50 ≈ MIC / mic_divisor.
      - If it's a probability 0..1: map prob -> μM with [min_uM, max_uM].
      - Else fall back to common column names.
    Settings in pd.ec50 (optional):
      - from: [col, "prefix_*", ...]
      - min_uM (default 0.25), max_uM (default 256.0)
      - mic_divisor (default 4.0)
    """
    ec = pd_cfg.get("ec50", {}) or {}
    min_uM = float(ec.get("min_uM", 0.25))
    max_uM = float(ec.get("max_uM", 256.0))
    mic_div = float(ec.get("mic_divisor", 4.0))

    sources = ec.get("from", []) or []
    col = first_existing(row.to_frame().T, sources) if sources else None

    val = None
    if col and col in row and pd.notna(row[col]):
        try:
            val = float(row[col])
        except Exception:
            val = None

    if val is None:
        for k in ["MIC_ultimate_pred","MIC_static_pred","MIC_pred_uM","IC50_uM",
                  "AMP_proxy","AMP_activity_prob","prob"]:
            if k in row and pd.notna(row[k]):
                try:
                    val = float(row[k]); break
                except Exception:
                    continue

    if val is None:
        p = 0.5
        ec50 = min_uM * (max_uM / min_uM) ** (1.0 - p)
    else:
        if 0.0 <= val <= 1.0:
            p = float(val)
            ec50 = min_uM * (max_uM / min_uM) ** (1.0 - p)
        else:
            ec50 = max(min_uM, float(val) / mic_div)

    return float(np.clip(ec50, min_uM, max_uM))

def map_kmax_per_h(row: pd.Series, pd_cfg: Dict[str, Any]) -> float:
    """
    kmax = base + scale * p, where p comes from:
      - pd.kmax.from: [col, "prefix_*", ...] (0..1)
      - else fallback to AMP_activity_prob/AMP_proxy/prob
    """
    kc = pd_cfg.get("kmax", {}) or {}
    base = float(kc.get("base", 0.03))
    scale = float(kc.get("scale_prob", 0.04))

    p = None
    sources = kc.get("from", []) or []
    if sources:
        col = first_existing(row.to_frame().T, sources)
        if col and col in row and pd.notna(row[col]):
            try:
                p = float(row[col])
            except Exception:
                p = None

    if p is None:
        for k in ["AMP_activity_prob","AMP_activity_ultimate_prob","AMP_proxy","prob"]:
            if k in row and pd.notna(row[k]):
                try:
                    p = float(row[k]); break
                except Exception:
                    continue

    if p is None:
        p = 0.5
    return base + scale * soft_clip(p, 0.0, 1.0)

def map_hill(row: pd.Series, pd_cfg: Dict[str, Any]) -> float:
    """
    Hill slope from pd.hill.base or pd.hill.from.
    Clamped to [0.5, 5.0] to avoid extreme shapes.
    """
    hc = pd_cfg.get("hill", {}) or {}
    if "from" in hc:
        col = first_existing(row.to_frame().T, hc.get("from", []))
        if col and col in row and pd.notna(row[col]):
            try:
                return float(np.clip(float(row[col]), 0.5, 5.0))
            except Exception:
                pass
    return float(hc.get("base", 1.3))

def map_emax(row: pd.Series, pd_cfg: Dict[str, Any]) -> float:
    """
    Emax from pd.emax.base or pd.emax.from (cancer model).
    Clamped to [0, 1].
    """
    ec = pd_cfg.get("emax", {}) or {}
    if "from" in ec:
        col = first_existing(row.to_frame().T, ec.get("from", []))
        if col and col in row and pd.notna(row[col]):
            try:
                return float(np.clip(float(row[col]), 0.0, 1.0))
            except Exception:
                pass
    return float(ec.get("base", 0.9))

def map_kel_per_h(row: pd.Series, plugin: Dict[str, Any]) -> float:
    kel = float(plugin.get("pk", {}).get("kel_per_h", 0.04))
    # small tweak: better protease stability ⇒ slower elimination (±15%)
    for k in ["ProteaseStability_score","ProteaseStability_proxy"]:
        if k in row and pd.notna(row[k]):
            try:
                s = float(row[k])
                kel *= (1.0 - 0.15*np.tanh(s))
            except Exception:
                pass
            break
    return max(0.002, min(1.0, kel))

def tox_penalty(row: pd.Series, safety_cfg: Dict[str, Any], C: float) -> float:
    """
    A small penalty [0..0.5] increasing with aggregation/hydrophobicity/tox signals and dose.
    If value is 0/1, sigmoid(β*v) handles it; if continuous, also fine.
    """
    alpha = float(safety_cfg.get("alpha", 0.2))
    beta  = float(safety_cfg.get("beta", 1.0))
    sig = 0.0
    for key in safety_cfg.get("tox_from", []):
        if key.endswith("*"):
            prefix = key[:-1]
            cand = [c for c in row.index if c.startswith(prefix)]
            if cand:
                try:
                    v = float(row[cand[0]])
                    sig += sigmoid(beta * v)
                except Exception:
                    pass
        else:
            if key in row and pd.notna(row[key]):
                try:
                    v = float(row[key])
                    sig += sigmoid(beta * v)
                except Exception:
                    pass
    # slightly scale with concentration (more exposure → more risk)
    return float(np.clip(alpha * sig * (1.0 + 0.01*C), 0.0, 0.5))

# ----------------------- Schedules & Simulation ----------------------------

@dataclass
class Dosing:
    dose_uM: float
    interval_h: float
    n_doses: int

def schedule_times(d: Dosing) -> List[float]:
    return [i*d.interval_h for i in range(d.n_doses)]

def simulate_bacteria(plugin: Dict[str, Any], row: pd.Series, dosing: Dosing, duration_h: float, dt_h: float=0.25, collect_ts: bool=False):
    params = plugin["params"]
    r = float(params["r_per_h"]); K = float(params["K"]); N = float(params["N0"])
    pd_cfg = plugin["pd"]
    EC50 = map_ec50_uM(row, pd_cfg)
    kmax = map_kmax_per_h(row, pd_cfg)
    h    = map_hill(row, pd_cfg)
    kel  = map_kel_per_h(row, plugin)

    tgrid = np.arange(0.0, duration_h+1e-9, dt_h)
    C = 0.0
    times = set(schedule_times(dosing))
    ts_rows = []
    for t in tgrid:
        if any(abs(t - ti) < 1e-12 for ti in times):
            C += dosing.dose_uM  # bolus
        # kill
        kk = kmax * (C**h)/(EC50**h + C**h)
        dN = r*N*(1.0 - N/K)*dt_h - kk*N*dt_h
        N = max(1.0, N + dN)
        # PK decay
        C = max(0.0, C * math.exp(-kel*dt_h))
        if collect_ts:
            ts_rows.append((t, C, N))

    N0 = float(params["N0"]); kill_log10 = math.log10(N0) - math.log10(N)
    peakC = dosing.dose_uM  # approx peak after first bolus
    penalty = tox_penalty(row, plugin.get("safety",{}), peakC)
    score = kill_log10 - penalty
    summary = {
        "EC50_uM": EC50, "kmax_per_h": kmax, "hill": h, "kel_per_h": kel,
        "CFU_start": N0, "CFU_end": N, "kill_log10": kill_log10,
        "safety_penalty": penalty, "efficacy_score": score
    }
    return summary, ts_rows

def simulate_cancer(plugin: Dict[str, Any], row: pd.Series, dosing: Dosing, duration_h: float, dt_h: float=0.25, collect_ts: bool=False):
    params = plugin["params"]
    Viab = float(params.get("Viab0", 1.0))
    pd_cfg = plugin["pd"]
    EC50 = map_ec50_uM(row, pd_cfg)
    Emax = map_emax(row, pd_cfg)
    h    = map_hill(row, pd_cfg)
    kel  = map_kel_per_h(row, plugin)

    tgrid = np.arange(0.0, duration_h+1e-9, dt_h)
    C = 0.0
    times = set(schedule_times(dosing))
    ts_rows = []
    for t in tgrid:
        if any(abs(t - ti) < 1e-12 for ti in times):
            C += dosing.dose_uM
        effect = Emax * (C**h)/(EC50**h + C**h)
        target = 1.0 - effect
        Viab += (target - Viab) * 0.25  # relax toward target
        Viab = float(np.clip(Viab, 0.0, 1.5))
        C = max(0.0, C * math.exp(-kel*dt_h))
        if collect_ts:
            ts_rows.append((t, C, Viab))

    peakC = dosing.dose_uM
    penalty = tox_penalty(row, plugin.get("safety",{}), peakC)
    score = (1.0 - Viab) - penalty
    summary = {
        "EC50_uM": EC50, "Emax": Emax, "hill": h, "kel_per_h": kel,
        "Viability_end": Viab, "kill_fraction": 1.0 - Viab,
        "safety_penalty": penalty, "efficacy_score": score
    }
    return summary, ts_rows

# ----------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Peptide PK/PD demo simulator (disease plugins).")
    ap.add_argument("--peptides_csv", required=True, help="CSV with at least 'sequence'. Include MIC/IC50/prob/AMP/tox columns if available.")
    ap.add_argument("--plugin", required=False, default="mrsa_demo", help="Builtin key (mrsa_demo,ecoli_uti_demo,cancer_invitro_demo) or JSON path.")
    ap.add_argument("--dose_uM", type=float, default=32.0)
    ap.add_argument("--interval_h", type=float, default=12.0)
    ap.add_argument("--n_doses", type=int, default=2)
    ap.add_argument("--duration_h", type=float, default=24.0)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_timeseries_csv", required=False, default="", help="Optional path for per-peptide time series output.")
    args = ap.parse_args()

    # Load peptides
    df = pd.read_csv(args.peptides_csv)
    if "sequence" not in df.columns:
        raise ValueError("peptides_csv must contain a 'sequence' column.")
    df = df.copy()

    # Load plugin
    if os.path.isfile(args.plugin):
        with open(args.plugin, "r") as f:
            plugin = json.load(f)
    else:
        plugin = builtin_plugin(args.plugin)

    # auto endpoint duration if missing/non-positive
    if "endpoint_hours" in plugin and (args.duration_h is None or args.duration_h <= 0):
        args.duration_h = float(plugin["endpoint_hours"])

    dosing = Dosing(dose_uM=args.dose_uM, interval_h=args.interval_h, n_doses=args.n_doses)

    rows = []
    ts_all = []
    collect_ts = bool(args.out_timeseries_csv)
    for i, row in df.iterrows():
        base = {"sequence": row.get("sequence", f"seq_{i}")}
        try:
            if plugin["type"] == "bacteria":
                res, ts_rows = simulate_bacteria(plugin, row, dosing, duration_h=args.duration_h, collect_ts=collect_ts)
                base.update({
                    "Endpoint_h": plugin.get("endpoint_hours", args.duration_h),
                    "kill_log10": res["kill_log10"],
                    "CFU_end": res["CFU_end"],
                    "safety_penalty": res["safety_penalty"],
                    "efficacy_score": res["efficacy_score"],
                    "EC50_uM": res["EC50_uM"], "kmax_per_h": res["kmax_per_h"], "hill": res["hill"], "kel_per_h": res["kel_per_h"]
                })
                if collect_ts:
                    for (t, C, N) in ts_rows:
                        ts_all.append({"sequence": base["sequence"], "t_h": t, "C_uM": C, "CFU": N})
            elif plugin["type"] == "cancer":
                res, ts_rows = simulate_cancer(plugin, row, dosing, duration_h=args.duration_h, collect_ts=collect_ts)
                base.update({
                    "Endpoint_h": plugin.get("endpoint_hours", args.duration_h),
                    "Viability_end": res["Viability_end"],
                    "kill_fraction": res["kill_fraction"],
                    "safety_penalty": res["safety_penalty"],
                    "efficacy_score": res["efficacy_score"],
                    "EC50_uM": res["EC50_uM"], "Emax": res["Emax"], "hill": res["hill"], "kel_per_h": res["kel_per_h"]
                })
                if collect_ts:
                    for (t, C, Viab) in ts_rows:
                        ts_all.append({"sequence": base["sequence"], "t_h": t, "C_uM": C, "Viability": Viab})
            else:
                raise ValueError(f"Unsupported plugin type: {plugin['type']}")
        except Exception as e:
            base.update({"error": str(e)})
        rows.append(base)

    out = pd.DataFrame(rows)
    if "efficacy_score" in out.columns:
        out = out.sort_values("efficacy_score", ascending=False)

    out.to_csv(args.out_csv, index=False)
    print(f"Saved simulation results → {args.out_csv}")
    print(out.head(5).to_string(index=False))

    if collect_ts and ts_all:
        ts_df = pd.DataFrame(ts_all)
        ts_df.to_csv(args.out_timeseries_csv, index=False)
        print(f"Saved time series → {args.out_timeseries_csv}")

if __name__ == "__main__":
    main()
