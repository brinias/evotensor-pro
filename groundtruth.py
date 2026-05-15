
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report
)

def main():
    ap = argparse.ArgumentParser(description="Evaluate AMP predictions vs ground truth")
    ap.add_argument("--pred_csv", required=True, help="CSV με predictions από pep_product.py")
    ap.add_argument("--gt_csv", required=True, help="CSV με ground truth (πρέπει να έχει sequence,label)")
    ap.add_argument("--prob_col", default="amp_head_prob", help="Column με probabilities")
    ap.add_argument("--label_col", default="amp_head_label", help="Column με predicted labels")
    ap.add_argument("--decision_col", default="amp_head_decision", help="Column με αποφάσεις (positive/negative/abstain)")
    ap.add_argument("--seq_col", default="sequence", help="Sequence column (σε gt & preds)")
    ap.add_argument("--gt_label_col", default="label", help="Ground truth label column στο gt file")
    args = ap.parse_args()

    # --- load ---
    preds = pd.read_csv(args.pred_csv)
    gt_raw = pd.read_csv(args.gt_csv)

    required_pred_cols = [args.seq_col, args.prob_col, args.label_col, args.decision_col]
    missing_pred = [c for c in required_pred_cols if c not in preds.columns]
    missing_gt = [c for c in [args.seq_col, args.gt_label_col] if c not in gt_raw.columns]
    if missing_pred:
        raise RuntimeError(f"Missing prediction columns: {missing_pred}")
    if missing_gt:
        raise RuntimeError(f"Missing ground-truth columns: {missing_gt}")

    gt = gt_raw[[args.seq_col, args.gt_label_col]]

    # --- normalize sequences ---
    for df in (preds, gt):
        df[args.seq_col] = df[args.seq_col].astype(str).str.strip().str.upper()

    # --- merge ---
    df = preds.merge(gt, on=args.seq_col, how="inner")
    if df.empty:
        raise RuntimeError("⚠️ Μετά το merge δεν βρέθηκαν σειρές. Σιγουρέψου ότι οι sequence ταιριάζουν ακριβώς.")

    # --- ground truth & predictions ---
    y   = df[args.gt_label_col].astype(int).to_numpy()
    p   = df[args.prob_col].astype(float).to_numpy()
    yh  = df[args.label_col].astype(int).to_numpy()
    dec = df[args.decision_col].astype(str).to_numpy()

    # --- coverage aware: μετράμε μόνο όσα δεν είναι abstain ---
    mask = dec != "abstain"
    print(f"\nCoverage: {mask.mean():.1%}  (kept {mask.sum()} / {len(df)})")

    if mask.any():
        if np.unique(y[mask]).size > 1:
            print("AUROC:", roc_auc_score(y[mask], p[mask]))
            print("AUPRC:", average_precision_score(y[mask], p[mask]))
        print("F1:", f1_score(y[mask], yh[mask], zero_division=0))
        print("Precision:", precision_score(y[mask], yh[mask], zero_division=0))
        print("Recall:", recall_score(y[mask], yh[mask], zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y[mask], yh[mask]))
    else:
        print("⚠️ Κανένα non-abstain prediction!")

    # --- strict: μετράμε abstain ως λάθος ---
    yh_strict = yh.copy()
    yh_strict[~mask] = 1 - y[~mask]

    print("\nStrict evaluation (abstain = λάθος):")
    print("F1:", f1_score(y, yh_strict, zero_division=0))
    print("Precision:", precision_score(y, yh_strict, zero_division=0))
    print("Recall:", recall_score(y, yh_strict, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y, yh_strict))
    print("\nClassification report:\n", classification_report(y, yh_strict, digits=3))


if __name__ == "__main__":
    main()
