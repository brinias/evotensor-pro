# Evaluation Demo

This folder contains a ready-to-run AMP evaluation pair:

- `amp_predictions.csv` - prediction output from `pep_product.py`
- `amp_ground_truth.csv` - matching `sequence,label` ground-truth file

Run from the repository root:

```bash
source .venv/bin/activate
python groundtruth.py \
  --pred_csv demo/evaluation/amp_predictions.csv \
  --gt_csv demo/evaluation/amp_ground_truth.csv
```

Expected headline metrics are approximately:

- AUROC: `0.955`
- AUPRC: `0.957`
- F1: `0.912`

In the local web app, create a project, upload both CSVs, open the `evaluate`
tab, select `amp_predictions.csv` as predictions and `amp_ground_truth.csv` as
ground truth, then run evaluation.

