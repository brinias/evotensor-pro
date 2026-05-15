#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m py_compile backend/main.py pep_product.py groundtruth.py labsim_demo.py
"$PYTHON_BIN" -c 'import fastapi, pandas, numpy, sklearn, joblib, jax; print("python deps ok")'

"$PYTHON_BIN" groundtruth.py \
  --pred_csv demo/evaluation/amp_predictions.csv \
  --gt_csv demo/evaluation/amp_ground_truth.csv

"$PYTHON_BIN" labsim_demo.py \
  --peptides_csv demo/ecoli_uti/for_ecoli_lab_test.csv \
  --plugin demo/ecoli_uti/ecoli_disease_plugin.json \
  --dose_uM 32 \
  --interval_h 12 \
  --n_doses 2 \
  --duration_h 24 \
  --out_csv /tmp/evotensor_ecoli_sim.csv

"$PYTHON_BIN" pep_product.py predict \
  --input_csv demo/ecoli_uti/for_ecoli_lab_test.csv \
  --out_csv /tmp/evotensor_predict.csv \
  --heads_dir heads

cd "$ROOT_DIR/frontend/evotensor-pro-ui"
npm run build

echo "Smoke tests passed."
