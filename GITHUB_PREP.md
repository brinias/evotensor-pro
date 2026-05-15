# Evotensor GitHub Prep

## What is included

- `frontend/evotensor-pro-ui`: Vite React UI from the live server.
- `backend/main.py`: live FastAPI backend, made portable with environment variables.
- `pep_product.py`, `groundtruth.py`, `labsim_demo.py`: inference, evaluation, and simulation scripts.
- `heads/`: lightweight demo heads.
- `specialists/`: specialist model artifacts used by inference.
- `datasets/`: demo/train/test CSVs from the server.
- `demo/ecoli_uti/`: disease-plugin demo files recovered from the live server.
- `demo/evaluation/`: prediction and ground-truth CSVs for local evaluation.
- `training/`: original training pipeline scripts.
- `Eval-results/`: evaluation outputs.
- `DATA_PROVENANCE.md`: source/provenance notes for training and demo data.

## What is intentionally excluded

- `.env`
- `secrets/`
- `volumes/`
- `server-live/`
- `node_modules/`
- frontend build output

## License

This repository uses the Evotensor Research Non-Commercial License in `LICENSE`.
Research and education are allowed. Commercial use, paid services, hosted APIs,
commercial biotech/pharma/diagnostics workflows, and repackaging or monetizing
the code, model artifacts, heads, predictions, or derivatives require prior
written permission.

Data provenance is documented in `DATA_PROVENANCE.md`. The historical
base/specialist training used UniProt and Pfam-derived sequence resources;
toxicity and CPP demo heads are synthetic/pseudo-labeled; upstream dataset terms
still apply and are not replaced by this repository license.

## Local run

Backend:

```bash
brew install python@3.12 expat
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
bash scripts/run_backend.sh
```

Use Python 3.12 locally. The live server runs Python 3.11, and the pinned
scientific stack is not intended for Python 3.13.

On recent macOS/Homebrew installs, `DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib`
is required when creating the venv so Python's `pyexpat` module loads the
Homebrew `expat` library instead of the system one.

Frontend:

```bash
cd frontend/evotensor-pro-ui
npm install
cd ../..
bash scripts/run_frontend.sh
```

Open:

```text
http://127.0.0.1:5173
```

Default local mode uses:

```text
EVOTENSOR_AUTH_MODE=dev
VITE_AUTH_MODE=dev
```

That bypasses Firebase for local API testing. For production, set Firebase env values and provide a service account JSON via `FIREBASE_CRED`.

## GitHub note

The `specialists/` artifacts are about 39 MB total. They are below GitHub's hard per-file limit, but Git LFS is still recommended for model files:

```bash
git lfs install
git lfs track "*.pkl" "*.joblib"
git add .gitattributes
```
