# Data Provenance And Source Notes

This document records the known data provenance for the Evotensor repository.
It is intended to make the project safer to publish and easier to audit.

This is not legal advice. Before using the project beyond personal or academic
research, verify every upstream dataset license and keep record-level provenance
for any new data you add.

## Upstream Scientific Sequence Sources

The historical base model, master model, and specialist tensor-network training
pipeline used peptide/protein sequence resources from:

| Source | Used For | License / Source Note |
| --- | --- | --- |
| UniProt | natural protein/peptide sequence resources used in historical pretraining and curation | UniProt RDF metadata reports CC BY 4.0 for UniProt data. Official source: `https://sparql.uniprot.org/.well-known/void` |
| Pfam-A.seed / Pfam-A.full | family-alignment sequence resources used by range fine-tuning and specialist training scripts | InterPro/Pfam downloadable data is documented as CC0 1.0. Official source: `https://interpro-documentation.readthedocs.io/en/latest/license.html` |

Relevant local training scripts reference Pfam inputs directly:

- `training/train_with_symbolics.py`
- `training/range_finetune_master.py`
- `training/multi-specialist-fine-tuning.py`

The repository does not include the full raw UniProt or Pfam dumps. It includes
trained artifacts, compact heads, demo CSVs, and evaluation/demo outputs.

## Included Task Heads And Demo Datasets

| Artifact / Dataset | Provenance Status | Notes |
| --- | --- | --- |
| `heads/amp_head.*` | project-curated AMP labeled peptide data | `datasets/amp_test.csv` is held out from default AMP-head training. Keep attribution/source records for any future rebuild of this dataset. |
| `datasets/amp_train.csv`, `datasets/amp_test.csv` | project-curated AMP train/test CSVs | Included for reproducible research demo. The test split is excluded from default AMP-head training. |
| `heads/is_tox_head.*`, `datasets/tox_train.csv`, `datasets/tox_test.csv` | synthetic/pseudo-labeled demo data | Treat as workflow/demo assets, not independently validated real toxicity data. |
| `heads/pseudo_cpp_head.*`, `datasets/cpp_train.csv`, `datasets/cpp_test.csv` | synthetic/pseudo-labeled demo data | Treat as workflow/demo assets, not independently validated real CPP data. |
| `heads/pseudo_stability_head.*`, `datasets/stability_train.csv`, `datasets/stability_test.csv` | pseudo-labeled/demo data | Used for workflow demonstration and local checks. |
| `heads/pseudo_mic_head.*`, `datasets/mic_train.csv`, `datasets/mic_test.csv` | pseudo-labeled/demo regression data | Used for MIC-like regression workflow demonstration. |
| `demo/evaluation/*` | derived demo evaluation pair | Provides a reproducible AMP prediction-vs-ground-truth evaluation example. |
| `demo/ecoli_uti/*` | project demo simulation files | Disease-plugin simulation demo recovered from the live server; not a clinical dataset. |
| `Eval-results/*` | saved evaluation outputs | Includes historical YAP1 evaluation artifacts; retain original DMS source citation if regenerating those evaluations. |

## License Boundary

The Evotensor repository is released under the Evotensor Research
Non-Commercial License in `LICENSE`.

That license applies to this repository's code, model artifacts, specialist
heads, learned parameters, demo files, generated predictions, and derivatives.
It does not override or relicense upstream third-party datasets. If upstream data
has separate terms, those terms still apply.

## Practical Release Rules

- Keep this provenance file in the repository.
- Do not claim that synthetic/pseudo-labeled heads are real-world validated.
- Do not claim commercial rights to upstream data.
- If new real biological datasets are added, include their source, URL,
  retrieval date, license, and any required citation.
- If rebuilding AMP or other heads from public databases, keep a record-level
  source table outside the model artifacts.

