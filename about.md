# Evotensor: A Compact Tensor-Network System for Peptide Function Modeling

## Abstract

Evotensor is a research prototype for peptide and short protein sequence
modeling. It combines compact tensor-network sequence encoders with lightweight
specialist heads for function prediction and downstream simulation. The system is
designed around a multiscale inductive bias: local residue interactions are
encoded with Matrix Product State style operations, while longer-range sequence
structure is summarized through hierarchical coarse-graining inspired by MERA
and renormalization methods.

The repository includes the local web application, backend API, specialist
model artifacts, inference scripts, evaluation utilities, demo datasets, and a
small disease-plugin simulation workflow. On the committed demo evaluation
assets, the AMP classifier reaches AUROC 0.9555 and F1 0.9122, the toxicity
classifier reaches AUROC 0.9528 and F1 0.8951, and the stability classifier
reaches AUROC 0.8333 and F1 0.7655. These results should be interpreted as
research and demo benchmarks, not as clinical validation.

## 1. Introduction

Modern protein language models often rely on transformer architectures trained
at very large scale. Evotensor explores a different design point: small,
structured sequence models that encode biological sequences through explicit
multiscale constraints. The goal is not to replace large protein language
models, but to test whether tensor-network representations can provide useful
peptide embeddings, interpretable scalar features, and efficient downstream
heads in a lightweight system.

The working hypothesis is that short biological sequences contain local motifs,
charge patterns, hydrophobic structure, and medium-range dependencies that can
be represented efficiently by low-rank sequential contractions and
coarse-grained global summaries. This hypothesis motivates the use of
tensor-network style representations.

## 2. Model Design

Evotensor contains three main layers of computation.

First, the sequence is processed through local residue-level features and
heuristics, including length, charge, hydrophobicity, aromatic content,
cysteine-related features, amphipathic indicators, and guardrail flags.

Second, specialist tensor-network models encode sequence windows at multiple
length scales. The included specialists cover short and medium peptide regimes:

| Specialist | Max Length | Intended Regime |
| --- | ---: | --- |
| `specialist_win24-36_len30.pkl` | 30 aa | short peptides and local motifs |
| `specialist_win30-48_len40.pkl` | 40 aa | medium peptides |
| `specialist_win44-60_len52.pkl` | 52 aa | longer short proteins and peptides |

Third, downstream heads combine tensor-network-derived features with scalar
sequence descriptors. The included heads target AMP activity, toxicity,
cell-penetrating peptide behavior, MIC-like regression, and stability.

## 3. Physics-Inspired Inductive Bias

The model uses concepts from tensor networks as computational inspiration:

- Matrix Product State style contractions model local dependencies along a
  one-dimensional amino-acid chain.
- MERA-like coarse-graining motivates a hierarchy where local information is
  compressed into larger-scale sequence summaries.
- The resulting embedding acts as a compact multiscale representation that can
  support downstream biological tasks.

These analogies are architectural motivations. They do not imply that peptide
biology is literally being solved as a quantum field theory. The practical claim
is narrower: tensor-network structure can be a useful inductive bias for compact
sequence modeling.

## 4. Data And Artifacts

The repository contains:

- specialist model artifacts in `specialists/`
- task heads in `heads/`
- train/test/demo CSVs in `datasets/`
- evaluation outputs in `Eval-results/`
- original training scripts in `training/`
- a reproducible AMP evaluation demo in `demo/evaluation/`
- an E.coli UTI disease-plugin simulation demo in `demo/ecoli_uti/`

The historical training pipeline used peptide/protein sequence resources
including UniProt and Pfam-A.seed derived data. Upstream dataset licenses should
be reviewed before redistribution or derivative use.

Known provenance is summarized in `DATA_PROVENANCE.md`. The base/master and
specialist training scripts reference Pfam-derived inputs, and project notes
identify UniProt-derived sequence resources as part of the historical curation.
The AMP head uses project-curated labeled peptide CSVs with a held-out test
split. Toxicity and CPP heads are synthetic/pseudo-labeled workflow demos.

## 5. Evaluation Protocol

Two types of metrics are included.

The first type is reproducible local held-out test-set evaluation. These metrics
are computed by `groundtruth.py` from committed prediction CSVs and committed
matching label files. In particular, `datasets/amp_test.csv` contains peptides
that were excluded from default AMP-head training. Evidence level differs across
heads: some heads, including toxicity and CPP, were trained with synthetic or
pseudo-labeled data and should be read as workflow demonstrations until
externally validated.

The second type is saved cross-validation metadata in `heads/*.summary.json` and
`Eval-results/*.json`. These files document training-time or notebook-time
evaluation runs and are useful for inspection, but the local CSV evaluation is
the easiest path for a new user to reproduce.

## 6. Reproducible Local Results

The following results were reproduced locally on May 14, 2026.

| Task | n | AUROC | AUPRC | F1 | Precision | Recall | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AMP activity held-out | 357 | 0.9555 | 0.9574 | 0.9122 | 0.9200 | 0.9045 | 0.913 |
| Toxicity synthetic/pseudo | 468 | 0.9528 | 0.9410 | 0.8951 | 0.8970 | 0.8932 | 0.895 |
| Stability | 601 | 0.8333 | 0.7834 | 0.7655 | 0.7556 | 0.7756 | 0.760 |
| CPP synthetic/pseudo demo | 19 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.000 |

The toxicity and CPP results are synthetic/pseudo-data checks. The CPP result is
also included as a sanity check only because the test set is very small.

## 7. Saved Cross-Validation Summaries

The included head summaries report:

| Head | n | Primary Metric | Secondary Metric |
| --- | ---: | --- | --- |
| `amp_head` | 1427 | AUROC 0.9437 | AUPRC 0.9450 |
| `is_toxin_head` synthetic/pseudo | 1868 | AUROC 0.9470 | AUPRC 0.9367 |
| `pseudo_stability_head` | 2401 | AUROC 0.8247 | AUPRC 0.7848 |
| `pseudo_cpp_head` | 75 | AUROC 0.9920 | AUPRC 0.9886 |
| `pseudo_mic_head` | 2504 | R2 0.7009 | MAE 32.2586 |

The YAP1 evaluation artifacts in `Eval-results/` include:

- zero-shot combination score Spearman rho 0.1918 with 95 percent CI
  0.0954 to 0.2907
- frozen embedding Ridge regression Spearman 0.4772, Pearson 0.5313, R2 0.2245
- YAP1 q33 classification AUROC 0.7300 and AUPRC 0.6096

These results indicate that the representation contains useful biological
signal beyond the directly supervised AMP/toxicity heads, but they remain
research benchmarks.

## 8. Application Layer

The application exposes the research workflow through both CLI scripts and a web
UI.

The backend provides project storage, CSV upload, prediction, training,
evaluation, head management, disease-plugin simulation, and patient/demo panels.
The local development configuration bypasses Firebase authentication with a
development token, while production mode can use Firebase credentials.

The frontend lets a user:

1. upload sequence CSVs
2. run prediction
3. inspect output columns
4. evaluate predictions against ground truth
5. train small project-level heads
6. run disease-plugin simulation

## 9. Limitations

Evotensor is not a medical, diagnostic, or therapeutic decision system. It is a
research software artifact. The included datasets and heads are suitable for
demonstration, reproducibility checks, and method development, but stronger
claims require independent validation, careful dataset provenance review,
calibration analysis, and domain-specific experimental confirmation.

Several heads are lightweight or pseudo-labeled. In particular, the toxicity and
CPP demo heads were trained with synthetic or pseudo-labeled data. Some demo
test sets are small. The AMP test set is held out from default-head training,
but broader independent external validation is still required before making
operational claims. The physics terminology should be understood as an
architectural analogy and not as a claim of physical mechanistic proof.

Upstream data sources have their own licenses and attribution requirements. The
Evotensor license governs this repository's code, models, heads, predictions,
and derivatives, but it does not relicense third-party data such as UniProt,
Pfam, or any other source used in future rebuilds.

## 10. License And Use Restrictions

Evotensor is released under a research-only, non-commercial license. Academic
research, non-commercial scientific work, personal experimentation, and
educational use are permitted. Commercial use is not permitted without prior
written permission from the author.

The restriction applies to the code, model artifacts, specialist heads, learned
parameters, demo files, evaluation files, generated predictions, and derivative
works. Prohibited commercial use includes paid products, hosted services, paid
APIs, consulting deliverables, proprietary commercial pipelines, commercial
biotech/pharma/diagnostics discovery workflows, and using the project to train,
validate, improve, benchmark, or monetize another commercial system.

## 11. Conclusion

Evotensor demonstrates that compact tensor-network-inspired representations can
support practical peptide inference, evaluation, and simulation workflows in a
small local application. The strongest clean held-out result in this repository
is the AMP evaluation, while toxicity and CPP remain synthetic/pseudo-data
workflow checks. Stability, MIC-like regression, and YAP1 evaluation artifacts
provide additional evidence that the representation captures useful biological
sequence structure. The system is best viewed as a compact, inspectable research
platform for peptide modeling experiments.
