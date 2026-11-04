# Cov-Caldas COVID-19 chest X-ray project

Research code for **Stage 1 only**: binary COVID-19 detection using the Cov-Caldas chest X-ray release. The active comparison includes metadata-only classical baselines, a metadata MLP, an image-only ResNet-18, and an image-plus-metadata fusion ResNet-18. RICORD severity prediction (Stage 2) is deliberately outside this checkout's execution scope.

## Start here

- [Complete PowerShell setup, preprocessing, independent training, evaluation, and app commands](EXECUTION_GUIDE.md)
- [Code audit and verification record](AUDIT_REPORT.md)
- [Stage 1 research paper scaffold](reports/paper/STAGE1_PAPER_DRAFT.md)

## Data and outputs

Place the positive and negative MIDS releases under `data/raw/cov_caldas/{positive,negative}/MIDS/`. Raw data, processed CSVs, model checkpoints, and generated metrics/figures are excluded from Git. Rebuild processed tables on each computer so image paths point to local files.

Model outputs are saved below `models/`; metrics and per-patient predictions are saved with each trained deep model. Do not commit datasets, trained weights, or generated results. The app is a research demonstration, not a clinical diagnostic tool.

## Current verification boundary

Syntax and lightweight synthetic checks are documented in `AUDIT_REPORT.md`. Full dataset preprocessing, PyTorch model initialization/save-load, frontend interaction, and training have not been run as part of code preparation. Do not interpret example commands as evidence of experimental performance.

## Dataset-free checks

Run `python -m compileall -q src app` and `python -m pytest -q tests` for syntax, patient-split, age-matching, model forward, and checkpoint state round-trip checks. These checks do not train models or generate experimental results.
