# Phase 1 code audit report

## Scope and repository map

The checkout is a source archive rather than a Git working tree. It contains the Cov-Caldas Stage 1 data pipeline, metadata baselines, image-only ResNet-18, metadata MLP, multimodal fusion, Streamlit demo, explainability helpers, and setup documentation. Raw images are absent from the archive; processed tables and generated files are present but may contain machine-specific paths.

The active data path is `data/raw/cov_caldas` → `src/data_verification/verify_cov_caldas.py` → `src/data_processing/build_cov_caldas_split.py` → `filter_frontal_views.py` → `src/evaluation/build_age_matched_eval_set_FIXED.py` → available models. The old RICORD/ChestX-ray14 verification and shared Stage 1/Stage 2 utilities are stale for the later Cov-Caldas pivot.

## Issues corrected

| File | Finding | Change |
|---|---|---|
| `src/evaluation/build_age_matched_eval_set_FIXED.py` | Pairwise distances were flattened before 2-D index unpacking; after correcting that, duplicate pair selection also needed prevention. | Retained the distance matrix, sorted flat pair indices, and greedily selected only unused patients. |
| `src/models/cnn/train_cnn.py` | The image loader required train-only normalization; Windows-spawned DataLoader workers would not inherit the parent process global; evaluation used obsolete age-match filenames. | Computes train-only stats, copies stats into each dataset for worker processes, stores them in the checkpoint, evaluates patient-level outputs, adds full metrics/predictions, and uses fixed age/view subsets. |
| `src/models/cnn/gradcam_cnn.py` | Grad-CAM loaded raw images unlike the corrected CNN preprocessing and looked for the obsolete matched-set filename. | Uses the shared normalized loader and the checkpoint's frozen statistics; prefers the fixed matched test table. Legacy checkpoints without stats now stop with a clear retraining message. |
| `src/models/baselines/baseline_metadata.py` | Metadata estimators/results were not saved as independently reloadable artifacts and had no single-model CLI. | Adds `--model` choices, saves each fitted estimator with config and per-model metrics, and retains combined metrics for `--model all`. |
| `src/models/train_stage1_dl.py` | Stage 1 metadata MLP and multimodal fusion trainers were absent. | Adds independent `--model mlp` and `--model fusion` training paths, validation checkpoint selection, final split metrics and predictions, config/history/checkpoints. |
| `src/evaluation/plot_model_evaluation.py` | Saved test predictions had no standalone plotting workflow. | Adds ROC, precision–recall, calibration, and confusion-matrix plots without retraining. |
| `src/evaluation/plot_model_evaluation.py` | Saved test predictions had no standalone plotting workflow. | Adds ROC, precision–recall, calibration, and confusion-matrix plots without retraining. |
| `app/app.py` | Streamlit research demo was absent. | Loads fusion, image-only, and metadata MLP checkpoints; adds probability output, Grad-CAM for image models, metadata SHAP for metadata branches, and a non-clinical disclaimer. |
| `src/eda/{analyze_dicom_metadata,acquisition_domain_overlap,check_native_image_properties,check_uint16_range,inspect_ge_negative_overlap,check_controlled_split_feasibility,metadata_confound_test}.py` | EDA helpers contained hardcoded paths from one Windows machine; some joined per-image metadata by row position and one-hot encoded using the full dataset before splitting. | Replaced paths with repository-relative paths, joined by `image_path`, and fit categorical encoding on training data only. |
| `requirements.txt` | Listed only a small subset of direct dependencies. | Added the libraries used by current active scripts, including Torch, torchvision, Pillow, scikit-learn, SciPy, matplotlib, joblib, and XGBoost. |
| `.gitignore` | Did not exclude processed tables and generated model/result files. | Excludes raw/processed data, checkpoints, saved baseline models, and generated reports/figures. |

## Stage completion and model status

- **Cov-Caldas Stage 1 verification, split, and frontal filtering:** scripts are present; execution against a fresh extracted dataset was not possible in this environment because raw images are absent.
- **Metadata baselines:** Logistic Regression and Random Forest are implemented; XGBoost is optional. Each now has a separate train command and artifact output.
- **Image-only baseline:** ResNet-18 training/evaluation/checkpoint code is present. Training was not run.
- **Stage 1 metadata MLP:** implemented in `src/models/train_stage1_dl.py --model mlp`.
- **Stage 1 multimodal fusion model:** implemented in `src/models/train_stage1_dl.py --model fusion` (ResNet-18 plus age/sex).
- **Stage 2 severity prediction:** intentionally untouched. Do not interpret this as complete; it remains excluded from this task.
- **Evaluation:** deep models report patient-level ROC-AUC, PR-AUC, accuracy, precision, recall/sensitivity, specificity, F1, Brier score, confusion counts, and patient predictions. CNN view-restricted and age-matched evaluations are supported. The app loads MLP/fusion/image CNN checkpoints and provides Grad-CAM where applicable plus SHAP for metadata branches.

The Stage 1 machinery and paper scaffold are implemented. A full readiness claim is still pending because the PyTorch model forward/save/load paths and frontend could not be executed in this environment and the raw dataset is absent. The paper scaffold contains result placeholders until the friend's experiments are run. Stage 2 is intentionally excluded.

## Verification run

- `python -m compileall -q src` — passed.
- Core imports (`numpy`, `pandas`, `PIL`, `sklearn`, `scipy`, `joblib`) — passed.
- Synthetic test of corrected age matching — passed after exposing and fixing the flattened/unhashable index issue; selected closest unique patients as expected.
- Python syntax compilation for source and app — passed.
- Synthetic metadata model fit, artifact save/load, and metric generation — passed.
- Synthetic image normalization and patient-level split leakage checks — passed.
- PyTorch/Torchvision import and model forward/checkpoint smoke tests in `tests/test_stage1_smoke.py` — authored but not run because PyTorch is not installed in the current environment. No model training was performed.
- Dataset verification, preprocessing, and end-to-end pipeline — not run: required raw Cov-Caldas dataset is not included in this checkout.
- Interactive Streamlit app test — not run because PyTorch/Streamlit dependencies are unavailable in this environment.

## Stage 2 freeze

The RICORD severity task is Stage 2 per the project tracker. No Stage 2 trainer, labels, architecture, or feature builder was implemented or changed. Legacy files `src/data_verification/verify_join.py` and `src/utils/data_utils.py` were left unchanged; the latter contains Stage 2 utilities and should not be used for this Cov-Caldas-only execution sequence.

## GitHub handoff

Use the source-only workflow and exclusions in `EXECUTION_GUIDE.md`. The supplied directory has no `.git` folder, so it cannot be committed directly until copied into a clone. Keep raw and processed data, checkpoints, model objects, and generated reports out of Git. The user performs the Git operations; no remote changes were made.
