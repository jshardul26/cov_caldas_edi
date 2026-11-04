# Cov-Caldas Stage 1 execution guide

This checkout is the later Cov-Caldas, same-source **binary detection** project. The RICORD severity task is Stage 2 and is out of scope; do not run the legacy RICORD/ChestX-ray14 utilities in `src/data_verification/verify_join.py` or the old multi-source helpers in `src/utils/data_utils.py` for the Cov-Caldas pipeline.

## Scope

This is the Cov-Caldas, same-source **Stage 1 binary detection** pipeline. It provides metadata Logistic Regression, Random Forest, optional XGBoost, metadata MLP, image-only ResNet-18, and multimodal ResNet-18 plus age/sex fusion. Stage 2 severity remains intentionally untouched. This code pass did not train models. The new PyTorch model and app paths need initialization/inference verification in an environment with the listed dependencies.

The repository archive does not include the raw Cov-Caldas images. The processed CSVs present in the archive may contain paths from the machine where they were generated. Rebuild them from your own extracted dataset before training.

## 1. Clone and environment setup (PowerShell)

```powershell
git clone https://github.com/jshardul26/cov_caldas_edi.git
Set-Location .\cov_caldas_edi
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`git clone` downloads the uploaded source. The next command enters the checkout. The next two create and activate an isolated Python environment. The final commands install the dependencies. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process RemoteSigned` in that terminal and activate again. Python 3.11+ is recommended for current PyTorch releases; use the official PyTorch selector if a CUDA-specific wheel is needed: [PyTorch install selector](https://pytorch.org/get-started/locally/).

Check the environment and CUDA visibility:

```powershell
python --version
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The first command prints the Python version; the second prints PyTorch and whether it can see an NVIDIA GPU. CPU is supported but CNN training will be slower. If CUDA reports false on an NVIDIA laptop, use the official selector above to install a Windows/Pip wheel for a supported CUDA build, then rerun the check. The first CNN/fusion run needs internet access to download pretrained ResNet-18 weights.

## 2. Place the dataset

Extract the Cov-Caldas release into this layout (the verification script expects `participants.tsv` plus `sub-*` folders under each class):

```text
data/raw/cov_caldas/positive/MIDS/participants.tsv
data/raw/cov_caldas/positive/MIDS/sub-.../...
data/raw/cov_caldas/negative/MIDS/participants.tsv
data/raw/cov_caldas/negative/MIDS/sub-.../...
```

Do not commit the dataset. The scripts use paths relative to the repository root; no path editing should be necessary when this layout is followed.

## 3. Recommended Stage 1 sequence

Run commands from the repository root, in order.

### Step 1 — verify image and metadata structure

```powershell
python .\src\data_verification\verify_cov_caldas.py
```

Checks class metadata, missing age/sex handling, patient-image linkage, and writes `src/data_processing/cov_caldas_verified_table.csv`. It prints observed counts and the next step is the patient split. Requires `pandas` and the extracted data.

### Step 2 — freeze patient-level train/validation/test split

```powershell
python .\src\data_processing\build_cov_caldas_split.py
```

Stratifies patients by class, checks patient isolation, and writes `data/processed/cov_caldas_master_table.csv` plus `cov_caldas_split_train.csv`, `cov_caldas_split_val.csv`, and `cov_caldas_split_test.csv`. Review the printed patient and image counts. Treat this split as frozen; do not rerun casually after modeling. Next: Step 3, frontal filtering.

### Step 3 — filter to frontal images

```powershell
python .\src\data_processing\filter_frontal_views.py
```

Keeps AP/PA images and preserves the frozen patient assignments. Writes `data/processed/cov_caldas_master_frontal.csv`. Review dropped-patient counts and remaining view proportions. This is the input to the available Stage 1 models. Next: Step 3a, Cov-Caldas EDA.

### Step 3a — run Cov-Caldas exploratory analysis

```powershell
python .\src\eda\eda_cov_caldas.py
```

Reports class/split counts, age/sex distributions, age-estimation flags, and patient image counts; writes summary tables under `reports/eda/`. Next: optional Step 3b or Step 4.

### Step 3b — inspect acquisition metadata (optional)

```powershell
python .\src\eda\analyze_dicom_metadata.py
python .\src\eda\acquisition_domain_overlap.py
python .\src\eda\metadata_confound_test.py
```

Requires per-image JSON sidecars. The first command writes `reports/eda/cov_caldas_dicom_metadata_analysis.csv`; the next commands inspect acquisition-domain overlap and metadata shortcut signals. Next: Step 4.

### Step 4 — create age-matched evaluation subsets

```powershell
python .\src\evaluation\build_age_matched_eval_set_FIXED.py
```

Selects closest age pairs within age bands in validation and test, and writes `data/processed/cov_caldas_val_age_matched_FIXED.csv` and `cov_caldas_test_age_matched_FIXED.csv`. Small matched cohorts are exploratory and should be reported with uncertainty. Next: Step 4a, AP/PA subsets.

### Step 4a — create view-restricted test subsets

```powershell
python .\src\data_processing\add_ap_pa_stratified_eval.py
```

Writes `data/processed/cov_caldas_master_frontal_AP_ONLY.csv` and `..._PA_ONLY.csv`. These are view-restricted subsets, not age- or class-matched cohorts. The CNN reports patient-level test metrics separately when both classes are present. Next: Step 5, metadata baselines.

### Optional source-metadata audit

```powershell
python .\src\data_verification\check_age_confound_scope.py
python .\src\eda\check_native_image_properties.py
python .\src\eda\check_uint16_range.py
```

The first checks participant-ID uniqueness and estimated-age skew; the next two report image dtype, range, and dimensions. They read the extracted data and processed tables and print or save diagnostics under `reports/eda/`.

### Step 5 — train metadata baselines

```powershell
python .\src\models\baselines\baseline_metadata.py
```

Trains available Logistic Regression, Random Forest, and XGBoost (when installed), evaluating train/validation/test at the patient level. Saves separate estimators, configs, metrics, and predictions under `models/metadata_baselines/`; combined metrics go to `reports/baselines/cov_caldas_metadata_baseline_results.csv`. Next: Step 6, image CNN.

### Step 6 — train the image-only CNN

```powershell
python .\src\models\cnn\train_cnn.py
```

Trains ResNet-18 on images only, selects the best checkpoint by validation AUC, reloads it, and evaluates patient-level full, age-matched, and (when prepared) AP-only/PA-only test subsets. Saves `models/checkpoints/cov_caldas_resnet18_best.pt`, config and training history JSON files, plus `reports/baselines/cov_caldas_cnn_results.csv`. Full training can take substantial time; this is the command the friend runs for the image-only model. Next: Steps 7–9 for explanations and remaining models.

### Step 7 — optional Grad-CAM visualization

```powershell
python .\src\models\cnn\gradcam_cnn.py
```

Loads the CNN checkpoint and creates a random-sample visualization at `reports/figures/gradcam_test_age_matched.png` (or `gradcam_test_full.png` if matched data is unavailable). Requires a completed CNN checkpoint. Next: Step 8 metadata MLP, then Step 9 fusion.

### Step 8 — train the metadata MLP

```powershell
python .\src\models\train_stage1_dl.py --model mlp
```

Uses one row per patient and age/sex features. Saves best checkpoint, config, training history, train/validation/test metrics, and prediction CSVs in `models/metadata_mlp/stage1/`. Next: Step 9 fusion.

### Step 9 — train multimodal fusion

```powershell
python .\src\models\train_stage1_dl.py --model fusion
```

Combines a ResNet-18 image branch with age/sex metadata and class-weighted loss. Saves best checkpoint, config, history, metrics, and predictions in `models/fusion/stage1/`. Next: Step 10 saved-prediction plots.

### Step 10 — plot saved test evaluation outputs

```powershell
python .\src\evaluation\plot_model_evaluation.py
```

Reads the saved patient-level test predictions and does not retrain models. Writes combined ROC, precision–recall, and calibration curves to `reports/figures/stage1_test_curves.png`, plus one confusion-matrix image per model in the same folder. Run after the desired models have completed. Next: Step 11, launch the app.

### Step 11 — run the research demo frontend

```powershell
streamlit run .\app\app.py
```

Opens the local Streamlit app. Choose a trained fusion, image-only, or metadata MLP checkpoint. Fusion takes an image and metadata and shows probability, Grad-CAM, and SHAP values; image-only shows Grad-CAM; MLP uses age/sex. A non-clinical disclaimer is displayed in the app.

## 4. Train one metadata model independently

Each command trains only its named estimator and writes a reloadable `model.joblib`, `config.json`, and `metrics.csv` under its own directory. Run after Steps 1–3.

```powershell
python .\src\models\baselines\baseline_metadata.py --model logistic
python .\src\models\baselines\baseline_metadata.py --model random_forest
python .\src\models\baselines\baseline_metadata.py --model xgboost
```

The XGBoost command requires `xgboost`; the all-model command skips XGBoost if that optional import is unavailable. Independent deep-learning commands are `python .\src\models\cnn\train_cnn.py`, `python .\src\models\train_stage1_dl.py --model mlp`, and `python .\src\models\train_stage1_dl.py --model fusion`.

## 5. Lightweight code checks

```powershell
python -m compileall -q .\src
python -m pytest -q .\tests
python -c "import numpy, pandas, PIL, sklearn, scipy, joblib; print('Core imports OK')"
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import streamlit, shap; print('Streamlit and SHAP imports OK')"
```

These checks perform synthetic patient-split, age-matching, metadata MLP/fusion forward, and checkpoint state round-trip checks; they never run a training loop or produce research results. The test suite needs the dependencies installed in the virtual environment.

## GitHub upload and artifact policy

Commit source, requirements, and project documentation. `.gitignore` excludes raw/processed data, trained checkpoints, baseline model files, generated result CSVs, and figures. Keep datasets and trained weights outside GitHub; use a separately approved storage location if collaborators need them. Git LFS is unnecessary for this source-only upload. The existing extracted directory is not a Git checkout, so the safest upload workflow is to clone the GitHub repository into a clean directory, copy the reviewed source/docs into it, inspect `git status`, then commit and push from that clone:

```powershell
git clone https://github.com/jshardul26/cov_caldas_edi.git .\cov_caldas_edi-upload
Set-Location .\cov_caldas_edi-upload
git status --short
```

Copy `.gitignore`, `requirements.txt`, Markdown documentation, `src/`, `app/`, and the paper scaffold from the reviewed directory into this clone, preserving paths. Set `$sourceRoot` to the reviewed extracted folder if it differs from this example:

```powershell
$sourceRoot = 'C:\Users\Dell\Downloads\cov_caldas_edi-main\cov_caldas_edi-main'
Copy-Item "$sourceRoot\src" . -Recurse -Force
Copy-Item "$sourceRoot\tests" . -Recurse -Force
Copy-Item "$sourceRoot\.gitignore", "$sourceRoot\requirements.txt" . -Force
Get-ChildItem $sourceRoot -Filter '*.md' | Copy-Item -Destination . -Force
Copy-Item "$sourceRoot\app" . -Recurse -Force
New-Item -ItemType Directory -Path .\reports -Force | Out-Null
Copy-Item "$sourceRoot\reports\paper" .\reports -Recurse -Force
```

Then:

```powershell
git status --short
git add .gitignore requirements.txt *.md src tests app reports/paper
git diff --cached --stat
git commit -m "Prepare Cov-Caldas Stage 1 pipeline"
git push origin main
```

Before committing, confirm `git status` does not list datasets, checkpoints, or generated outputs. After pushing, verify the GitHub file list and confirm the latest commit contains source/docs only.

## Troubleshooting

- **Missing package:** activate `.venv`, then run `python -m pip install -r requirements.txt`.
- **CUDA unavailable:** check `nvidia-smi`, then install the matching Windows/Pip/CUDA build using the official PyTorch selector. CPU remains available.
- **Dataset path missing:** confirm both `data/raw/cov_caldas/{positive,negative}/MIDS` directories exist and each has `participants.tsv` and subject folders.
- **Missing images in processed paths:** rebuild Steps 1–3 on the current computer; do not rely on archived processed CSVs with paths from another laptop.
- **Age-matching output missing:** run Step 4 after Step 3; the CNN looks for files ending `_FIXED.csv`.
- **GPU out of memory:** reduce `batch_size` in `src/models/cnn/train_cnn.py` (currently 32) to 16 or 8; this changes runtime configuration, not methodology.
- **Fusion out of memory:** pass `--batch-size 8` or `--batch-size 4` to `train_stage1_dl.py`; the default is 16.
- **Checkpoint missing:** finish CNN training before running Grad-CAM. The checkpoint is intentionally excluded from Git.
- **GitHub rejects large files:** remove generated data/checkpoints from the staging area; they are excluded by `.gitignore` and should not be uploaded.
