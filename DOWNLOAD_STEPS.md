# Phase 1 — Download Steps

## Dataset 1: MIDRC-RICORD-1c
1. Go to https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/
2. Install the **NBIA Data Retriever** (link on that page) if not already installed.
3. Download the **Images manifest** (`.tcia` file) → open it in NBIA Data Retriever → it bulk-downloads all 998 DICOMs.
4. Download the **Annotations** (JSON+ZIP, 2.34 MB) — direct browser download.
5. Download the **Clinical data** (XLSX, 87.53 KB) — direct browser download.
6. Put all three under `phase1/data/ricord/` (`images/`, `annotations.json`, `clinical.xlsx`).

## Dataset 2: NIH ChestX-ray14 (No Finding subset)
1. Get it from the Kaggle mirror (`nih-chest-xrays/data`) or Hugging Face (`alkzar90/NIH-Chest-X-ray-dataset`) — either works, no registration needed.
2. Download `Data_Entry_2017_v2020.csv` + the image folders.
3. Filter rows where the label column is exactly `No Finding`.
4. Randomly sample ~1,000 of those (fixed seed — log it).
5. Put under `phase1/data/chestxray14/` (`images/`, `metadata.csv`).

## Then run
```bash
pip install pandas pydicom --break-system-packages
python phase1/verify_join.py
```
This checks that every metadata row has a matching image and flags orphans on either side, before anyone starts modeling.
