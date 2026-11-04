"""
Phase 1 verification script.

Run this AFTER downloading both datasets (see DOWNLOAD_STEPS.md) and BEFORE
any modeling starts. It checks:
  1. Every metadata row has a matching image file (and vice versa).
  2. Patient-level image counts, so the team knows how many images per
     patient exist BEFORE deciding the train/val/test split.
  3. Basic class balance and missingness — no fabricated stats, everything
     printed here comes straight from the files on disk.

Expected layout (your actual NBIA-generated nesting under images/ is fine — this
script searches recursively, so extra subfolders like
MIDRC-RICORD-1c-manifest-Jan-2021/midrc_ricord_1c/ don't need to be flattened):
  phase1/data/ricord/images/             <- DICOM files from NBIA Data Retriever (any nesting)
  phase1/data/ricord/annotations.json    <- RICORD severity/appearance labels
  phase1/data/ricord/clinical.xlsx       <- RICORD clinical data
  phase1/data/chestxray14/images/        <- PNGs
  phase1/data/chestxray14/metadata.csv   <- Data_Entry_2017_v2020.csv (or subset)

Note: NBIA Data Retriever also drops its own auto-generated metadata.csv
inside the ricord/images/ folder (file-manifest info, not clinical data) —
that one is safely ignored by this script; only ricord/clinical.xlsx matters.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent / "data"


def _find_dicom_files(img_dir):
    """NBIA downloads don't always add a .dcm extension, so try both:
    first the fast path (files literally named *.dcm), and if that finds
    nothing, fall back to attempting to read every file as DICOM."""
    dcm_named = list(img_dir.rglob("*.dcm"))
    if dcm_named:
        return dcm_named, "matched by .dcm extension"

    try:
        import pydicom
    except ImportError:
        return [], "pydicom not installed — cannot fall back to content-sniffing"

    sniffed = []
    all_files = [p for p in img_dir.rglob("*") if p.is_file()]
    for p in all_files:
        try:
            pydicom.dcmread(str(p), stop_before_pixels=True)
            sniffed.append(p)
        except Exception:
            continue
    return sniffed, f"no .dcm-named files found ({len(all_files)} files scanned); matched by content-sniffing instead"


def verify_ricord():
    print("\n=== RICORD-1c ===")
    img_dir = BASE / "ricord" / "images"
    clinical_path = BASE / "ricord" / "clinical.xlsx"

    if not clinical_path.exists():
        print(f"  [MISSING] {clinical_path} not found — download it first.")
        return None
    if not img_dir.exists():
        print(f"  [MISSING] {img_dir} not found — run NBIA Data Retriever first.")
        return None

    df = pd.read_excel(clinical_path)
    print(f"  Clinical rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Try to find the patient/study identifier columns by common names.
    id_candidates = [c for c in df.columns if "MRN" in c.upper() or "PATIENT" in c.upper()]
    study_candidates = [c for c in df.columns if "STUDY" in c.upper() and "UID" in c.upper()]
    print(f"  Candidate patient-ID columns found: {id_candidates or 'NONE — check column names manually'}")
    print(f"  Candidate Study UID columns found: {study_candidates or 'NONE — check column names manually'}")

    # Scan actual DICOM files on disk (handles missing .dcm extensions).
    dicom_files = []
    try:
        import pydicom
        dicom_files, method = _find_dicom_files(img_dir)
        print(f"  DICOM files found on disk: {len(dicom_files)} ({method})")
        if dicom_files:
            sample = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
            sample_patient_id = getattr(sample, "PatientID", None)
            sample_study_uid = getattr(sample, "StudyInstanceUID", None)
            print(f"  Sample DICOM PatientID tag: {sample_patient_id}")
            print(f"  Sample DICOM StudyInstanceUID tag: {sample_study_uid}")

            # ACTUAL value-level cross-check, not just column-name presence.
            if id_candidates and sample_patient_id is not None:
                mrn_values = set(df[id_candidates[0]].astype(str))
                if sample_patient_id in mrn_values:
                    print(f"  [OK] Sample DICOM PatientID matches a value in '{id_candidates[0]}' exactly.")
                else:
                    # Known transform confirmed on this dataset: DICOM PatientID
                    # is "MIDRC-RICORD-1C-<anon-mrn>"; the clinical file stores
                    # just the <anon-mrn> suffix. Try that first.
                    stripped = sample_patient_id.replace("MIDRC-RICORD-1C-", "")
                    if stripped in mrn_values:
                        print(f"  [OK, VIA TRANSFORM] DICOM PatientID matches '{id_candidates[0]}' after stripping the 'MIDRC-RICORD-1C-' prefix.")
                        print(f"      -> Use this transform when joining: mrn = dicom_patient_id.replace('MIDRC-RICORD-1C-', '')")
                    else:
                        partial_hits = [v for v in mrn_values if v in sample_patient_id or sample_patient_id in v]
                        if partial_hits:
                            print(f"  [PARTIAL MATCH] No exact/known-transform match, but found a related value: {partial_hits[:3]}")
                            print(f"      -> Different transform needed here. Check manually.")
                        else:
                            print(f"  [WARNING] Sample DICOM PatientID '{sample_patient_id}' has NO match, exact or partial, in '{id_candidates[0]}'. Investigate before Phase 2.")

            if study_candidates and sample_study_uid is not None:
                uid_values = set(df[study_candidates[0]].astype(str))
                if sample_study_uid in uid_values:
                    print(f"  [OK] Sample DICOM StudyInstanceUID matches a value in '{study_candidates[0]}' exactly.")
                else:
                    print(f"  [WARNING] Sample DICOM StudyInstanceUID has NO exact match in '{study_candidates[0]}'. Investigate before Phase 2.")
    except ImportError:
        print("  [SKIPPED] pydicom not installed — run: pip install pydicom --break-system-packages")

    if id_candidates:
        n_patients = df[id_candidates[0]].nunique()
        print(f"  Unique patients in clinical file: {n_patients} (dataset page reports 361)")
        print(f"  Avg images/patient (from clinical file): {len(df) / n_patients:.2f}")

    return df, dicom_files


def verify_ricord_annotations(clinical_df):
    print("\n=== RICORD-1c annotations.json (Stage 2 severity labels) ===")
    ann_path = BASE / "ricord" / "annotations.json"
    if not ann_path.exists():
        print(f"  [MISSING] {ann_path} not found.")
        return

    with open(ann_path) as f:
        ann = json.load(f)

    # MD.ai JSON exports vary in shape; report what's actually there rather
    # than assume a fixed schema.
    if isinstance(ann, dict) and "datasets" in ann and ann["datasets"]:
        dataset_obj = ann["datasets"][0]
        print(f"  Keys inside datasets[0]: {list(dataset_obj.keys())}")
    else:
        dataset_obj = {}
        print(f"  Top-level keys in JSON: {list(ann.keys()) if isinstance(ann, dict) else 'JSON is a list'}")

    studies = dataset_obj.get("studies", [])
    annotations = dataset_obj.get("annotations", [])
    print(f"  'studies' list length: {len(studies)} — this is a study index/lookup, NOT the severity labels.")
    print(f"  'annotations' list length: {len(annotations)} — this is very likely where the real labels live.")

    if annotations:
        sample_ann = annotations[0]
        print(f"  Sample annotation entry: {sample_ann}")

    # Decode labelGroups so labelId values become human-readable names.
    label_groups = ann.get("labelGroups", [])
    id_to_name = {}
    for group in label_groups:
        for label in group.get("labels", []):
            if "id" in label and "name" in label:
                id_to_name[label["id"]] = label["name"]
    print(f"  Decoded {len(id_to_name)} labelId -> name mappings from labelGroups.")
    if id_to_name:
        preview = dict(list(id_to_name.items())[:8])
        print(f"  Sample label names available: {preview}")

    if annotations and "labelId" in annotations[0]:
        decoded_sample = id_to_name.get(annotations[0]["labelId"], "UNKNOWN — id not found in labelGroups")
        print(f"  First annotation's labelId decodes to: {decoded_sample}")
        print("  -> If this looks like 'Typical Appearance', 'Mild', etc., you've found your Stage 2 severity/appearance labels.")
    else:
        print("  [ACTION NEEDED] Could not auto-locate a 'labelId' field in the annotations entries.")
        print("        Manually inspect the sample annotation entry printed above and tell me its exact")
        print("        keys — the label field may be named differently, and the script can be adjusted.")


def verify_chestxray14():
    print("\n=== ChestX-ray14 (No Finding subset) ===")
    img_dir = BASE / "chestxray14" / "images"
    meta_path = BASE / "chestxray14" / "metadata.csv"

    if not meta_path.exists():
        print(f"  [MISSING] {meta_path} not found — download it first.")
        return None
    if not img_dir.exists():
        print(f"  [MISSING] {img_dir} not found — download images first.")
        return None

    df = pd.read_csv(meta_path)
    print(f"  Metadata rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    id_col = "Patient ID" if "Patient ID" in df.columns else None
    img_col = "Image Index" if "Image Index" in df.columns else None

    if id_col is None or img_col is None:
        print("  [WARNING] Expected 'Patient ID' / 'Image Index' columns not found — check CSV headers.")
        return df

    n_patients = df[id_col].nunique()
    print(f"  Unique patients: {n_patients}")
    print(f"  Avg images/patient: {len(df) / n_patients:.2f}")

    # Check every metadata row has a matching image file.
    existing_files = {p.name for p in img_dir.glob("*")}
    missing = [f for f in df[img_col] if f not in existing_files]
    print(f"  Metadata rows with NO matching image file: {len(missing)}")
    if missing[:5]:
        print(f"    e.g. {missing[:5]}")

    # Check for orphan images (files with no metadata row).
    metadata_files = set(df[img_col])
    orphans = [f for f in existing_files if f not in metadata_files]
    print(f"  Image files with NO matching metadata row: {len(orphans)}")

    return df


def check_class_balance(ricord_n, cxr14_n):
    print("\n=== Stage 1 class balance (as currently sampled) ===")
    print(f"  Positive (RICORD-1c): {ricord_n if ricord_n is not None else 'unknown'}")
    print(f"  Negative (ChestX-ray14 sample): {cxr14_n if cxr14_n is not None else 'unknown'}")
    if ricord_n and cxr14_n:
        ratio = ricord_n / cxr14_n
        print(f"  Ratio pos:neg = {ratio:.2f} : 1")
        if not (0.5 <= ratio <= 2.0):
            print("  [FLAG] Class imbalance is more than 2:1 — consider resampling before training.")


if __name__ == "__main__":
    ricord_result = verify_ricord()
    if ricord_result is not None:
        ricord_df, ricord_dicoms = ricord_result
        verify_ricord_annotations(ricord_df)
    else:
        ricord_df = None

    cxr14_df = verify_chestxray14()

    ricord_n = len(ricord_df) if ricord_df is not None else None
    cxr14_n = len(cxr14_df) if cxr14_df is not None else None
    check_class_balance(ricord_n, cxr14_n)

    print("\nDone. Fix any [MISSING]/[WARNING]/[FLAG] items above before starting Phase 2.")