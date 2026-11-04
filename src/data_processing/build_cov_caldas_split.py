"""
Phase 2 (Cov-Caldas / Stage 1) — frozen patient-level train/val/test split.

Reads src/data_processing/cov_caldas_verified_table.csv (built by
verify_cov_caldas.py) and produces the same frozen-split guarantee used for
RICORD: every image from one patient lands entirely in one split, zero
patient overlap, confirmed by assertion before saving anything.

Run from src/data_processing/.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
IN_PATH = Path(__file__).parent / "cov_caldas_verified_table.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def frozen_patient_split(df, train_frac=0.7, val_frac=0.15, seed=42):
    # Stratify by class at the PATIENT level (not just shuffle-and-slice) —
    # with only ~22% positive patients, an unstratified split risks val/test
    # ending up with too few positives to evaluate reliably.
    patient_labels = df.groupby("patient_id")["stage1_label"].first()

    train_patients, val_patients, test_patients = set(), set(), set()
    for label in patient_labels.unique():
        group = patient_labels[patient_labels == label].index.to_series().sample(frac=1, random_state=seed)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_patients.update(group.iloc[:n_train])
        val_patients.update(group.iloc[n_train:n_train + n_val])
        test_patients.update(group.iloc[n_train + n_val:])

    def label_split(pid):
        if pid in train_patients:
            return "train"
        if pid in val_patients:
            return "val"
        return "test"

    df = df.copy()
    df["split"] = df["patient_id"].apply(label_split)

    assert not (train_patients & val_patients), "LEAKAGE: train/val overlap!"
    assert not (train_patients & test_patients), "LEAKAGE: train/test overlap!"
    assert not (val_patients & test_patients), "LEAKAGE: val/test overlap!"
    print("Patient-level split confirmed: zero overlap between train/val/test.")
    print("Split was stratified by class — each split should preserve the ~22:78 ratio.")

    return df


if __name__ == "__main__":
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} images, {df['patient_id'].nunique()} patients")

    df = frozen_patient_split(df, seed=42)
    df.to_csv(OUT / "cov_caldas_master_table.csv", index=False)

    for split_name in ["train", "val", "test"]:
        subset = df[df["split"] == split_name]
        subset.to_csv(OUT / f"cov_caldas_split_{split_name}.csv", index=False)
        pos = (subset["stage1_label"] == "positive").sum()
        neg = (subset["stage1_label"] == "negative").sum()
        print(f"{split_name}: {len(subset)} images ({pos} positive, {neg} negative), "
              f"{subset['patient_id'].nunique()} unique patients")

    print(f"\nSaved cov_caldas_master_table.csv + split files to {OUT}")
    print("This split is now FROZEN for Cov-Caldas Stage 1. Do not regenerate without team review.")