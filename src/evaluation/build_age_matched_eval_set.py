"""
Age-matched evaluation subsets (val and test), patient-level matching.

Within each 10-year age band, keep an EQUAL number of positive and negative
patients (random, seed=42), so age carries no information about the label.
All images of a kept patient are retained. Matched sets are small, so treat
results on them as directional and report counts.

Run from src/evaluation/.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("Could not locate project root (no data/processed folder found above this file).")


ROOT = find_project_root()
PROCESSED = ROOT / "data" / "processed"

BIN_EDGES = list(range(0, 121, 10))
SEED = 42


def match_split(df, split_name):
    patients = df.groupby("patient_id").agg(age=("age", "first"),
                                            label=("stage1_label", "first")).reset_index()
    patients["age_bin"] = pd.cut(patients["age"], bins=BIN_EDGES, right=False)

    rng = np.random.RandomState(SEED)
    keep_ids = []
    rows = []
    for age_bin, grp in patients.groupby("age_bin", observed=True):
        pos = grp[grp["label"] == "positive"]["patient_id"].values
        neg = grp[grp["label"] == "negative"]["patient_id"].values
        k = min(len(pos), len(neg))
        rows.append({"age_bin": str(age_bin), "pos_patients": len(pos),
                     "neg_patients": len(neg), "kept_per_class": k})
        if k == 0:
            continue
        keep_ids.extend(rng.choice(pos, k, replace=False))
        keep_ids.extend(rng.choice(neg, k, replace=False))

    print(f"\n=== {split_name}: per-age-band matching ===")
    print(pd.DataFrame(rows).to_string(index=False))

    matched = df[df["patient_id"].isin(keep_ids)].copy()
    if matched.empty:
        print(f"[FLAG] No matched patients in {split_name}.")
        return matched

    per_patient = matched.groupby("patient_id").agg(age=("age", "first"),
                                                    label=("stage1_label", "first"))
    n_pat = per_patient["label"].value_counts().to_dict()
    n_img = matched["stage1_label"].value_counts().to_dict()
    gap = abs(per_patient.groupby("label")["age"].mean().diff().dropna().iloc[0])
    print(f"{split_name} matched: patients {n_pat}, images {n_img}, "
          f"residual mean-age gap {gap:.1f} years")
    if min(n_pat.values()) < 20:
        print(f"[FLAG] Fewer than 20 patients in the smaller class of matched {split_name} - "
              f"AUC on this subset will be very noisy; report as directional, with counts.")
    return matched


def main():
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    val_matched = match_split(df[df["split"] == "val"], "val")
    test_matched = match_split(df[df["split"] == "test"], "test")

    val_matched.to_csv(PROCESSED / "cov_caldas_val_age_matched.csv", index=False)
    test_matched.to_csv(PROCESSED / "cov_caldas_test_age_matched.csv", index=False)
    print(f"\nSaved cov_caldas_val_age_matched.csv and cov_caldas_test_age_matched.csv to {PROCESSED}")


if __name__ == "__main__":
    main()
