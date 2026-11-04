"""
Build a FRONTAL-VIEW-ONLY version of the Cov-Caldas master table.

Why: the file names carry the X-ray view (vp-ap, vp-pa, vp-ll = lateral).
Counting them by class showed side views are almost exclusively negative:
  ll (left lateral): 699 negative vs 1 positive
A side-view X-ray looks completely different from a front view, so a CNN can
learn "side view -> negative" without any lung analysis. Filtering to the
frontal views (AP + PA) removes that shortcut.

What this does NOT change: patient-to-split assignment. Every patient keeps
the split they were frozen into; rows are only removed inside each split, so
zero patient overlap is preserved (asserted below). The original
cov_caldas_master_table.csv is left untouched. Patients whose only images
were side views drop out of the frontal table entirely - the counts are
printed so the team can review them (the frozen-split rule asks for review).

Remaining known confound: among frontal images, AP vs PA mix still differs
by class (see printed table) - report it as a limitation and consider
evaluating within view.

Run from src/data_processing/.
"""

from pathlib import Path

import pandas as pd


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("Could not locate project root (no data/processed folder found above this file).")


ROOT = find_project_root()
PROCESSED = ROOT / "data" / "processed"
FRONTAL = {"ap", "pa"}


def main():
    df = pd.read_csv(PROCESSED / "cov_caldas_master_table.csv")
    df["view"] = df["image_path"].str.extract(r"vp-([a-z]+)", expand=False)

    n_missing = df["view"].isna().sum()
    if n_missing:
        print(f"[FLAG] {n_missing} images have no vp-<view> token in the file name; "
              f"they are treated as NOT frontal and dropped. Inspect them.")

    print("=== Before: view by class ===")
    print(pd.crosstab(df["view"].fillna("unknown"), df["stage1_label"]).to_string())

    front = df[df["view"].isin(FRONTAL)].copy()

    lost = set(df["patient_id"]) - set(front["patient_id"])
    lost_by_class = df[df["patient_id"].isin(lost)].groupby("stage1_label")["patient_id"].nunique()
    print(f"\nPatients removed entirely (only had non-frontal images): {len(lost)}  "
          f"{lost_by_class.to_dict()}")

    assert front.groupby("patient_id")["split"].nunique().max() == 1, "patient in >1 split!"
    print("Confirmed: every kept patient is still in exactly one split.")

    print("\n=== After: frontal-only, images and patients per split/class ===")
    summary = front.groupby(["split", "stage1_label"]).agg(
        images=("image_path", "count"), patients=("patient_id", "nunique")).unstack(fill_value=0)
    print(summary.to_string())

    print("\n=== Remaining confound: AP vs PA share by class (frontal only) ===")
    ct = pd.crosstab(front["stage1_label"], front["view"], normalize="index").round(3)
    print(ct.to_string())

    out = PROCESSED / "cov_caldas_master_frontal.csv"
    front.to_csv(out, index=False)
    print(f"\nSaved {out}  ({len(front)} of {len(df)} images kept)")
    print("Original cov_caldas_master_table.csv is unchanged.")


if __name__ == "__main__":
    main()