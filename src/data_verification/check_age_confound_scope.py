"""
Two checks on the comma-joined-age issue, now that we know the root cause
(participants.tsv stores one row per participant, and that row's age field
can itself hold multiple values for multi-session participants — this is a
source-data property, not a bug in verify_cov_caldas.py's merge logic).

Check 1: does participants.tsv actually have exactly one row per
    'participant', confirming there's no duplicate-row / duplicate-image
    mechanism at play (which would be a different, worse bug)?

Check 2: are the 595 multi-value-age rows evenly split between
    positive/negative, or do they skew one way? If they skew, dropping or
    approximating them isn't neutral — it would itself bias the age-gap
    number you're trying to measure honestly.

Run from src/data_verification/ (same place as verify_cov_caldas.py).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data" / "raw" / "cov_caldas"
MASTER = ROOT / "data" / "processed" / "cov_caldas_master_table.csv"


def check_participants_tsv_duplicates():
    print("=== Check 1: duplicate participant rows in participants.tsv? ===")
    for cls in ("positive", "negative"):
        path = DATA / cls / "MIDS" / "participants.tsv"
        df = pd.read_csv(path, sep="\t")
        n_rows = len(df)
        n_unique = df["participant"].nunique()
        print(f"  {cls}: {n_rows} rows, {n_unique} unique participant IDs "
              f"({'OK — one row per participant' if n_rows == n_unique else '[FLAG] DUPLICATES FOUND'})")
        if n_rows != n_unique:
            dupes = df[df["participant"].duplicated(keep=False)].sort_values("participant")
            print(dupes)


def check_class_skew_in_bad_age():
    print("\n=== Check 2: do multi-value-age rows skew by class? ===")
    df = pd.read_csv(MASTER)
    df["age_str"] = df["age"].astype(str)
    is_multi = df["age_str"].str.contains(",", na=False)

    print(f"Multi-value-age rows: {is_multi.sum()} / {len(df)}")
    print("\nClass breakdown of multi-value-age rows:")
    print(df[is_multi]["stage1_label"].value_counts())
    print(f"\nAs a % of each class's total:")
    for label in df["stage1_label"].unique():
        cls_total = (df["stage1_label"] == label).sum()
        cls_bad = (is_multi & (df["stage1_label"] == label)).sum()
        print(f"  {label}: {cls_bad}/{cls_total} = {100*cls_bad/cls_total:.1f}% affected")


if __name__ == "__main__":
    check_participants_tsv_duplicates()
    check_class_skew_in_bad_age()