"""
Phase 3 (Cov-Caldas / Stage 1) — EDA, with emphasis on the same-source check.

This is the concrete test of whether the Cov-Caldas pivot actually solved the
shortcut-learning problem: since positive AND negative both come from the
SAME hospital, age/sex distributions between classes should look similar
(unlike the old RICORD+ChestX-ray14 pairing, where a 13-year age gap between
classes was the smoking gun for shortcut learning).

Run from src/eda/.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)


def inspect_raw(df):
    """Basic dataset hygiene — always look at this before trusting any
    downstream stat. Skipping this is exactly how the age-parsing bug
    went unnoticed: the old code counted failures but never showed one."""
    print("\n=== Basic inspection (raw, before any cleaning) ===")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumns and dtypes:\n{df.dtypes}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nMissing values per column:\n{df.isna().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    if "age" in df.columns:
        print(f"\nRaw 'age' dtype: {df['age'].dtype}")
        print(f"Raw 'age' unique sample (first 20 distinct values): {df['age'].unique()[:20]}")


def load():
    df = pd.read_csv(PROCESSED / "cov_caldas_master_table.csv")
    inspect_raw(df)

    # As of the verify_cov_caldas.py fix, age is resolved upstream (mean of
    # multi-session values where ambiguous) and always numeric, with
    # age_is_estimated flagging which rows are approximations. Coerce
    # defensively anyway rather than assume the upstream fix is perfect.
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    n_bad_age = df["age"].isna().sum()
    if n_bad_age:
        print(f"\n[FLAG] {n_bad_age} rows still have an unparseable age even after the "
              f"verify_cov_caldas.py fix — this means the fix didn't cover every case; "
              f"inspect these before trusting age stats.")
    if "age_is_estimated" in df.columns:
        n_est = df["age_is_estimated"].sum()
        print(f"\n{n_est} rows ({100*n_est/len(df):.1f}%) have an age estimated as the mean "
              f"of a multi-session value (age_is_estimated=True) — a documented source-data "
              f"ambiguity, not a modeling error. See same_source_sanity_check() for whether "
              f"this skews by class.")
    else:
        print("\n[FLAG] No 'age_is_estimated' column found — you're likely running this against "
              "a master table built BEFORE the verify_cov_caldas.py fix. Rerun "
              "verify_cov_caldas.py then build_cov_caldas_split.py to regenerate it.")
    print(f"\nLoaded {len(df)} rows from cov_caldas_master_table.csv")
    return df


def class_balance(df):
    print("\n=== Class balance per split ===")
    table = df.groupby(["split", "stage1_label"]).size().unstack(fill_value=0)
    print(table)
    table.to_csv(REPORTS / "cov_caldas_class_balance_by_split.csv")
    return table


def same_source_sanity_check(df):
    print("\n=== THE KEY CHECK: age/sex by class (not by source — there's only one source) ===")
    print("If this pivot worked, positive and negative should look SIMILAR here —")
    print("unlike RICORD+ChestX-ray14, which showed a 13-year age gap between classes.")

    age_by_label = df.groupby("stage1_label")["age"].describe()
    print("\nAge by class:")
    print(age_by_label)
    age_by_label.to_csv(REPORTS / "cov_caldas_age_by_class.csv")

    sex_by_label = df.groupby(["stage1_label", "sex"]).size().unstack(fill_value=0)
    print("\nSex by class:")
    print(sex_by_label)
    sex_by_label.to_csv(REPORTS / "cov_caldas_sex_by_class.csv")

    # Flag explicitly if a meaningful gap still exists, rather than assuming
    # same-source automatically means no gap.
    means = df.groupby("stage1_label")["age"].mean()
    if len(means) == 2:
        gap = abs(means.iloc[0] - means.iloc[1])
        print(f"\nMean age gap between classes (all rows, including estimated ages): {gap:.1f} years")
        if gap > 5:
            print(f"[FLAG] Age gap is {gap:.1f} years — still notable even though same-source. "
                  f"Worth a sentence in Limitations regardless of source match.")
        else:
            print(f"[GOOD] Age gap is small ({gap:.1f} years) — consistent with a genuine "
                  f"same-source dataset, unlike the old pairing's 13-year gap.")

    # Recompute on clean-only ages, to see how much the estimated (multi-
    # session-mean) rows are moving the headline number one way or the other.
    if "age_is_estimated" in df.columns:
        clean = df[~df["age_is_estimated"]]
        means_clean = clean.groupby("stage1_label")["age"].mean()
        if len(means_clean) == 2:
            gap_clean = abs(means_clean.iloc[0] - means_clean.iloc[1])
            print(f"Mean age gap between classes (clean ages only, {len(clean)}/{len(df)} rows): {gap_clean:.1f} years")
            print(f"  -> difference from all-rows gap: {abs(gap - gap_clean):.1f} years")

        # The class-skew check: is age_is_estimated itself imbalanced by class?
        # This showed 18.6% of negatives vs 2.9% of positives in this dataset —
        # a second confound independent of age itself.
        est_by_label = df.groupby("stage1_label")["age_is_estimated"].agg(["sum", "count"])
        est_by_label["pct"] = 100 * est_by_label["sum"] / est_by_label["count"]
        print("\nage_is_estimated rate by class (checking for a SECOND confound, "
              "independent of the age value itself):")
        print(est_by_label)
        if est_by_label["pct"].max() - est_by_label["pct"].min() > 5:
            print("[FLAG] age_is_estimated rate differs notably by class — this is a second, "
                  "independent confound. Exclude age_is_estimated from any model features, "
                  "and mention it in Limitations alongside the age gap.")

    return age_by_label, sex_by_label


def patient_level_sanity(df):
    print("\n=== Patient-level sanity check ===")
    per_patient = df.groupby("patient_id").size()
    print(f"Images per patient: min={per_patient.min()}, max={per_patient.max()}, "
          f"mean={per_patient.mean():.2f}")
    outliers = per_patient[per_patient > per_patient.mean() + 3 * per_patient.std()]
    if len(outliers):
        print(f"[FLAG] {len(outliers)} patient(s) with unusually many images:")
        print(outliers)
    else:
        print("No extreme outliers in images-per-patient.")


if __name__ == "__main__":
    df = load()
    class_balance(df)
    same_source_sanity_check(df)
    patient_level_sanity(df)
    print(f"\nAll summary CSVs saved to {REPORTS}")