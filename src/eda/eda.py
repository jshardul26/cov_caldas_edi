"""
Phase 3 — EDA on the frozen split.

Reads data/processed/master_table.csv (built in Phase 2) and produces:
  - Class balance per split (already partly known from Phase 2's own output,
    reported here alongside everything else for one single reference report)
  - Missingness report (age/sex/appearance/severity — no fabricated stats,
    every number below is computed straight from the file, never assumed)
  - Age/sex distribution, split out by source dataset (this is where the
    domain-shift signal from the Decisions Log becomes visible, if present)
  - Severity/appearance label distribution among RICORD positives only

Run from src/eda/. Outputs a CSV summary (for the paper's tables) and
prints everything to console too.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)


def load():
    df = pd.read_csv(PROCESSED / "master_table.csv")
    print(f"Loaded {len(df)} rows from master_table.csv")
    return df


def class_balance(df):
    print("\n=== Class balance per split ===")
    table = df.groupby(["split", "stage1_label"]).size().unstack(fill_value=0)
    print(table)
    table.to_csv(REPORTS / "class_balance_by_split.csv")
    return table


def missingness(df):
    print("\n=== Missingness (real counts, not assumed) ===")
    cols = ["age", "sex", "appearance", "severity"]
    miss = df[cols].isna().sum()
    miss_pct = (miss / len(df) * 100).round(1)
    report = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
    print(report)
    report.to_csv(REPORTS / "missingness.csv")

    # Note on expected missingness, so nobody misreads this as a data bug:
    print("\nNote: 'severity' missing is EXPECTED for all ChestX-ray14 rows (no")
    print("severity concept for that dataset) and for RICORD rows labeled")
    print("'Negative for Pneumonia' / 'Invalid Study' (no opacities to grade).")
    return report


def demographics_by_source(df):
    print("\n=== Age/sex distribution by source dataset ===")
    print("(This is the check for the domain-shift risk already logged in")
    print(" 00_PROJECT_MASTER.md — if age/sex differ sharply by source, that's")
    print(" the confound becoming visible, and it belongs in the paper's")
    print(" Limitations section, not hidden.)")

    age_summary = df.groupby("source")["age"].describe()
    print("\nAge by source:")
    print(age_summary)
    age_summary.to_csv(REPORTS / "age_by_source.csv")

    sex_summary = df.groupby(["source", "sex"]).size().unstack(fill_value=0)
    print("\nSex by source:")
    print(sex_summary)
    sex_summary.to_csv(REPORTS / "sex_by_source.csv")

    return age_summary, sex_summary


def severity_distribution(df):
    print("\n=== Appearance/severity distribution (RICORD positives only) ===")
    ricord = df[df["source"] == "ricord"]

    appearance_counts = ricord["appearance"].value_counts(dropna=False)
    print("\nAppearance categories:")
    print(appearance_counts)
    appearance_counts.to_csv(REPORTS / "appearance_distribution.csv")

    severity_counts = ricord["severity"].value_counts(dropna=False)
    print("\nSeverity grades:")
    print(severity_counts)
    severity_counts.to_csv(REPORTS / "severity_distribution.csv")

    return appearance_counts, severity_counts


def patient_level_sanity(df):
    print("\n=== Patient-level sanity check ===")
    per_patient = df.groupby("patient_id").size()
    print(f"Images per patient: min={per_patient.min()}, max={per_patient.max()}, "
          f"mean={per_patient.mean():.2f}")
    # Flag any patient with an unusually high image count — worth a manual
    # look before training, not necessarily an error.
    outliers = per_patient[per_patient > per_patient.mean() + 3 * per_patient.std()]
    if len(outliers):
        print(f"[FLAG] {len(outliers)} patient(s) with unusually many images — worth a manual check:")
        print(outliers)
    else:
        print("No extreme outliers in images-per-patient.")


if __name__ == "__main__":
    df = load()
    class_balance(df)
    missingness(df)
    demographics_by_source(df)
    severity_distribution(df)
    patient_level_sanity(df)
    print(f"\nAll summary CSVs saved to {REPORTS}")