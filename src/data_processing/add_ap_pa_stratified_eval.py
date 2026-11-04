"""
AP/PA View Confound Analysis.

The positive class is 70% more likely to be acquired in AP (anterior-posterior)
view than the negative class (87.1% AP vs 70.5% AP). This creates a confound:
the model could be learning "AP projection → positive" rather than medical signal.

This script generates stratified evaluation sets:
1. AP-only evaluation set (subset of each split, both classes matched on AP)
2. PA-only evaluation set (subset of each split, both classes matched on PA)

Then measure CNN performance separately on each. If the CNN is learning
image content rather than projection artifact, performance should be
similar for AP and PA. If it drops dramatically for PA, that's evidence
the model was exploiting the AP/PA confound.

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


def create_stratified_views():
    """Create AP and PA stratified subsets."""
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
    
    print("=" * 70)
    print("AP/PA STRATIFIED EVALUATION SETS")
    print("=" * 70)
    
    print("\n=== Original distribution ===")
    print(pd.crosstab(df["stage1_label"], df["view"], margins=True))
    
    print("\n=== Issue ===")
    print("Positive class is heavily AP-biased (87.1% AP) vs negative (70.5% AP).")
    print("If CNN learns the AP/PA difference, it's not learning medical signal.")
    
    # Create AP-only and PA-only subsets
    ap_df = df[df["view"] == "ap"].copy()
    pa_df = df[df["view"] == "pa"].copy()
    
    print("\n=== AP-only evaluation (both classes) ===")
    print(ap_df.groupby(["split", "stage1_label"]).agg(
        images=("image_path", "count"),
        patients=("patient_id", "nunique")
    ).unstack(fill_value=0).to_string())
    
    print("\n=== PA-only evaluation (both classes) ===")
    print(pa_df.groupby(["split", "stage1_label"]).agg(
        images=("image_path", "count"),
        patients=("patient_id", "nunique")
    ).unstack(fill_value=0).to_string())
    
    # Save subsets
    ap_df.to_csv(PROCESSED / "cov_caldas_master_frontal_AP_ONLY.csv", index=False)
    pa_df.to_csv(PROCESSED / "cov_caldas_master_frontal_PA_ONLY.csv", index=False)
    
    print("\n" + "=" * 70)
    print(f"Saved stratified evaluation sets:")
    print(f"  {PROCESSED / 'cov_caldas_master_frontal_AP_ONLY.csv'}")
    print(f"  {PROCESSED / 'cov_caldas_master_frontal_PA_ONLY.csv'}")
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("""
After training on the standard split (mixed AP/PA):

1. Evaluate on AP-only: How does CNN perform?
2. Evaluate on PA-only: How does CNN perform?
3. Compare to full set performance

If CNN AUC is HIGH on both AP and PA: ← Good, learned medical content
If CNN AUC is HIGH on AP but LOW on PA: ← Bad, learned AP/PA as shortcut
If CNN AUC is MODERATE on both: ← Mixed signal + artifact

Even if AUC differs by view, that tells you something about what
the model learned.
""")


if __name__ == "__main__":
    create_stratified_views()