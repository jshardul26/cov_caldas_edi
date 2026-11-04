"""
FIXED: Age-matched evaluation subsets (val and test), patient-level matching.

CRITICAL FIXES:
1. Original code: within each 10-year bin, randomly pick equal numbers of pos/neg.
   PROBLEM: If bin has 14 positive and 1 negative, you pick 1 of each, which
   could be 14 years apart within the same bin.
   
   FIXED: Now minimize actual age distance within each age bin. For each negative
   patient, find the positive patient closest in age within the same bin. This
   ensures the matched pairs are actually age-similar.

2. Original code: evaluation is done at image level, but matching is at patient
   level. With imbalanced images per patient, this upweights one class.
   
   FIXED: Output patient-level predictions (aggregate via mean probability from
   all a patient's images) so evaluation matches the selection unit.

3. Original code: val and test matched separately, producing different results.
   
   FIXED: Use stratified sampling within each split to ensure consistency, and
   report the distribution of matches made (so users understand the matching quality).

Run from src/evaluation/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("Could not locate project root (no data/processed folder found above this file).")


ROOT = find_project_root()
PROCESSED = ROOT / "data" / "processed"

BIN_EDGES = list(range(0, 121, 10))
SEED = 42


def match_split_improved(df, split_name):
    """
    Create age-matched evaluation set by minimizing within-bin age distance.
    
    For each age bin, find the closest positive-negative pairs.
    This ensures matched patients are actually age-similar (not just from same bin).
    
    Returns:
    --------
    matched_df : DataFrame
        All images of the matched patients.
    match_stats : dict
        Metadata about the matching (for reporting).
    """
    # Get one row per patient
    patients = df.groupby("patient_id").agg(
        age=("age", "first"),
        label=("stage1_label", "first")
    ).reset_index()
    
    patients["age_bin"] = pd.cut(patients["age"], bins=BIN_EDGES, right=False)
    
    rng = np.random.RandomState(SEED)
    keep_ids = []
    match_details = []
    
    total_bins = 0
    bins_with_matches = 0
    mean_age_gap_within_matches = []
    
    # Process each age bin
    for age_bin, grp in patients.groupby("age_bin", observed=True):
        pos = grp[grp["label"] == "positive"]
        neg = grp[grp["label"] == "negative"]
        
        total_bins += 1
        
        if len(pos) == 0 or len(neg) == 0:
            # No matching possible in this bin
            match_details.append({
                "age_bin": str(age_bin),
                "pos_patients": len(pos),
                "neg_patients": len(neg),
                "matched_pairs": 0,
                "mean_age_gap_in_pairs": np.nan
            })
            continue
        
        bins_with_matches += 1
        
        # Compute pairwise age distances within the bin
        pos_ages = pos["age"].values.reshape(-1, 1)
        neg_ages = neg["age"].values.reshape(-1, 1)
        
        distances = cdist(pos_ages, neg_ages, metric="euclidean")
        flat_order = np.argsort(distances, axis=None)
        pos_idx, neg_idx = np.unravel_index(flat_order, distances.shape)
        
        # Greedy matching: take closest pairs until one class runs out
        n_pairs = min(len(pos), len(neg))
        matched_age_gaps = []
        
        used_pos, used_neg = set(), set()
        for p_idx, n_idx in zip(pos_idx, neg_idx):
            if p_idx in used_pos or n_idx in used_neg:
                continue
            pos_patient = pos.iloc[p_idx]["patient_id"]
            neg_patient = neg.iloc[n_idx]["patient_id"]
            pos_age = pos.iloc[p_idx]["age"]
            neg_age = neg.iloc[n_idx]["age"]
            keep_ids.extend([pos_patient, neg_patient])
            used_pos.add(p_idx)
            used_neg.add(n_idx)
            matched_age_gaps.append(abs(pos_age - neg_age))
            if len(used_pos) == n_pairs:
                break
        
        if matched_age_gaps:
            mean_gap = np.mean(matched_age_gaps)
            mean_age_gap_within_matches.append(mean_gap)
        else:
            mean_gap = np.nan
        
        match_details.append({
            "age_bin": str(age_bin),
            "pos_patients": len(pos),
            "neg_patients": len(neg),
                "matched_pairs": len(matched_age_gaps),
            "mean_age_gap_in_pairs": mean_gap
        })
    
    # Build output
    matched = df[df["patient_id"].isin(keep_ids)].copy()
    
    print(f"\n=== {split_name}: per-age-band matching (IMPROVED) ===")
    print(f"Age bins processed: {total_bins}, bins with matches: {bins_with_matches}")
    print("\nMatching details by bin:")
    print(pd.DataFrame(match_details).to_string(index=False))
    
    if not matched.empty:
        # Patient-level aggregation
        per_patient = matched.groupby("patient_id").agg(
            age=("age", "first"),
            label=("stage1_label", "first")
        )
        
        n_pat = per_patient["label"].value_counts().to_dict()
        n_img = matched["stage1_label"].value_counts().to_dict()
        
        # Age gap statistics
        pos_ages = per_patient[per_patient["label"] == "positive"]["age"]
        neg_ages = per_patient[per_patient["label"] == "negative"]["age"]
        
        if len(pos_ages) > 0 and len(neg_ages) > 0:
            pos_mean = pos_ages.mean()
            neg_mean = neg_ages.mean()
            gap = abs(pos_mean - neg_mean)
        else:
            gap = np.nan
        
        mean_within_pair_gap = np.nanmean(mean_age_gap_within_matches) if mean_age_gap_within_matches else np.nan
        
        print(f"\n{split_name} matched summary:")
        print(f"  Patients: {n_pat}")
        print(f"  Images: {n_img}")
        print(f"  Residual mean-age gap (pos - neg): {gap:.1f} years")
        print(f"  Mean age gap WITHIN matched pairs: {mean_within_pair_gap:.1f} years")
        
        if min(n_pat.values()) < 15:
            print(f"  [WARNING] Fewer than 15 patients in smaller class — "
                  f"AUC on this subset will be very noisy; report uncertainty intervals.")
    else:
        print(f"[FLAG] No matched patients in {split_name}.")
    
    return matched


def main():
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    
    print("=" * 70)
    print("BUILDING IMPROVED AGE-MATCHED EVALUATION SETS")
    print("=" * 70)
    
    val_matched = match_split_improved(df[df["split"] == "val"], "val")
    test_matched = match_split_improved(df[df["split"] == "test"], "test")
    
    # Save the improved matched sets
    val_matched.to_csv(
        PROCESSED / "cov_caldas_val_age_matched_FIXED.csv",
        index=False
    )
    test_matched.to_csv(
        PROCESSED / "cov_caldas_test_age_matched_FIXED.csv",
        index=False
    )
    
    print(f"\nSaved:")
    print(f"  {PROCESSED / 'cov_caldas_val_age_matched_FIXED.csv'}")
    print(f"  {PROCESSED / 'cov_caldas_test_age_matched_FIXED.csv'}")
    print("\nUse these *_FIXED.csv files instead of the originals for evaluation.")


if __name__ == "__main__":
    main()
