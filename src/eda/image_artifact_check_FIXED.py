"""
FIXED: Phase 5 validation - detect non-anatomical image artifacts.

The original version applied per-image percentile normalization, which
masked (didn't eliminate) the uint16 vs uint8 artifact by encoding it
differently. This version:

1. Computes normalization stats from TRAINING DATA ONLY
2. Applies identical normalization to all images
3. Then checks if images still separate by intensity/technical features

If features still predict class after proper normalization, either:
  A) Classes differ in genuine image content (medical signal) ← good
  B) There's a residual acquisition artifact (exposure, processing) ← bad
  C) Data quality differences (compression, resolution) ← bad

Run from src/eda/ AFTER setting up normalization stats.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def find_project_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "processed").exists():
            return parent
    raise RuntimeError("Could not locate project root (no data/processed folder found above this file).")


ROOT = find_project_root()
sys.path.insert(0, str(ROOT / "src" / "utils"))
from image_io import (
    load_normalized_image,
    set_normalization_stats,
    _compute_normalization_stats
)

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)

RAW_FEATURES = ["width", "height", "aspect", "file_kb"]
STAT_FEATURES = ["mean", "std", "p1", "p99", "border_mean", "border_std",
                 "center_mean", "frac_dark", "frac_bright"]
THUMB_FEATURES = [f"px{i}" for i in range(64)]
PIXEL_FEATURES = STAT_FEATURES + THUMB_FEATURES
ALL_FEATURES = RAW_FEATURES + PIXEL_FEATURES


def image_features(path):
    """Extract technical features from a single image (after normalization)."""
    p = Path(path)
    w, h = Image.open(p).size
    img = load_normalized_image(p)
    g = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    m = int(256 * 0.08)
    border = np.concatenate([g[:m].ravel(), g[-m:].ravel(), g[:, :m].ravel(), g[:, -m:].ravel()])
    thumb = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float32).ravel()
    feats = {
        "width": w, "height": h, "aspect": w / h, "file_kb": p.stat().st_size / 1024,
        "mean": g.mean(), "std": g.std(),
        "p1": np.percentile(g, 1), "p99": np.percentile(g, 99),
        "border_mean": border.mean(), "border_std": border.std(),
        "center_mean": g[64:192, 64:192].mean(),
        "frac_dark": (g <= 5).mean(), "frac_bright": (g >= 250).mean(),
    }
    feats.update({f"px{i}": v for i, v in enumerate(thumb)})
    return feats


def fit_and_report(name, features, feat_df, train):
    """Train RF on features and report val/test AUC."""
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight="balanced", random_state=42)
    rf.fit(train[features], (train["stage1_label"] == "positive").astype(int))
    out = {}
    for split in ["val", "test"]:
        sub = feat_df[feat_df["split"] == split]
        y = (sub["stage1_label"] == "positive").astype(int)
        if len(y) > 0 and y.nunique() > 1:
            out[split] = roc_auc_score(y, rf.predict_proba(sub[features])[:, 1])
        else:
            out[split] = np.nan
    print(f"  {name:<40} val AUC={out['val']:.4f}   test AUC={out['test']:.4f}")
    return rf


def main():
    print("=" * 70)
    print("ARTIFACT ANALYSIS (using FIXED dataset-level normalization)")
    print("=" * 70)
    
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
    
    # Compute and set normalization statistics from training data
    train_df = df[df["split"] == "train"]
    print(f"\nComputing normalization statistics from {len(train_df)} training images...")
    lo, hi = _compute_normalization_stats(train_df)
    set_normalization_stats(lo, hi)
    
    # Extract features from all images using fixed normalization
    print(f"\nExtracting features from {len(df)} images using fixed normalization...")
    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        feats = image_features(r.image_path)
        feats.update(patient_id=r.patient_id, stage1_label=r.stage1_label, split=r.split)
        rows.append(feats)
        if i % 500 == 0:
            print(f"  {i}/{len(df)}", flush=True)
    
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(REPORTS / "cov_caldas_image_features_normalized_FIXED.csv", index=False)
    
    print("\n=== Mean of summary features by class (FIXED normalization) ===")
    print(feat_df.groupby("stage1_label")[RAW_FEATURES + STAT_FEATURES].mean().T.round(2).to_string())
    
    train = feat_df[feat_df["split"] == "train"]
    print("\n=== Non-anatomical baselines (Random Forest, trained on TRAIN only) ===")
    fit_and_report("A) raw size/file + pixel features", ALL_FEATURES, feat_df, train)
    rf_b = fit_and_report("B) PIXEL-ONLY after FIXED normalization", PIXEL_FEATURES, feat_df, train)
    
    imp = pd.Series(rf_b.feature_importances_, index=PIXEL_FEATURES).sort_values(ascending=False)
    print("\nTop 5 pixel-only feature importances (B):")
    print(imp.head(5).round(3).to_string())
    
    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print("""
If B is HIGH (AUC > 0.80):
  → Classes still differ significantly in pixel statistics after fixing normalization
  → Either: (1) genuine anatomical difference (medical signal) or
  →         (2) residual acquisition artifact (exposure, processing, filtering)
  → Next step: Inspect Grad-CAM to determine which

If B is LOW (AUC < 0.70):
  → Pixel statistics alone don't predict class after proper normalization
  → Artifact was successfully eliminated
  → High CNN AUC would then suggest genuine medical learning
  
If B is MODERATE (0.70-0.80):
  → Ambiguous; mixed artifact + medical signal
  → Grad-CAM + domain expertise required to interpret
  → Consider stratified evaluation (by view, by protocol) to disentangle
""")


if __name__ == "__main__":
    main()
