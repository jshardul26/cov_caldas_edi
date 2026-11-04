"""
Phase 5 validation - can the label be predicted from NON-ANATOMICAL image
properties alone?

History: the first CNN scored test AUC 1.0000, and this check (run on
unnormalized images) scored 1.0000 too. Cause found: positives are 16-bit,
negatives 8-bit, and a plain 8-bit conversion clips the positives to white.

This version loads every image through utils/image_io.py (identical
intensity normalization for both classes) and asks two questions:
  A) raw file properties + pixel features   (size, file size, brightness...)
  B) PIXEL-ONLY features after normalization (brightness/border stats plus
     an 8x8 thumbnail) - this is closest to what the CNN can actually see.
If B is still high, the bit-depth fix was not enough: the classes differ in
layout, contrast or field of view, and retraining would just relearn that.
If B is low, bit depth was the main leak and retraining is worthwhile.
A low AUC here does NOT clear the CNN - it only rules out THESE features.

Run from src/eda/.
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
from image_io import load_normalized_image  # noqa: E402

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
    rf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                class_weight="balanced", random_state=42)
    rf.fit(train[features], (train["stage1_label"] == "positive").astype(int))
    out = {}
    for split in ["val", "test"]:
        sub = feat_df[feat_df["split"] == split]
        y = (sub["stage1_label"] == "positive").astype(int)
        out[split] = roc_auc_score(y, rf.predict_proba(sub[features])[:, 1])
    print(f"  {name:<34} val AUC={out['val']:.4f}   test AUC={out['test']:.4f}")
    return rf


def main():
    df = pd.read_csv(PROCESSED / "cov_caldas_master_frontal.csv")
    print(f"Extracting features (normalized loader) from {len(df)} images...")
    rows = []
    for i, r in enumerate(df.itertuples(index=False), 1):
        feats = image_features(r.image_path)
        feats.update(patient_id=r.patient_id, stage1_label=r.stage1_label, split=r.split)
        rows.append(feats)
        if i % 500 == 0:
            print(f"  {i}/{len(df)}", flush=True)
    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(REPORTS / "cov_caldas_image_features_normalized.csv", index=False)

    print("\n=== Mean of summary features by class (after normalization) ===")
    print(feat_df.groupby("stage1_label")[RAW_FEATURES + STAT_FEATURES].mean().T.round(2).to_string())

    train = feat_df[feat_df["split"] == "train"]
    print("\n=== Non-anatomical baselines (Random Forest, trained on TRAIN only) ===")
    fit_and_report("A) raw size/file + pixel features", ALL_FEATURES, feat_df, train)
    rf_b = fit_and_report("B) PIXEL-ONLY after normalization", PIXEL_FEATURES, feat_df, train)

    imp = pd.Series(rf_b.feature_importances_, index=PIXEL_FEATURES).sort_values(ascending=False)
    print("\nTop pixel-only feature importances (B):")
    print(imp.head(5).round(3).to_string())
    print("\nReading it: B high = classes still differ in layout/contrast/field of view "
          "after fixing bit depth; B low = bit depth was the main leak. Neither proves "
          "what a CNN would learn.")


if __name__ == "__main__":
    main()