import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "processed" / "cov_caldas_master_frontal.csv"

df = pd.read_csv(CSV)

rows = []

for i, r in enumerate(df.itertuples(index=False), 1):
    arr = np.asarray(Image.open(r.image_path))

    if arr.dtype == np.uint16:
        rows.append({
            "stage1_label": r.stage1_label,
            "split": r.split,
            "view": r.view,
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "p99": float(np.percentile(arr, 99)),
            "frac_above_255": float((arr > 255).mean()),
            "frac_above_4095": float((arr > 4095).mean()),
            "frac_above_8191": float((arr > 8191).mean()),
            "frac_above_16383": float((arr > 16383).mean()),
        })

out = pd.DataFrame(rows)

if out.empty:
    print("No uint16 images found in the frontal table; no range summary to compute.")
    raise SystemExit(0)

print("=" * 70)
print("UINT16 IMAGE RANGE CHECK")
print("=" * 70)

print(f"uint16 images: {len(out)}")

print("\nMAX VALUE DISTRIBUTION:")
print(out["max"].describe().round(2).to_string())

print("\nMAX VALUE COUNTS:")
print(out["max"].value_counts().head(20).to_string())

print("\nSTATISTICS:")
print(
    out[
        [
            "max",
            "mean",
            "p99",
            "frac_above_255",
            "frac_above_4095",
            "frac_above_8191",
            "frac_above_16383",
        ]
    ].describe().round(4).to_string()
)

print("\nBY CLASS:")
print(
    out.groupby("stage1_label")[
        [
            "max",
            "mean",
            "p99",
            "frac_above_255",
            "frac_above_4095",
            "frac_above_8191",
        ]
    ].mean().round(4).to_string()
)

print("\nBY VIEW:")
print(
    out.groupby(["stage1_label", "view"])[
        [
            "max",
            "mean",
            "p99",
            "frac_above_255",
            "frac_above_4095",
            "frac_above_8191",
        ]
    ].mean().round(4).to_string()
)

