import pandas as pd
import numpy as np
from PIL import Image

CSV = r"D:\EDI PROJECT 5\data\processed\cov_caldas_master_frontal.csv"

df = pd.read_csv(CSV)

rows = []

print("Checking native image properties...")

for i, r in enumerate(df.itertuples(index=False), 1):
    img = Image.open(r.image_path)
    arr = np.asarray(img)

    rows.append({
        "stage1_label": r.stage1_label,
        "split": r.split,
        "view": r.view,
        "dtype": str(arr.dtype),
        "ndim": arr.ndim,
        "width": arr.shape[1],
        "height": arr.shape[0],
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    })

    if i % 500 == 0:
        print(f"{i}/{len(df)}")

out = pd.DataFrame(rows)

print("\n" + "=" * 70)
print("NATIVE DTYPE BY CLASS")
print("=" * 70)
print(pd.crosstab(out["stage1_label"], out["dtype"]))

print("\n" + "=" * 70)
print("NATIVE PIXEL STATISTICS BY CLASS")
print("=" * 70)

print(
    out.groupby("stage1_label")[
        ["min", "max", "mean", "std", "width", "height"]
    ].agg(["mean", "median", "min", "max"]).round(2).to_string()
)

print("\n" + "=" * 70)
print("NATIVE DTYPE + VIEW")
print("=" * 70)

print(
    pd.crosstab(
        [out["stage1_label"], out["view"]],
        out["dtype"]
    ).to_string()
)

out.to_csv(
    r"D:\EDI PROJECT 5\reports\eda\cov_caldas_native_image_properties.csv",
    index=False
)

print("\nSaved:")
print(r"D:\EDI PROJECT 5\reports\eda\cov_caldas_native_image_properties.csv")
