import json
import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV = PROJECT_ROOT / "data" / "processed" / "cov_caldas_master_frontal.csv"

df = pd.read_csv(CSV)

# Metadata tags we specifically care about
TAGS = {
    "00080070": "manufacturer",
    "00081090": "model",
    "00181020": "software",
    "00185101": "view_position",
    "00280002": "samples_per_pixel",
    "00280004": "photometric",
    "00280100": "bits_allocated",
    "00280101": "bits_stored",
    "00280102": "high_bit",
    "00280103": "pixel_representation",
    "00281050": "window_center",
    "00281051": "window_width",
    "00281052": "rescale_intercept",
    "00281053": "rescale_slope",
    "00281054": "rescale_type",
    "20500020": "presentation_lut",
}

def get_value(d, tag):
    x = d.get(tag, {})
    if "Value" not in x:
        return None

    v = x["Value"]

    if isinstance(v, list) and len(v) == 1:
        return v[0]

    return str(v)

rows = []

print("=" * 80)
print("READING DICOM METADATA FROM COV-CALDAS JSON")
print("=" * 80)

for i, r in enumerate(df.itertuples(index=False), 1):

    image_path = r.image_path

    # PNG -> JSON with same filename
    json_path = os.path.splitext(image_path)[0] + ".json"

    if not os.path.exists(json_path):
        rows.append({
            "image_path": image_path,
            "stage1_label": r.stage1_label,
            "split": r.split,
            "view": r.view,
            "json_found": False
        })
        continue

    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    row = {
        "image_path": image_path,
        "stage1_label": r.stage1_label,
        "split": r.split,
        "view": r.view,
        "json_found": True
    }

    for tag, name in TAGS.items():
        row[name] = get_value(d, tag)

    rows.append(row)

    if i % 500 == 0:
        print(f"{i}/{len(df)}")

out = pd.DataFrame(rows)

print("\n" + "=" * 80)
print("JSON COVERAGE")
print("=" * 80)
print(out["json_found"].value_counts())

for name in TAGS.values():

    print("\n" + "=" * 80)
    print(f"{name.upper()} BY CLASS")
    print("=" * 80)

    print(
        pd.crosstab(
            out["stage1_label"],
            out[name].fillna("<missing>")
        ).to_string()
    )

print("\n" + "=" * 80)
print("KEY NUMERIC METADATA BY CLASS")
print("=" * 80)

numeric_cols = [
    "samples_per_pixel",
    "bits_allocated",
    "bits_stored",
    "high_bit",
    "pixel_representation",
    "window_center",
    "window_width",
    "rescale_intercept",
    "rescale_slope",
]

for c in numeric_cols:
    out[c] = pd.to_numeric(out[c], errors="coerce")

print(
    out.groupby("stage1_label")[numeric_cols]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
    .to_string()
)

print("\n" + "=" * 80)
print("KEY METADATA BY CLASS + VIEW")
print("=" * 80)

print(
    out.groupby(["stage1_label", "view"])[
        [
            "bits_allocated",
            "bits_stored",
            "window_center",
            "window_width",
            "rescale_intercept",
            "rescale_slope"
        ]
    ]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
    .to_string()
)

OUTPUT = PROJECT_ROOT / "reports" / "eda" / "cov_caldas_dicom_metadata_analysis.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

out.to_csv(OUTPUT, index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)
print(OUTPUT)
