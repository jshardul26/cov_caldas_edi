import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "data" / "processed" / "cov_caldas_master_frontal.csv"
META = ROOT / "reports" / "eda" / "cov_caldas_dicom_metadata_analysis.csv"

master = pd.read_csv(MASTER)
meta = pd.read_csv(META)

if "image_path" not in meta:
    raise KeyError("Metadata table has no image_path; rerun analyze_dicom_metadata.py")
meta = meta.merge(master[["image_path", "patient_id"]], on="image_path", how="left", validate="one_to_one")
if meta.patient_id.isna().any():
    raise RuntimeError("Metadata rows failed to join to patient IDs")

ge = meta[
    (meta["manufacturer"] == "GE Healthcare") &
    (meta["bits_stored"] == 14)
].copy()

print("=" * 90)
print("PATIENT-LEVEL ACQUISITION-CONTROLLED SPLIT FEASIBILITY")
print("=" * 90)

patients = (
    ge.groupby(["patient_id", "stage1_label"])
    .agg(
        images=("stage1_label", "size"),
        views=("view_position", lambda x: ",".join(sorted(set(x.dropna().astype(str)))))
    )
    .reset_index()
)

print("\nGE/14-bit patients by class:")
print(
    patients.groupby("stage1_label")
    .agg(
        patients=("patient_id", "nunique"),
        images=("images", "sum")
    )
    .to_string()
)

print("\n" + "=" * 90)
print("CURRENT SPLIT - PATIENT LEVEL")
print("=" * 90)

print(
    ge.groupby(["split", "stage1_label"])
    .agg(
        patients=("patient_id", "nunique"),
        images=("stage1_label", "size")
    )
    .to_string()
)

print("\n" + "=" * 90)
print("NEGATIVE GE/14-BIT PATIENTS")
print("=" * 90)

neg_patients = patients[patients["stage1_label"] == "negative"].copy()

print(
    neg_patients[
        ["patient_id", "images", "views"]
    ]
    .sort_values("patient_id")
    .to_string(index=False)
)

print("\n" + "=" * 90)
print("POSITIVE GE/14-BIT PATIENT COUNTS BY VIEW")
print("=" * 90)

print(
    ge[ge["stage1_label"] == "positive"]
    .groupby("view_position")
    .agg(
        patients=("patient_id", "nunique"),
        images=("stage1_label", "size")
    )
    .to_string()
)

print("\n" + "=" * 90)
print("NEGATIVE GE/14-BIT COUNTS BY VIEW")
print("=" * 90)

print(
    ge[ge["stage1_label"] == "negative"]
    .groupby("view_position")
    .agg(
        patients=("patient_id", "nunique"),
        images=("stage1_label", "size")
    )
    .to_string()
)

print("\n" + "=" * 90)
print("CONTROLLED SPLIT FEASIBILITY")
print("=" * 90)

n_neg = neg_patients["patient_id"].nunique()

print(f"Negative GE/14-bit patients available: {n_neg}")
print("Minimum needed for train/val/test: at least 3 patients.")
print("With only 13 patients, the controlled test estimate will have high uncertainty.")
print("Patient-level separation must be preserved.")
