import pandas as pd

MASTER = r"D:\EDI PROJECT 5\data\processed\cov_caldas_master_frontal.csv"
META = r"D:\EDI PROJECT 5\reports\eda\cov_caldas_dicom_metadata_analysis.csv"

master = pd.read_csv(MASTER)
meta = pd.read_csv(META)

meta["patient_id"] = master["patient_id"].values

print("=" * 90)
print("GE NEGATIVE PATIENT / ACQUISITION INSPECTION")
print("=" * 90)

ge = meta[
    (meta["manufacturer"] == "GE Healthcare") &
    (meta["bits_stored"] == 14)
].copy()

print("\nOverall GE/14-bit:")
print(
    ge.groupby("stage1_label")
    .agg(
        images=("stage1_label", "size"),
        patients=("patient_id", "nunique")
    )
    .to_string()
)

print("\nGE/14-bit by class + view:")
print(
    ge.groupby(["stage1_label", "view_position"])
    .agg(
        images=("stage1_label", "size"),
        patients=("patient_id", "nunique")
    )
    .to_string()
)

print("\nGE/14-bit by class + model:")
print(
    ge.groupby(["stage1_label", "model"])
    .agg(
        images=("stage1_label", "size"),
        patients=("patient_id", "nunique")
    )
    .to_string()
)

print("\nGE/14-bit by class + software:")
print(
    ge.groupby(["stage1_label", "software"])
    .agg(
        images=("stage1_label", "size"),
        patients=("patient_id", "nunique")
    )
    .to_string()
)

print("\nGE/14-bit NEGATIVE PATIENTS:")
neg = ge[ge["stage1_label"] == "negative"]

print(
    neg[
        [
            "patient_id",
            "view_position",
            "model",
            "software",
            "bits_stored",
            "photometric",
            "presentation_lut"
        ]
    ]
    .drop_duplicates()
    .sort_values("patient_id")
    .to_string(index=False)
)

print("\n" + "=" * 90)
print("SPLIT CHECK")
print("=" * 90)

print(
    ge.groupby(["split", "stage1_label"])
    .agg(
        images=("stage1_label", "size"),
        patients=("patient_id", "nunique")
    )
    .to_string()
)
