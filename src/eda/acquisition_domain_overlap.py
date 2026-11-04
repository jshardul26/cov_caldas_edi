import pandas as pd

CSV = r"D:\EDI PROJECT 5\reports\eda\cov_caldas_dicom_metadata_analysis.csv"

df = pd.read_csv(CSV)

print("=" * 90)
print("ACQUISITION DOMAIN OVERLAP ANALYSIS")
print("=" * 90)

# The metadata analysis file doesn't contain patient_id,
# so merge it back from the master dataset.
MASTER = r"D:\EDI PROJECT 5\data\processed\cov_caldas_master_frontal.csv"

master = pd.read_csv(MASTER)

# Match rows using image_path if available in both files.
# The metadata analysis file currently does not contain image_path,
# so reconstruct alignment using the original row order.
if len(master) != len(df):
    raise RuntimeError(
        f"Row count mismatch: master={len(master)}, metadata={len(df)}"
    )

df["patient_id"] = master["patient_id"].values
df["image_path"] = master["image_path"].values

# Acquisition-domain combinations
DOMAIN_SETS = [
    ["manufacturer"],
    ["model"],
    ["software"],
    ["bits_stored"],
    ["manufacturer", "model"],
    ["manufacturer", "bits_stored"],
    ["manufacturer", "model", "bits_stored"],
    ["manufacturer", "model", "software"],
    ["manufacturer", "model", "software", "bits_stored"],
]

for cols in DOMAIN_SETS:

    print("\n" + "=" * 90)
    print("DOMAIN:", " + ".join(cols))
    print("=" * 90)

    tmp = df.copy()

    # Make missing metadata explicit
    for c in cols:
        tmp[c] = tmp[c].fillna("<MISSING>").astype(str)

    summary = (
        tmp.groupby(cols)
        .agg(
            images=("stage1_label", "size"),
            patients=("patient_id", "nunique"),
            positive_images=(
                "stage1_label",
                lambda x: (x == "positive").sum()
            ),
            negative_images=(
                "stage1_label",
                lambda x: (x == "negative").sum()
            ),
            positive_patients=(
                "stage1_label",
                lambda x: tmp.loc[x.index, "patient_id"][
                    x == "positive"
                ].nunique()
            ),
            negative_patients=(
                "stage1_label",
                lambda x: tmp.loc[x.index, "patient_id"][
                    x == "negative"
                ].nunique()
            ),
        )
        .reset_index()
    )

    # Positive + negative domains are the important ones.
    overlap = summary[
        (summary["positive_images"] > 0) &
        (summary["negative_images"] > 0)
    ].copy()

    if len(overlap) == 0:
        print("NO DOMAIN CONTAINS BOTH CLASSES.")
    else:
        print("\nDomains containing BOTH classes:")
        print(
            overlap.sort_values(
                ["positive_images", "negative_images"],
                ascending=False
            ).to_string(index=False)
        )

print("\n" + "=" * 90)
print("VIEW + ACQUISITION DOMAIN")
print("=" * 90)

cols = ["manufacturer", "model", "bits_stored", "view"]

tmp = df.copy()

for c in cols:
    tmp[c] = tmp[c].fillna("<MISSING>").astype(str)

summary = (
    tmp.groupby(cols)
    .agg(
        images=("stage1_label", "size"),
        patients=("patient_id", "nunique"),
        positive_images=(
            "stage1_label",
            lambda x: (x == "positive").sum()
        ),
        negative_images=(
            "stage1_label",
            lambda x: (x == "negative").sum()
        ),
        positive_patients=(
            "stage1_label",
            lambda x: tmp.loc[x.index, "patient_id"][
                x == "positive"
            ].nunique()
        ),
        negative_patients=(
            "stage1_label",
            lambda x: tmp.loc[x.index, "patient_id"][
                x == "negative"
            ].nunique()
        ),
    )
    .reset_index()
)

overlap = summary[
    (summary["positive_images"] > 0) &
    (summary["negative_images"] > 0)
]

print(
    overlap.sort_values(
        ["positive_images", "negative_images"],
        ascending=False
    ).to_string(index=False)
)

OUTPUT = r"D:\EDI PROJECT 5\reports\eda\acquisition_domain_overlap.csv"
summary.to_csv(OUTPUT, index=False)

print("\nSaved:")
print(OUTPUT)
