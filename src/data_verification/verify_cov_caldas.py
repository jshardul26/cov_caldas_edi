"""
Cov-Caldas verification script (Stage 1 dataset, same-source positive+negative).

Confirmed facts this script relies on (verified against real extracted data,
see 00_PROJECT_MASTER.md decisions log):
  - Structure: data/raw/cov_caldas/{positive,negative}/MIDS/sub-<ID>/ses-<ID>/...
  - Metadata: participants.tsv (tab-separated) at the MIDS root of each class,
    columns: participant, modality_dicom, body_parts, age, gender
  - Some participant rows have age/gender = "None" (metadata withheld for
    those subjects) — these must be filtered out, not treated as errors.
  - Public release (4,468 images total) is smaller than the paper's reported
    collection (7,307) — this is expected (their stated anonymization/
    selection process), not a corrupted download. Many session folders on
    disk contain no PNG at all — also expected, not an error.
  - participants.tsv has exactly ONE row per participant (confirmed via
    check_age_confound_scope.py — no duplicate participant IDs in either
    class file). That single row's 'age' field is sometimes a LIST of ages
    (e.g. "[56, 57]") for participants imaged across multiple ses-<ID>
    sessions at different ages — this is a property of the source metadata
    (one row per participant, not per session), not a bug in this script.
    595/4468 images (13.3%) are affected, and it skews heavily by class:
    18.6% of negative-class images vs. 2.9% of positive-class images have
    an ambiguous multi-value age. Documented as a limitation regardless of
    the resolution strategy below.

Run from src/data_verification/.
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data" / "raw" / "cov_caldas"

AGE_NUM_PATTERN = re.compile(r"\d+")


def resolve_age(raw_age):
    """Turn a raw participants.tsv age value into (numeric_age, is_estimated).

    Handles three cases:
      - clean single value, e.g. 61 or "61"          -> (61.0, False)
      - bracketed single value, e.g. "[61]"           -> (61.0, False)
      - bracketed/plain multi-value, e.g. "[56, 57]"
        or "56, 57"                                   -> (mean, True)

    Mean-of-listed-values is a documented approximation, not a guess dressed
    up as fact — every row this touches is flagged via is_estimated so it
    can be excluded from analyses (like the metadata-only shortcut-learning
    sanity check) where an approximate age could itself become a confound.
    """
    if not isinstance(raw_age, str):
        # already numeric (int/float) from pandas — nothing to resolve
        return raw_age, False

    nums = [int(n) for n in AGE_NUM_PATTERN.findall(raw_age)]
    if not nums:
        return None, False
    if len(nums) == 1:
        return float(nums[0]), False
    return sum(nums) / len(nums), True


def load_participants(cls):
    path = DATA / cls / "MIDS" / "participants.tsv"
    if not path.exists():
        print(f"  [MISSING] {path}")
        return None
    df = pd.read_csv(path, sep="\t")
    print(f"  {cls}: {len(df)} participant rows in participants.tsv")
    return df


def filter_usable_metadata(df, cls):
    # pandas treats the literal text "None" as a missing-value token and
    # silently converts it to NaN on read — so checking `!= "None"` never
    # matches anything. Use pd.isna() to actually catch these rows.
    usable = df[df["age"].notna() & df["gender"].notna()].copy()
    dropped = len(df) - len(usable)
    print(f"  {cls}: {len(usable)} participants have usable age+gender "
          f"({dropped} dropped for missing metadata — expected, not an error)")
    return usable


def find_images_per_participant(cls):
    mids_dir = DATA / cls / "MIDS"
    counts = {}
    for sub_dir in mids_dir.glob("sub-*"):
        if not sub_dir.is_dir():
            continue
        pngs = list(sub_dir.rglob("*.png"))
        counts[sub_dir.name] = len(pngs)
    total_images = sum(counts.values())
    participants_with_images = sum(1 for c in counts.values() if c > 0)
    print(f"  {cls}: {total_images} total PNGs across {participants_with_images} "
          f"participants with at least 1 image (out of {len(counts)} participant folders)")
    return counts


def build_class_table(cls, label):
    participants_df = load_participants(cls)
    if participants_df is None:
        return None

    usable_df = filter_usable_metadata(participants_df, cls)
    image_counts = find_images_per_participant(cls)

    rows = []
    n_estimated = 0
    mids_dir = DATA / cls / "MIDS"
    for _, prow in usable_df.iterrows():
        pid = prow["participant"]
        n_images = image_counts.get(pid, 0)
        if n_images == 0:
            continue  # participant has usable metadata but no released images — skip
        age_value, is_estimated = resolve_age(prow["age"])
        if age_value is None:
            continue  # age string had no parseable number at all — treat like missing metadata
        if is_estimated:
            n_estimated += 1
        sub_dir = mids_dir / pid
        for png_path in sub_dir.rglob("*.png"):
            rows.append({
                "image_path": str(png_path),
                "source": "cov_caldas",
                "patient_id": f"covcaldas_{pid}",
                "age": age_value,
                "age_is_estimated": is_estimated,
                "sex": prow["gender"],
                "stage1_label": label,
            })
    df = pd.DataFrame(rows)
    if len(df):
        n_bad = df["age"].isna().sum() + df["sex"].isna().sum()
        assert n_bad == 0, f"{n_bad} rows in the final {cls} table have missing age/sex — filter did not work as expected!"
        print(f"  {cls}: confirmed zero missing age/sex in final table (not assumed)")
        print(f"  {cls}: {n_estimated} participants ({df['age_is_estimated'].sum()} images) "
              f"had multi-value ages resolved to a mean — flagged via age_is_estimated")
    print(f"  {cls}: final usable table has {len(df)} images from "
          f"{df['patient_id'].nunique() if len(df) else 0} patients")
    return df


if __name__ == "__main__":
    print("=== Cov-Caldas Positive ===")
    pos_df = build_class_table("positive", "positive")

    print("\n=== Cov-Caldas Negative ===")
    neg_df = build_class_table("negative", "negative")

    if pos_df is not None and neg_df is not None:
        combined = pd.concat([pos_df, neg_df], ignore_index=True)
        print(f"\n=== Combined ===")
        print(f"Total usable images: {len(combined)}")
        print(f"Total usable patients: {combined['patient_id'].nunique()}")
        print(combined.groupby("stage1_label").size())
        print(f"\nAge-estimated breakdown by class:")
        print(combined.groupby("stage1_label")["age_is_estimated"].sum())

        out_path = Path(__file__).parent.parent / "data_processing"
        out_path.mkdir(exist_ok=True)
        combined.to_csv(out_path / "cov_caldas_verified_table.csv", index=False)
        print(f"\nSaved verified table to {out_path / 'cov_caldas_verified_table.csv'}")
        print("Next: rebuild the frozen patient-level split (build_cov_caldas_split.py) "
              "— the age column format changed (now always numeric + a new "
              "age_is_estimated column) but patient_id set is unchanged, so the "
              "same seed=42 split assignment will be identical.")