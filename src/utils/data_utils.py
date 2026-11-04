"""
Shared utilities for Phase 4 (ML baselines) and Phase 5 (CNN), and later
Phase 7 (fusion). Centralizes:

  - loading the frozen split CSVs with the exact schema build_labels_and_split.py
    produces: image_path, source, patient_id, study_uid, age, sex,
              stage1_label, appearance, severity, split
              (+ testing_result, specimen_source, once enrich_stage2_metadata.py
               has been run — see src/data_processing/enrich_stage2_metadata.py)
  - the age-outlier check/filter flagged as an open EDA item (RICORD min age 0,
    ChestX-ray14 min age 2) — resolved here as an explicit, logged, documented
    step, not silently dropped
  - building Stage 1 (shared age/sex, both datasets) and Stage 2 (RICORD-only,
    severity-graded subset, full clinical fields) feature tables
  - per-patient inverse-frequency sample weights (the WeightedRandomSampler
    decision in the Decisions Log), reusable for both the CNN loader (Phase 5)
    and fusion loader (Phase 7)
  - image loading that transparently handles RICORD's DICOM files and
    ChestX-ray14's PNGs

If your actual CSV headers differ from what's listed below, fix the COLUMNS
mapping in one place here — nothing downstream needs to change.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema — matches build_labels_and_split.py exactly. Adjust here only if
# your real files differ.
# ---------------------------------------------------------------------------
COLUMNS = {
    "image_path": "image_path",
    "source": "source",             # 'ricord' | 'chestxray14'
    "patient_id": "patient_id",     # prefixed, e.g. 'ricord_419639-000002' / 'cxr14_00000001'
    "study_uid": "study_uid",
    "age": "age",
    "sex": "sex",
    "stage1_label": "stage1_label",  # 'positive' | 'negative'
    "appearance": "appearance",
    "severity": "severity",
    "split": "split",
    "testing_result": "testing_result",     # added by enrich_stage2_metadata.py
    "specimen_source": "specimen_source",   # added by enrich_stage2_metadata.py
}

RICORD = "ricord"
CHESTXRAY14 = "chestxray14"

# Per Decisions Log: RICORD inclusion criteria required age >= 18, so age 0
# is a de-identification artifact, not a real patient. ChestX-ray14 has no
# such documented adult-only criterion, but age 2 on an otherwise adult-heavy
# dataset (mean 44.4) is implausible enough to flag the same way. We use a
# generous, dataset-aware floor rather than one global cutoff.
AGE_FLOOR = {RICORD: 18, CHESTXRAY14: 5}
AGE_CEILING = 100  # ages above this are also almost certainly artifacts


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_repo_root(start: Path = None) -> Path:
    """Walk up from `start` (default: this file) until a folder containing
    both 'data' and 'src' is found. Lets scripts be run from anywhere."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here] + list(here.parents):
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    # Fall back to the conventional 3-levels-up-from-src/utils/ location.
    return Path(__file__).parent.parent.parent


def load_split(split_name: str, processed_dir: Path = None) -> pd.DataFrame:
    """Load one of split_train.csv / split_val.csv / split_test.csv."""
    assert split_name in ("train", "val", "test"), split_name
    root = find_repo_root()
    processed_dir = processed_dir or (root / "data" / "processed")
    path = processed_dir / f"split_{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run build_labels_and_split.py (Phase 2) first, "
            f"and enrich_stage2_metadata.py before any Stage 2 work."
        )
    df = pd.read_csv(path)

    required = [COLUMNS[k] for k in ("image_path", "source", "patient_id", "age", "sex", "stage1_label", "split")]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path} is missing expected column(s) {missing}. "
            f"Actual columns: {list(df.columns)}. Update COLUMNS in data_utils.py if headers differ."
        )
    return df


def load_all_splits(processed_dir: Path = None) -> dict:
    return {s: load_split(s, processed_dir) for s in ("train", "val", "test")}


# ---------------------------------------------------------------------------
# Age-outlier handling (EDA open item #1)
# ---------------------------------------------------------------------------
def clean_ages(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Flag implausible ages per-dataset, impute with the per-source median
    of *plausible* ages, and keep an `age_imputed` flag column so the
    imputation is visible/auditable rather than silently baked in.

    This does not drop rows or change split membership — only the `age`
    value and a new boolean column are affected.
    """
    df = df.copy()
    age_col, source_col = COLUMNS["age"], COLUMNS["source"]
    df["age_imputed"] = False

    for source, floor in AGE_FLOOR.items():
        mask_source = df[source_col] == source
        implausible = mask_source & (
            (df[age_col] < floor) | (df[age_col] > AGE_CEILING) | df[age_col].isna()
        )
        n_implausible = int(implausible.sum())
        if n_implausible == 0:
            continue

        plausible_median = df.loc[mask_source & ~implausible, age_col].median()
        if verbose:
            print(
                f"[clean_ages] {source}: {n_implausible} row(s) with age < {floor}, "
                f"> {AGE_CEILING}, or missing -> imputed with plausible median "
                f"({plausible_median:.1f})."
            )
        df.loc[implausible, "age_imputed"] = True
        df.loc[implausible, age_col] = plausible_median

    return df


# ---------------------------------------------------------------------------
# Stage 1 / Stage 2 feature builders
# ---------------------------------------------------------------------------
def encode_sex(series: pd.Series) -> pd.Series:
    """Robust to whatever casing/values are actually in the sheet (M/F,
    Male/Female, etc.) — anything not recognizably male maps to 0, female to 1,
    unrecognized/missing gets its own flag rather than being silently guessed."""
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {"M": 0, "MALE": 0, "F": 1, "FEMALE": 1}
    encoded = normalized.map(mapping)
    n_unmapped = int(encoded.isna().sum())
    if n_unmapped:
        print(f"[encode_sex] {n_unmapped} value(s) not recognized as M/F "
              f"(e.g. {normalized[encoded.isna()].unique()[:5]}) -> encoded as 0.5 (unknown).")
    return encoded.fillna(0.5)


def get_stage1_features(df: pd.DataFrame):
    """Shared minimal feature set per the locked architecture: Age, Sex only
    (the only fields common to both source datasets). Returns (X, y, groups).
    y is 1 for 'positive', 0 for 'negative'. groups = patient_id, for any
    group-aware CV the modeling script chooses to do downstream."""
    df = clean_ages(df, verbose=True)
    X = pd.DataFrame({
        "age": df[COLUMNS["age"]].astype(float),
        "sex": encode_sex(df[COLUMNS["sex"]]),
    })
    y = (df[COLUMNS["stage1_label"]].astype(str).str.lower() == "positive").astype(int)
    groups = df[COLUMNS["patient_id"]]
    return X, y, groups


def get_stage2_subset(df: pd.DataFrame) -> pd.DataFrame:
    """RICORD-only, severity-graded rows (excludes the 246/1257 with no
    severity grade, i.e. 'Negative for Pneumonia' / 'Invalid Study' appearance —
    per the Decisions Log, severity only applies when pneumonia findings are
    present, so those rows have no Stage 2 target)."""
    ricord = df[df[COLUMNS["source"]] == RICORD].copy()
    graded = ricord[ricord[COLUMNS["severity"]].notna()].copy()
    return graded


def get_stage2_features(df: pd.DataFrame):
    """Full RICORD clinical fields: Age, Sex, Testing Result, Specimen Source.
    Requires enrich_stage2_metadata.py to have been run (testing_result /
    specimen_source columns present) — raises a clear error if not, rather
    than silently training on Age+Sex only and contradicting the architecture
    spec in 01_RESEARCH_dataset_literature.md."""
    subset = get_stage2_subset(df)
    subset = clean_ages(subset, verbose=True)

    for col_key in ("testing_result", "specimen_source"):
        col = COLUMNS[col_key]
        if col not in subset.columns:
            raise KeyError(
                f"'{col}' not found in the split table. Stage 2 requires the full "
                f"RICORD clinical fields (Age, Sex, Testing Result, Specimen Source) "
                f"per the locked architecture. Run "
                f"src/data_processing/enrich_stage2_metadata.py first."
            )

    testing_result = pd.get_dummies(subset[COLUMNS["testing_result"]].astype(str),
                                     prefix="testing_result")
    specimen_source = pd.get_dummies(subset[COLUMNS["specimen_source"]].astype(str),
                                      prefix="specimen_source")

    X = pd.concat([
        pd.DataFrame({
            "age": subset[COLUMNS["age"]].astype(float),
            "sex": encode_sex(subset[COLUMNS["sex"]]),
        }, index=subset.index),
        testing_result,
        specimen_source,
    ], axis=1)

    y = subset[COLUMNS["severity"]]
    groups = subset[COLUMNS["patient_id"]]
    return X, y, groups, subset


# ---------------------------------------------------------------------------
# Per-patient inverse-frequency sample weighting
# (Decisions Log: 25 RICORD patients have far more images than the ~1.73
#  average, max 34 — decided to NOT drop images, use inverse-frequency
#  per-patient weighting instead.)
# ---------------------------------------------------------------------------
def per_patient_inverse_freq_weights(df: pd.DataFrame) -> np.ndarray:
    """One weight per row: 1 / (number of images that row's patient has),
    so each patient contributes equal *total* weight to an epoch regardless
    of how many images they have. Suitable for sklearn's `sample_weight`
    and as the basis for a torch WeightedRandomSampler."""
    counts = df[COLUMNS["patient_id"]].map(df[COLUMNS["patient_id"]].value_counts())
    return (1.0 / counts).to_numpy()


# ---------------------------------------------------------------------------
# Image loading (DICOM for RICORD, PNG for ChestX-ray14)
# ---------------------------------------------------------------------------
def load_image_as_array(image_path, source: str) -> np.ndarray:
    """Returns a float32 HxW array scaled to [0, 255], regardless of source
    format. Caller is responsible for resizing/normalizing/channel-stacking
    for whatever model consumes it (kept separate so this module has no
    hard torch/torchvision dependency)."""
    path = Path(image_path)
    if source == RICORD:
        import pydicom
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        # MONOCHROME1 means pixel intensity is inverted relative to the usual
        # X-ray convention (higher value = darker) — flip so all images share
        # one convention before display/model input.
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)
        return arr
    elif source == CHESTXRAY14:
        from PIL import Image
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)
    else:
        raise ValueError(f"Unknown source '{source}' — expected '{RICORD}' or '{CHESTXRAY14}'.")


if __name__ == "__main__":
    # Quick self-check when run directly: load whichever splits exist and
    # print the Stage 1 / Stage 2 feature-table shapes, so a team member can
    # sanity-check the pipeline before kicking off a real training run.
    for split_name in ("train", "val", "test"):
        try:
            df = load_split(split_name)
        except FileNotFoundError as e:
            print(e)
            continue
        X1, y1, g1 = get_stage1_features(df)
        print(f"[{split_name}] Stage 1: X={X1.shape}, positives={y1.sum()}, negatives={(1 - y1).sum()}")
        try:
            X2, y2, g2, _ = get_stage2_features(df)
            print(f"[{split_name}] Stage 2: X={X2.shape}, classes={sorted(y2.unique())}")
        except KeyError as e:
            print(f"[{split_name}] Stage 2 skipped: {e}")