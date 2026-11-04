"""
Shared image loader for Cov-Caldas / Stage 1.

BIT-DEPTH HARMONIZATION:
- uint8 images are converted from [0, 255] -> [0, 1]
- uint16 images are converted from 14-bit [0, 16383] -> [0, 1]

After native bit-depth harmonization, normalization statistics are computed
ONLY from the training set and the same frozen transformation is applied
to train / validation / test.

Raw images are never modified.
"""

import numpy as np
from PIL import Image


_NORM_STATS = None


def _to_common_scale(path):
    """
    Load an image and convert its native intensity representation to [0, 1].

    uint8:
        native range = 0..255

    uint16:
        Cov-Caldas images use a 14-bit intensity range stored in uint16:
        native range = 0..16383
    """
    with Image.open(path) as img:
        arr = np.asarray(img)

    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)

    original_dtype = arr.dtype
    arr = arr.astype(np.float32)

    if original_dtype == np.uint8:
        arr /= 255.0

    elif original_dtype == np.uint16:
        arr /= 16383.0

    else:
        raise ValueError(
            f"Unsupported image dtype {original_dtype} for {path}. "
            "Expected uint8 or uint16."
        )

    return np.clip(arr, 0.0, 1.0)


def _compute_normalization_stats(train_df):
    """
    Compute normalization statistics from TRAINING IMAGES ONLY.

    Statistics are calculated AFTER native bit-depth harmonization.

    Returns:
        (lo, hi): 1st and 99th percentiles in common [0,1] scale.
    """
    all_pixels = []

    print(
        "Computing normalization statistics from training images "
        "after bit-depth harmonization..."
    )

    for i, path in enumerate(train_df["image_path"].values, 1):

        arr = _to_common_scale(path)

        # Subsample for speed.
        pixels = arr[::4, ::4].ravel()
        all_pixels.append(pixels)

        if i % 500 == 0:
            print(f"  {i}/{len(train_df)}", flush=True)

    all_pixels = np.concatenate(all_pixels)

    lo = float(np.percentile(all_pixels, 1))
    hi = float(np.percentile(all_pixels, 99))

    print(f"Harmonized normalization range: [{lo:.6f}, {hi:.6f}]")

    return lo, hi


def set_normalization_stats(lo, hi):
    """
    Freeze the normalization statistics.
    """
    global _NORM_STATS

    _NORM_STATS = (float(lo), float(hi))


def get_normalization_stats():
    """
    Return frozen normalization statistics.
    """
    return _NORM_STATS


def load_normalized_image(path):
    """
    Load an image using:

        native bit-depth harmonization
                    ?
        frozen training-set normalization
                    ?
        8-bit RGB PIL image
    """

    if _NORM_STATS is None:
        raise RuntimeError(
            "Normalization statistics not set. "
            "Call load_and_set_norm_stats(train_df) first."
        )

    lo, hi = _NORM_STATS

    arr = _to_common_scale(path)

    if hi <= lo:
        hi = lo + 1e-6

    arr = np.clip(
        (arr - lo) / (hi - lo),
        0.0,
        1.0
    )

    arr = (arr * 255.0).astype(np.uint8)

    return Image.fromarray(arr, mode="L").convert("RGB")


def load_and_set_norm_stats(train_df):
    """
    Compute and freeze normalization statistics from TRAINING DATA ONLY.
    """
    lo, hi = _compute_normalization_stats(train_df)
    set_normalization_stats(lo, hi)
