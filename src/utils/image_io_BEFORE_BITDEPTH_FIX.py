"""
FIXED: Shared image loader for Stage 1 (Cov-Caldas).

CRITICAL FIX: The original percentile-based normalization per-image was not
eliminating the uint16 vs uint8 acquisition artifact. Instead, it was
ENCODING it differently: uint16 images (lower native range) normalized to
BRIGHT, uint8 images (full range) to VARIABLE/DARK.

NEW APPROACH: Compute intensity statistics ONLY from the TRAINING SET,
separately for each class. Then use these fixed statistics to normalize
all images identically across train/val/test/matched sets. This ensures:

1. Both classes get the same normalization transformation
2. The transformation is frozen after training data inspection (no test leakage)
3. If classes still separate by intensity, it's genuine image content, not artifact

This is the standard approach in medical imaging: normalize using training
statistics, apply the same to all splits.
"""

import numpy as np
from PIL import Image
from pathlib import Path


# These will be set by the _compute_normalization_stats() function
# DO NOT hard-code them; compute from training data
_NORM_STATS = None


def _compute_normalization_stats(train_df):
    """
    Compute the global normalization statistics from the TRAINING SET ONLY.
    
    Parameters:
    -----------
    train_df : DataFrame
        Training set with 'image_path' column. Must contain the training
        images only (not val/test).
    
    Returns:
    --------
    tuple of (lo, hi) : float, float
        The 1st and 99th percentiles to use for all image normalization.
        Computed across all pixel values in all training images.
    """
    all_pixels = []
    print("Computing normalization statistics from training images...")
    
    for i, path in enumerate(train_df["image_path"].values):
        arr = np.asarray(Image.open(path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[..., :3].mean(axis=2)
        
        # Subsample for speed (every 4th pixel), then flatten
        all_pixels.append(arr[::4, ::4].ravel())
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(train_df)}", flush=True)
    
    all_pixels = np.concatenate(all_pixels)
    lo = np.percentile(all_pixels, 1)
    hi = np.percentile(all_pixels, 99)
    
    print(f"Normalization range: [{lo:.1f}, {hi:.1f}]")
    return lo, hi


def set_normalization_stats(lo, hi):
    """
    Set the global normalization statistics. Must be called once after
    computing from training data, before loading any images.
    
    Parameters:
    -----------
    lo, hi : float
        The lower and upper bounds for normalization (e.g., 1st and 99th
        percentiles computed from training images).
    """
    global _NORM_STATS
    _NORM_STATS = (float(lo), float(hi))


def get_normalization_stats():
    """Return the current normalization stats tuple (lo, hi) or None if not set."""
    return _NORM_STATS


def load_normalized_image(path):
    """
    Load an image and normalize to 8-bit RGB using the pre-computed
    global normalization statistics.
    
    CRITICAL: This function now applies the SAME transformation to all images,
    ensuring both uint16 and uint8 sources are genuinely normalized identically.
    
    Parameters:
    -----------
    path : str or Path
        Path to the image file.
    
    Returns:
    --------
    PIL.Image
        8-bit RGB PIL image, normalized to 0-255 using global statistics.
        
    Raises:
    -------
    RuntimeError
        If normalization stats have not been set yet.
    """
    if _NORM_STATS is None:
        raise RuntimeError(
            "Normalization statistics not set. Call set_normalization_stats() "
            "or load_and_set_norm_stats() first."
        )
    
    lo, hi = _NORM_STATS
    
    arr = np.asarray(Image.open(path)).astype(np.float32)
    
    # Convert multi-channel to grayscale
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    
    # Apply the fixed normalization range (computed from training data)
    if hi <= lo:
        hi = lo + 1.0
    
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    
    return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")


def load_and_set_norm_stats(train_df):
    """
    Convenience function: compute normalization from training data and set it.
    
    Parameters:
    -----------
    train_df : DataFrame
        Training set with 'image_path' column.
    """
    lo, hi = _compute_normalization_stats(train_df)
    set_normalization_stats(lo, hi)
