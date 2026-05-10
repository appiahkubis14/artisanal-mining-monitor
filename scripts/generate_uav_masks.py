"""
generate_uav_masks.py  (Script 4.5)
=====================================
Automatically generate binary mining masks from a UAV orthomosaic
without manual digitising or GPS field points.

Pipeline
--------
1. Load preprocessed UAV orthomosaic (RGB or multispectral GeoTIFF)
2. Bare-soil detection   — RGB colour-space rules + NDVI if NIR present
3. Texture features      — GLCM entropy on 5×5 sliding window
4. Edge density          — Canny edges summed over 50×50 blocks
5. Rule-based scoring    — weighted sum; threshold at 0.6 → binary mask
6. Morphological cleanup — opening (3×3) + closing (5×5) + small-object removal
7. Resample to 10 m      — majority rule for Sentinel-2 alignment
8. Save outputs
   - data/masks/uav_mining_mask_fullres.tif   (UAV native resolution)
   - data/masks/uav_mining_mask_10m.tif       (10 m satellite resolution)
   - data/masks/mining_mask_10m.npy           (numpy array for U-Net)

All thresholds are read from config.yaml → uav_annotation section.

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from scipy.ndimage import generic_filter, label

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Step 2 — Bare-soil detection
# ---------------------------------------------------------------------------

def detect_bare_soil(
    img: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Return a binary bare-soil mask from RGB (and optional NIR) bands.

    Strategy A (NDVI, if NIR available):
        bare = NDVI < 0.2
    Strategy B (RGB colour rules):
        bare = Red > thresh_r  AND  Green > thresh_g
               AND  red/green ratio in [ratio_min, ratio_max]

    Both strategies are OR-combined when NIR is available.

    Parameters
    ----------
    img : float32 array [B, H, W], values normalised 0–1
    cfg : uav_annotation config section

    Returns
    -------
    uint8 binary array [H, W], 1 = bare soil
    """
    n_bands = img.shape[0]

    # --- RGB rules (always available) ---
    red   = img[0]
    green = img[1] if n_bands >= 2 else img[0]
    blue  = img[2] if n_bands >= 3 else img[0]

    r_thresh = cfg.get("bare_soil_red_threshold",   100) / 255.0
    g_thresh = cfg.get("bare_soil_green_threshold",  80) / 255.0
    ratio_lo = cfg.get("bare_soil_ratio_min", 0.8)
    ratio_hi = cfg.get("bare_soil_ratio_max", 1.5)

    ratio = np.where(green > 1e-4, red / (green + 1e-8), 0.0)
    rgb_bare = (
        (red   > r_thresh) &
        (green > g_thresh) &
        (ratio >= ratio_lo) &
        (ratio <= ratio_hi)
    ).astype(np.uint8)

    # --- HSV confirmation (refines the RGB rule) ---
    rgb_u8 = (img[:3].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] * 2          # OpenCV hue is 0–180, scale to 0–360
    sat = hsv[:, :, 1] / 255.0
    val = hsv[:, :, 2] / 255.0
    # Brown/yellow hue (20–60°), medium value, low-ish saturation
    hsv_bare = (
        (hue >= 20) & (hue <= 60) &
        (sat < 0.55) &
        (val > 0.25) & (val < 0.90)
    ).astype(np.uint8)

    bare = ((rgb_bare == 1) | (hsv_bare == 1)).astype(np.uint8)

    # --- NDVI (if NIR channel present) ---
    if n_bands >= 4:
        nir  = img[3]
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi_bare = (ndvi < 0.20).astype(np.uint8)
        # OR with spectral bare-soil
        bare = ((bare == 1) | (ndvi_bare == 1)).astype(np.uint8)
        log.debug("NDVI bare-soil mask incorporated.")

    log.info(
        f"Bare-soil: {bare.sum()} px / {bare.size} "
        f"({100*bare.mean():.2f}%)"
    )
    return bare


# ---------------------------------------------------------------------------
# Step 3 — Texture (GLCM entropy)
# ---------------------------------------------------------------------------

def compute_entropy_map(gray: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute per-pixel Shannon entropy using a sliding window.

    Approximated via scipy generic_filter with a 16-bin histogram.

    Parameters
    ----------
    gray   : float32 2-D array [H, W], values in [0, 1]
    window : sliding window radius (window × window kernel)

    Returns
    -------
    Entropy array [H, W], dtype float32
    """
    bins = 16

    def _entropy(patch):
        # patch is flat — reshape implicit
        counts, _ = np.histogram(patch, bins=bins, range=(0.0, 1.0))
        probs = counts / (counts.sum() + 1e-12)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    log.debug(f"Computing entropy map (window={window})…")
    entropy = generic_filter(
        gray.astype(np.float32),
        function=_entropy,
        size=window,
        mode="reflect",
    ).astype(np.float32)
    return entropy


# ---------------------------------------------------------------------------
# Step 4 — Edge density
# ---------------------------------------------------------------------------

def compute_edge_density(gray: np.ndarray, block: int = 50) -> np.ndarray:
    """Compute Canny edge density at block-level resolution.

    Result is upsampled back to the original image size.

    Parameters
    ----------
    gray  : float32 [H, W], values 0–1
    block : block side length in pixels

    Returns
    -------
    float32 array [H, W], values in [0, 1]
    """
    gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
    edges = cv2.Canny(gray_u8, threshold1=50, threshold2=150)

    H, W = gray.shape
    density = np.zeros_like(gray, dtype=np.float32)

    for r in range(0, H, block):
        for c in range(0, W, block):
            r_end = min(r + block, H)
            c_end = min(c + block, W)
            block_edges = edges[r:r_end, c:c_end]
            d = block_edges.mean() / 255.0
            density[r:r_end, c:c_end] = d

    return density


# ---------------------------------------------------------------------------
# Step 5 — Rule-based scoring & thresholding
# ---------------------------------------------------------------------------

def score_and_threshold(
    bare_soil: np.ndarray,
    entropy: np.ndarray,
    edge_density: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Combine three signals into a mining probability score, then threshold.

    score = 0.4 * bare_soil + 0.3 * (entropy > thresh) + 0.3 * (edge > thresh)

    Parameters
    ----------
    bare_soil    : uint8 binary [H, W]
    entropy      : float32 [H, W]
    edge_density : float32 [H, W]
    cfg          : uav_annotation config

    Returns
    -------
    uint8 binary mask [H, W], 1 = mining
    """
    entropy_thresh    = cfg.get("entropy_threshold",      2.0)
    edge_thresh       = cfg.get("edge_density_threshold", 0.3)
    score_thresh      = cfg.get("mining_score_threshold", 0.6)

    high_entropy  = (entropy      > entropy_thresh).astype(np.float32)
    high_edges    = (edge_density > edge_thresh   ).astype(np.float32)

    score = (
        0.4 * bare_soil.astype(np.float32) +
        0.3 * high_entropy +
        0.3 * high_edges
    )

    mining_mask = (score >= score_thresh).astype(np.uint8)
    log.info(
        f"Score threshold={score_thresh}: "
        f"{mining_mask.sum()} mining px ({100*mining_mask.mean():.2f}%)"
    )
    return mining_mask


# ---------------------------------------------------------------------------
# Step 6 — Morphological cleaning
# ---------------------------------------------------------------------------

def morphological_clean(
    mask: np.ndarray,
    min_area_px: int = 50,
) -> np.ndarray:
    """Remove noise and fill holes in binary mining mask.

    1. Opening  (3×3) — removes isolated speckle
    2. Closing  (5×5) — fills small holes
    3. Remove connected components smaller than min_area_px

    Parameters
    ----------
    mask        : uint8 binary [H, W]
    min_area_px : remove components smaller than this (pixels)

    Returns
    -------
    Cleaned uint8 binary mask [H, W]
    """
    k3 = np.ones((3, 3), dtype=np.uint8)
    k5 = np.ones((5, 5), dtype=np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

    # Remove small objects
    labeled, n_components = label(mask)
    for comp_id in range(1, n_components + 1):
        comp_mask = labeled == comp_id
        if comp_mask.sum() < min_area_px:
            mask[comp_mask] = 0

    log.info(
        f"After cleanup: {mask.sum()} mining px "
        f"({100*mask.mean():.2f}%), "
        f"{n_components} components found."
    )
    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Step 7 — Resample to 10 m (majority rule)
# ---------------------------------------------------------------------------

def resample_to_10m_majority(
    mask_fullres: np.ndarray,
    src_transform: Affine,
    target_pixel_m: float = 10.0,
    threshold: float = 0.30,
) -> tuple[np.ndarray, Affine]:
    """Resample binary mask from UAV to satellite resolution using majority vote.

    A 10 m cell is labelled mining if >threshold fraction of its constituent
    UAV pixels are mining (default 30%).

    Parameters
    ----------
    mask_fullres   : uint8 binary [H, W] at UAV native resolution
    src_transform  : affine transform of the full-res mask
    target_pixel_m : target pixel size in metres (default 10)
    threshold      : fraction of mining pixels to classify 10 m cell as mining

    Returns
    -------
    (mask_10m, transform_10m)
    """
    src_pixel_m = abs(src_transform.a)   # GSD in metres
    scale_factor = target_pixel_m / src_pixel_m  # e.g. 10 / 0.05 = 200

    H, W = mask_fullres.shape
    H_out = max(1, int(H / scale_factor))
    W_out = max(1, int(W / scale_factor))

    # Resize with linear interpolation to get fractional values, then threshold
    avg = cv2.resize(
        mask_fullres.astype(np.float32),
        (W_out, H_out),
        interpolation=cv2.INTER_LINEAR,
    )
    mask_10m = (avg >= threshold).astype(np.uint8)

    # Build new affine transform
    dx = src_transform.a * scale_factor
    dy = src_transform.e * scale_factor
    transform_10m = Affine(
        dx, src_transform.b, src_transform.c,
        src_transform.d, dy, src_transform.f,
    )

    log.info(
        f"Resampled {H}×{W} (GSD={src_pixel_m:.3f}m) → "
        f"{H_out}×{W_out} (GSD={target_pixel_m}m). "
        f"Mining cells: {mask_10m.sum()} ({100*mask_10m.mean():.2f}%)"
    )
    return mask_10m, transform_10m


# ---------------------------------------------------------------------------
# Step 8 — Save outputs
# ---------------------------------------------------------------------------

def _save_geotiff(
    array: np.ndarray,
    transform: Affine,
    crs,
    out_path: Path,
    dtype: str = "uint8",
) -> None:
    """Save a single-band array as a GeoTIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":    "GTiff",
        "dtype":     dtype,
        "width":     array.shape[1],
        "height":    array.shape[0],
        "count":     1,
        "crs":       crs,
        "transform": transform,
        "compress":  "lzw",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array[np.newaxis, ...])
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@timer
def generate_uav_masks(config: dict) -> dict:
    """Auto-generate mining masks from the preprocessed UAV orthomosaic.

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    Dict with paths to saved masks and basic statistics
    """
    ann_cfg   = config.get("uav_annotation", {})
    paths     = config["paths"]
    masks_dir = Path(paths["masks"])
    masks_dir.mkdir(parents=True, exist_ok=True)

    fullres_tif  = masks_dir / "uav_mining_mask_fullres.tif"
    mask_10m_tif = masks_dir / "uav_mining_mask_10m.tif"
    mask_10m_npy = masks_dir / "mining_mask_10m.npy"

    # ------------------------------------------------------------------ #
    # Checkpoint: reload from disk if all three outputs already exist
    # ------------------------------------------------------------------ #
    if fullres_tif.exists() and mask_10m_tif.exists() and mask_10m_npy.exists():
        log.info(
            f"[RESUME] UAV masks already exist in {masks_dir}. "
            "Skipping mask generation."
        )
        import rasterio as _rio
        with _rio.open(mask_10m_tif) as src:
            mask_10m = src.read(1).astype(np.uint8)
        with _rio.open(fullres_tif) as src:
            mask_fr  = src.read(1).astype(np.uint8)
        return {
            "uav_tif":             "cached",
            "gsd_m":               0.0,
            "fullres_shape":       list(mask_fr.shape),
            "mask_10m_shape":      list(mask_10m.shape),
            "fullres_mining_px":   int(mask_fr.sum()),
            "fullres_mining_pct":  round(100 * mask_fr.mean(), 3),
            "mask_10m_mining_px":  int(mask_10m.sum()),
            "mask_10m_mining_pct": round(100 * mask_10m.mean(), 3),
            "fullres_tif":         str(fullres_tif),
            "mask_10m_tif":        str(mask_10m_tif),
            "mask_10m_npy":        str(mask_10m_npy),
        }

    # ------------------------------------------------------------------ #
    # Step 1 — Locate UAV orthomosaic
    # ------------------------------------------------------------------ #
    uav_processed_dir = Path(paths["processed"]) / "uav"
    candidates = list(uav_processed_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"No UAV GeoTIFF found in {uav_processed_dir}. "
            "Run preprocessor.py (step: preprocess) first."
        )

    # Prefer the high-res file; fall back to any .tif
    uav_tif = next(
        (p for p in candidates if "highres" in p.name or "ortho" in p.name.lower()),
        candidates[0],
    )
    log.info(f"Loading UAV orthomosaic: {uav_tif}")

    with rasterio.open(uav_tif) as src:
        img_raw  = src.read().astype(np.float32)   # [B, H, W]
        crs      = src.crs
        transform_fullres = src.transform
        gsd_m    = abs(src.transform.a)

    n_bands, H, W = img_raw.shape
    log.info(f"UAV image: {n_bands} bands, {H}×{W} px, GSD={gsd_m:.4f} m")

    # Normalise to [0, 1] (handle already-normalised and raw uint8/uint16)
    if img_raw.max() > 1.0:
        p_high = np.percentile(img_raw, 98)
        p_low  = np.percentile(img_raw,  2)
        img = np.clip((img_raw - p_low) / (p_high - p_low + 1e-8), 0, 1)
    else:
        img = img_raw.clip(0, 1)

    # Grayscale for texture / edges
    if n_bands >= 3:
        gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
    else:
        gray = img[0]

    # ------------------------------------------------------------------ #
    # Steps 2–5 — Feature maps → score → threshold
    # ------------------------------------------------------------------ #
    log.info("Step 2: Bare-soil detection …")
    bare_soil = detect_bare_soil(img, ann_cfg)

    log.info("Step 3: Entropy texture …")
    entropy = compute_entropy_map(gray, window=5)

    log.info("Step 4: Edge density …")
    edge_density = compute_edge_density(gray, block=50)

    log.info("Step 5: Scoring and thresholding …")
    mining_mask_raw = score_and_threshold(bare_soil, entropy, edge_density, ann_cfg)

    # ------------------------------------------------------------------ #
    # Step 6 — Morphological cleaning
    # ------------------------------------------------------------------ #
    log.info("Step 6: Morphological cleaning …")
    min_px = max(
        1,
        int(ann_cfg.get("min_mining_area_m2", 50) / max(gsd_m ** 2, 1e-6)),
    )
    mining_mask_fullres = morphological_clean(mining_mask_raw, min_area_px=min_px)

    # ------------------------------------------------------------------ #
    # Step 7 — Resample to 10 m
    # ------------------------------------------------------------------ #
    log.info("Step 7: Resampling to 10 m …")
    target_m = float(ann_cfg.get("resample_cell_size_m", 10))
    mining_mask_10m, transform_10m = resample_to_10m_majority(
        mining_mask_fullres,
        transform_fullres,
        target_pixel_m=target_m,
        threshold=0.30,
    )

    # ------------------------------------------------------------------ #
    # Step 8 — Save outputs
    # ------------------------------------------------------------------ #
    fullres_tif  = masks_dir / "uav_mining_mask_fullres.tif"
    mask_10m_tif = masks_dir / "uav_mining_mask_10m.tif"
    mask_10m_npy = masks_dir / "mining_mask_10m.npy"

    log.info("Step 8: Saving outputs …")
    _save_geotiff(mining_mask_fullres, transform_fullres, crs, fullres_tif)
    _save_geotiff(mining_mask_10m,     transform_10m,     crs, mask_10m_tif)
    np.save(mask_10m_npy, mining_mask_10m)
    log.info(f"Saved numpy array: {mask_10m_npy}")

    stats = {
        "uav_tif":            str(uav_tif),
        "gsd_m":              gsd_m,
        "fullres_shape":      list(mining_mask_fullres.shape),
        "mask_10m_shape":     list(mining_mask_10m.shape),
        "fullres_mining_px":  int(mining_mask_fullres.sum()),
        "fullres_mining_pct": round(100 * mining_mask_fullres.mean(), 3),
        "mask_10m_mining_px": int(mining_mask_10m.sum()),
        "mask_10m_mining_pct":round(100 * mining_mask_10m.mean(), 3),
        "fullres_tif":        str(fullres_tif),
        "mask_10m_tif":       str(mask_10m_tif),
        "mask_10m_npy":       str(mask_10m_npy),
    }

    log.info(
        f"\n{'='*55}\n"
        f"UAV MASK GENERATION COMPLETE\n"
        f"  Full-res mining : {stats['fullres_mining_pct']:.2f}%\n"
        f"  10 m mining     : {stats['mask_10m_mining_pct']:.2f}%\n"
        f"  Full-res TIF    : {fullres_tif}\n"
        f"  10 m TIF        : {mask_10m_tif}\n"
        f"  10 m NPY        : {mask_10m_npy}\n"
        f"{'='*55}"
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-generate mining masks from UAV orthomosaic."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_uav_masks(cfg)