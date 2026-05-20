"""
feature_engineering.py - Domain-Specific Feature Extraction
=============================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Computes spectral indices (NDVI, NDWI, MNDWI, NDBI, SAVI, Brightness),
GLCM texture features, temporal change features, SAR polarimetric features,
and UAV-derived features. All features are appended to the input tensor.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import generic_filter

from scripts.utils import get_logger, numpy_to_geotiff

logger = get_logger(__name__)

# Band indices within preprocessed Sentinel-2 stack
# Corresponds to: [B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR1), B12(SWIR2)]
IDX = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir1": 4, "swir2": 5}


# =============================================================================
# Spectral Indices
# =============================================================================

def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Vegetation Index.

    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1]. Dense vegetation ≈ 0.6–0.9. Mining bare soil ≈ 0.0–0.2.

    Args:
        nir: NIR band array [..., H, W].
        red: Red band array [..., H, W].

    Returns:
        NDVI array same shape as input.
    """
    return _safe_ratio(nir - red, nir + red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Water Index (McFeeters 1996).

    NDWI = (Green - NIR) / (Green + NIR)
    Highlights open water. Mining ponds ≈ 0.3–0.8.
    """
    return _safe_ratio(green - nir, green + nir)


def compute_mndwi(green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """
    Modified NDWI (Xu 2006) — better for turbid mining water.

    MNDWI = (Green - SWIR1) / (Green + SWIR1)
    Mining settling ponds with turbid water have high MNDWI.
    """
    return _safe_ratio(green - swir1, green + swir1)


def compute_ndbi(swir1: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Normalised Difference Built-up Index.

    NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    Built-up / disturbed soil has higher NDBI.
    """
    return _safe_ratio(swir1 - nir, swir1 + nir)


def compute_savi(
    nir: np.ndarray,
    red: np.ndarray,
    L: float = 0.5
) -> np.ndarray:
    """
    Soil-Adjusted Vegetation Index (Huete 1988).

    SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
    L=0.5 balances vegetation/soil correction.
    """
    return (nir - red) / (nir + red + L + 1e-8) * (1 + L)


def compute_brightness(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray
) -> np.ndarray:
    """
    Tasseled Cap Brightness (simplified): mean of RGB bands.

    Mining clearings are significantly brighter than surrounding forest.
    """
    return (blue + green + red) / 3.0


def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    """
    Normalised Burn Ratio.

    NBR = (NIR - SWIR2) / (NIR + SWIR2)
    Negative dNBR indicates disturbance (forest removal for mining).
    """
    return _safe_ratio(nir - swir2, nir + swir2)


def compute_all_spectral_indices(
    s2_composite: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute all spectral indices from a Sentinel-2 composite.

    Args:
        s2_composite: Array [6, H, W] with bands in order:
                      [Blue, Green, Red, NIR, SWIR1, SWIR2].

    Returns:
        Dict mapping index name to [H, W] array.
    """
    b = {name: s2_composite[idx] for name, idx in IDX.items()}

    indices = {
        "ndvi":       compute_ndvi(b["nir"], b["red"]),
        "ndwi":       compute_ndwi(b["green"], b["nir"]),
        "mndwi":      compute_mndwi(b["green"], b["swir1"]),
        "ndbi":       compute_ndbi(b["swir1"], b["nir"]),
        "savi":       compute_savi(b["nir"], b["red"]),
        "brightness": compute_brightness(b["blue"], b["green"], b["red"]),
        "nbr":        compute_nbr(b["nir"], b["swir2"]),
    }

    logger.debug(f"Spectral indices computed: {list(indices.keys())}")
    return indices


# =============================================================================
# GLCM Texture Features
# =============================================================================

def compute_glcm_features(
    image: np.ndarray,
    window_size: int = 5
) -> Dict[str, np.ndarray]:
    """
    Compute GLCM-based texture features using a sliding window.

    Features: contrast, correlation, entropy, homogeneity.
    Applied to the NDVI band (captures forest vs. clearing texture).

    Note: Full GLCM computation is expensive for large images.
    This implementation uses a sliding window approximation with
    scipy's generic_filter for efficiency.

    Args:
        image: 2D array [H, W] (typically NDVI or brightness).
        window_size: Sliding window size in pixels.

    Returns:
        Dict with texture feature arrays, each shape [H, W].
    """
    logger.debug(f"Computing GLCM texture (window={window_size}×{window_size})...")

    def _local_contrast(values: np.ndarray) -> float:
        """Local contrast: variance of pixel values in window."""
        return float(np.var(values))

    def _local_entropy(values: np.ndarray) -> float:
        """Local entropy: Shannon entropy of quantised values."""
        bins = np.histogram(values, bins=16, range=(0, 1))[0]
        probs = bins / (bins.sum() + 1e-8)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs + 1e-8)))

    def _local_homogeneity(values: np.ndarray) -> float:
        """Homogeneity: inverse of (1 + contrast)."""
        contrast = np.var(values)
        return float(1.0 / (1.0 + contrast))

    def _local_correlation(values: np.ndarray) -> float:
        """Correlation: normalised second moment."""
        mu = np.mean(values)
        sigma = np.std(values)
        if sigma < 1e-8:
            return 1.0
        return float(np.mean((values - mu) ** 2) / (sigma ** 2 + 1e-8))

    # Normalise input to [0, 1]
    img = np.clip(image, 0, 1).astype(np.float64)

    size = (window_size, window_size)
    features = {
        "texture_contrast":    generic_filter(img, _local_contrast, size=size).astype(np.float32),
        "texture_entropy":     generic_filter(img, _local_entropy, size=size).astype(np.float32),
        "texture_homogeneity": generic_filter(img, _local_homogeneity, size=size).astype(np.float32),
        "texture_correlation": generic_filter(img, _local_correlation, size=size).astype(np.float32),
    }

    logger.debug("Texture features computed.")
    return features


# =============================================================================
# Temporal Features
# =============================================================================

def compute_temporal_features(
    ndvi_stack: np.ndarray,
    dates: List[str]
) -> Dict[str, np.ndarray]:
    """
    Compute change-detection features from a temporal NDVI stack.

    Args:
        ndvi_stack: Array [T, H, W] of NDVI values (float32).
        dates: List of date strings ('YYYYMMDD') corresponding to T.

    Returns:
        Dict with temporal feature arrays, each shape [H, W].
    """
    if ndvi_stack.shape[0] < 2:
        logger.warning("Only 1 date; returning zero temporal features.")
        H, W = ndvi_stack.shape[1], ndvi_stack.shape[2]
        return {
            "ndvi_diff":      np.zeros((H, W), dtype=np.float32),
            "ndvi_min":       ndvi_stack[0],
            "ndvi_max":       ndvi_stack[0],
            "ndvi_std":       np.zeros((H, W), dtype=np.float32),
            "ndvi_trend":     np.zeros((H, W), dtype=np.float32),
            "vegetation_loss": np.zeros((H, W), dtype=np.float32),
        }

    # NDVI difference (latest - earliest)
    ndvi_diff = ndvi_stack[-1] - ndvi_stack[0]

    # Temporal statistics
    ndvi_min = np.nanmin(ndvi_stack, axis=0)
    ndvi_max = np.nanmax(ndvi_stack, axis=0)
    ndvi_std = np.nanstd(ndvi_stack, axis=0)

    # Linear trend across time (pixel-wise slope using least squares)
    T = ndvi_stack.shape[0]
    t_values = np.arange(T, dtype=np.float32)
    t_mean = t_values.mean()
    ndvi_mean = np.nanmean(ndvi_stack, axis=0)
    numerator = np.nansum(
        (t_values[:, None, None] - t_mean) *
        (ndvi_stack - ndvi_mean[None, :, :]),
        axis=0
    )
    denominator = np.sum((t_values - t_mean) ** 2) + 1e-8
    ndvi_trend = numerator / denominator

    # Vegetation loss: fraction of time NDVI < 0.3 (bare soil threshold)
    vegetation_loss = (ndvi_stack < 0.30).mean(axis=0).astype(np.float32)

    features = {
        "ndvi_diff":       ndvi_diff.astype(np.float32),
        "ndvi_min":        ndvi_min.astype(np.float32),
        "ndvi_max":        ndvi_max.astype(np.float32),
        "ndvi_std":        ndvi_std.astype(np.float32),
        "ndvi_trend":      ndvi_trend.astype(np.float32),
        "vegetation_loss": vegetation_loss,
    }

    logger.debug("Temporal features computed.")
    return features


# =============================================================================
# SAR Features
# =============================================================================

def compute_sar_features(s1_composite: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute SAR-derived features from Sentinel-1 VV/VH composite.

    Args:
        s1_composite: Array [2, H, W] — VV=0, VH=1 (normalised, dB-scale).

    Returns:
        Dict with SAR feature arrays, each shape [H, W].
    """
    if s1_composite.shape[0] < 2:
        logger.warning("Only 1 SAR band; VV/VH ratio unavailable.")
        vv = s1_composite[0]
        return {"vv": vv, "vvvh_ratio": np.zeros_like(vv), "sar_entropy": np.zeros_like(vv)}

    vv = s1_composite[0]
    vh = s1_composite[1]

    # VV/VH cross-pol ratio — sensitive to surface roughness
    # Mining ponds (smooth) have low cross-pol; bare soil/equipment higher
    vv_linear = 10 ** (vv / 10.0)
    vh_linear = 10 ** (vh / 10.0)
    vvvh_ratio = _safe_ratio(vv_linear, vh_linear + 1e-8)
    vvvh_ratio = np.clip(vvvh_ratio, 0, 10)

    # Polarimetric entropy (simplified): measures disorder
    p_vv = vv_linear / (vv_linear + vh_linear + 1e-8)
    p_vh = 1.0 - p_vv
    sar_entropy = -(
        p_vv * np.log2(p_vv + 1e-8) +
        p_vh * np.log2(p_vh + 1e-8)
    )

    features = {
        "vv":            vv.astype(np.float32),
        "vh":            vh.astype(np.float32),
        "vvvh_ratio":    vvvh_ratio.astype(np.float32),
        "sar_entropy":   sar_entropy.astype(np.float32),
    }

    logger.debug("SAR features computed.")
    return features


# =============================================================================
# UAV Features
# =============================================================================

def compute_uav_features(
    uav_10m: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute features from the UAV orthomosaic resampled to 10m.

    Args:
        uav_10m: Array [bands, H, W] (float32, 0–1).
                 Bands: [Red, Green, Blue] (and NIR if multispectral).

    Returns:
        Dict with UAV feature arrays.
    """
    features = {}
    n_bands = uav_10m.shape[0]

    if n_bands >= 3:
        red, green, blue = uav_10m[0], uav_10m[1], uav_10m[2]
        features["uav_red"] = red
        features["uav_green"] = green
        features["uav_blue"] = blue
        features["uav_brightness"] = (red + green + blue) / 3.0

    # UAV NDVI (requires NIR band, band index 3)
    if n_bands >= 4:
        nir = uav_10m[3]
        features["uav_ndvi"] = compute_ndvi(nir, features.get("uav_red", uav_10m[0]))

    # Edge density (sharp edges indicate disturbed soil / equipment boundaries)
    try:
        import cv2
        rgb_gray = (
            uav_10m[0] * 0.299 + uav_10m[1] * 0.587 + uav_10m[2] * 0.114
            if n_bands >= 3 else uav_10m[0]
        )
        gray_uint8 = (rgb_gray * 255).astype(np.uint8)
        edges = cv2.Canny(gray_uint8, 50, 150)
        features["uav_edge_density"] = (edges > 0).astype(np.float32)
    except Exception:
        pass

    logger.debug(f"UAV features computed: {list(features.keys())}")
    return features


# =============================================================================
# Orchestrator
# =============================================================================

def compute_all_features(
    preprocessed: Dict[str, Any],
    config: Dict[str, Any],
    uav_available: bool = False
) -> Dict[str, Any]:
    """
    Compute all features and stack into a single feature array.

    The output feature stack is suitable for direct input to the U-Net model.

    Checkpoint/resume: if feature_stack.tif already exists in
    config['paths']['features'] the function reloads it from disk and
    skips all computation.

    Args:
        preprocessed: Output from preprocessor.preprocess_all().
        config: Config dict.
        uav_available: Whether UAV data is available.

    Returns:
        Dict with:
            - 'feature_stack': np.ndarray [C, H, W] — all features concatenated
            - 'feature_names': list of feature names (for interpretability)
            - 'meta': rasterio metadata for saving
    """
    paths = config["paths"]
    os.makedirs(paths["features"], exist_ok=True)

    out_path       = os.path.join(paths["features"], "feature_stack.tif")
    feat_names_path = os.path.join(paths["features"], "feature_names.json")

    # --- Checkpoint: reload from disk if already computed ---
    if os.path.exists(out_path) and os.path.exists(feat_names_path):
        logger.info(f"[RESUME] Feature stack already exists → {out_path}. Skipping feature engineering.")
        import rasterio, json as _json
        with rasterio.open(out_path) as src:
            feature_stack = src.read().astype(np.float32)
            meta = dict(src.meta)
        with open(feat_names_path) as f:
            feature_names = _json.load(f)
        logger.info(f"Loaded feature stack: {feature_stack.shape[0]} channels from disk.")
        return {
            "feature_stack": feature_stack,
            "feature_names": feature_names,
            "meta":          meta,
            "output_path":   out_path,
        }

    feature_arrays = []
    feature_names  = []
    meta = None

    total_steps = 5  # S2 bands, S2 indices, S2 texture, S1, Landsat
    step = 0

    def _progress(name):
        nonlocal step
        step += 1
        logger.info(f"  [{step}/{total_steps}] Computing {name}...")

    # ---- Sentinel-2 raw bands + indices ----
    if "sentinel2" in preprocessed:
        s2 = preprocessed["sentinel2"]["composite"]  # [6, H, W]
        meta = preprocessed["sentinel2"]["meta"]

        # Raw bands
        _progress("Sentinel-2 raw bands (6 channels)")
        band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
        for i, name in enumerate(band_names):
            feature_arrays.append(s2[i:i+1])
            feature_names.append(f"s2_{name}")

        # Spectral indices
        _progress("Sentinel-2 spectral indices (NDVI, NDWI, NDBI, BSI, MNDWI, EVI)")
        indices = compute_all_spectral_indices(s2)
        for name, arr in indices.items():
            feature_arrays.append(arr[np.newaxis])
            feature_names.append(name)

        # Texture on NDVI
        _progress("GLCM texture features on NDVI (this may take 1–2 minutes)")
        ndvi = indices["ndvi"]
        texture = compute_glcm_features(ndvi, window_size=5)
        for name, arr in texture.items():
            feature_arrays.append(arr[np.newaxis])
            feature_names.append(name)

        # Temporal features (if multi-date)
        s2_stack = preprocessed["sentinel2"]["data"]  # [T, 6, H, W]
        if s2_stack.shape[0] > 1:
            ndvi_stack = np.array([
                compute_ndvi(s2_stack[t, IDX["nir"]], s2_stack[t, IDX["red"]])
                for t in range(s2_stack.shape[0])
            ])
            dates = preprocessed["sentinel2"]["dates"]
            temporal = compute_temporal_features(ndvi_stack, dates)
            for name, arr in temporal.items():
                feature_arrays.append(arr[np.newaxis])
                feature_names.append(name)

    # ---- Sentinel-1 SAR features ----
    if "sentinel1" in preprocessed:
        _progress("Sentinel-1 SAR features (VV, VH, ratio, texture)")
        s1 = preprocessed["sentinel1"]["composite"]  # [2, H, W]

        # Resample S1 to match S2 spatial dimensions if needed
        if meta is not None:
            target_h = meta["height"]
            target_w = meta["width"]
            if s1.shape[1:] != (target_h, target_w):
                import cv2
                s1_resampled = np.zeros((s1.shape[0], target_h, target_w), dtype=np.float32)
                for b in range(s1.shape[0]):
                    s1_resampled[b] = cv2.resize(s1[b], (target_w, target_h), cv2.INTER_LINEAR)
                s1 = s1_resampled

        sar_feats = compute_sar_features(s1)
        for name, arr in sar_feats.items():
            feature_arrays.append(arr[np.newaxis])
            feature_names.append(f"sar_{name}")

    # ---- UAV features ----
    if uav_available and "uav" in preprocessed:
        _progress("UAV features (RGB indices, texture)")
        uav_10m = preprocessed["uav"]["data_10m"]  # [B, H', W']

        if meta is not None:
            import cv2
            target_h = meta["height"]
            target_w = meta["width"]
            uav_resampled = np.zeros((uav_10m.shape[0], target_h, target_w), dtype=np.float32)
            for b in range(uav_10m.shape[0]):
                uav_resampled[b] = cv2.resize(uav_10m[b], (target_w, target_h), cv2.INTER_LINEAR)
            uav_10m = uav_resampled

        uav_feats = compute_uav_features(uav_10m)
        for name, arr in uav_feats.items():
            feature_arrays.append(arr[np.newaxis])
            feature_names.append(f"uav_{name}")

    if not feature_arrays:
        raise RuntimeError("No preprocessed data available for feature extraction.")

    # Stack all features into a single tensor
    logger.info(f"  Stacking {len(feature_arrays)} feature arrays → [{sum(a.shape[0] for a in feature_arrays)}, H, W]")
    feature_stack = np.concatenate(feature_arrays, axis=0)  # [C, H, W]

    # Replace NaN with 0
    feature_stack = np.nan_to_num(feature_stack, nan=0.0, posinf=1.0, neginf=0.0)

    # Save
    out_path = os.path.join(paths["features"], "feature_stack.tif")
    if meta is not None:
        numpy_to_geotiff(feature_stack, out_path, meta, dtype="float32")

    feat_names_path = os.path.join(paths["features"], "feature_names.json")
    import json
    with open(feat_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    logger.info(
        f"Feature extraction complete: {feature_stack.shape[0]} channels, "
        f"shape={feature_stack.shape} → {out_path}"
    )

    return {
        "feature_stack": feature_stack,
        "feature_names": feature_names,
        "meta": meta,
        "output_path": out_path,
        "n_features": feature_stack.shape[0],
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    eps: float = 1e-8
) -> np.ndarray:
    """Compute a ratio, clamped to [-1, 1], avoiding division by zero."""
    result = numerator / (denominator + eps)
    return np.clip(result, -1.0, 1.0).astype(np.float32)