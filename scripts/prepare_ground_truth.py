"""
prepare_ground_truth.py - Ground-Truth Mask Generation
=======================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Converts field-collected GPS points (or UAV-derived polygons) to
binary segmentation masks matching the satellite tile geometry.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union
import cv2
from scripts.utils import get_logger, load_json, save_json

logger = get_logger(__name__)


# =============================================================================
# Core mask generation
# =============================================================================

def gps_to_mask(
    gt_df: pd.DataFrame,
    meta: Dict[str, Any],
    buffer_radius_m: float = 20.0,
) -> np.ndarray:
    """
    Convert GPS mining site coordinates to a binary raster mask.

    Each point is buffered by `buffer_radius_m` to create a circular footprint.
    The resulting mask matches the spatial extent and resolution of `meta`.

    Args:
        gt_df: DataFrame with columns [latitude, longitude, site_type, active_status].
        meta: Rasterio metadata dict (crs, transform, height, width).
        buffer_radius_m: Radius in metres for circular buffer around each point.

    Returns:
        Binary mask array [H, W] — 1 = mining site, 0 = background.
    """
    H = meta["height"]
    W = meta["width"]
    crs = meta["crs"]
    transform = meta["transform"]

    # Create GeoDataFrame from GPS points
    gdf = gpd.GeoDataFrame(
        gt_df,
        geometry=gpd.points_from_xy(gt_df["longitude"], gt_df["latitude"]),
        crs="EPSG:4326"
    )

    # Project to the raster's CRS
    gdf = gdf.to_crs(crs)

    # Buffer points to estimate spatial footprint
    # For projected CRS (metres), buffer is in metres directly
    if crs.is_geographic:
        # Convert metres to degrees (rough approximation at Atewa lat ≈ 6°N)
        buffer_deg = buffer_radius_m / 111_320.0
        gdf["geometry"] = gdf.geometry.buffer(buffer_deg)
    else:
        gdf["geometry"] = gdf.geometry.buffer(buffer_radius_m)

    if gdf.empty:
        logger.warning("No ground-truth points — returning empty mask.")
        return np.zeros((H, W), dtype=np.uint8)

    # Rasterize polygons to mask
    shapes = [(mapping(geom), 1) for geom in gdf.geometry if geom.is_valid]

    if not shapes:
        return np.zeros((H, W), dtype=np.uint8)

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    n_positive = mask.sum()
    logger.info(
        f"GPS mask: {len(gdf)} sites, buffer={buffer_radius_m}m, "
        f"positive pixels={n_positive} ({100*n_positive/(H*W):.3f}%)"
    )
    return mask


def uav_polygons_to_mask(
    uav_data: Dict[str, Any],
    meta: Dict[str, Any],
    ndvi_threshold: float = 0.25,
    brightness_threshold: float = 0.55,
) -> np.ndarray:
    """
    Derive mining site masks from UAV imagery using spectral thresholds.

    Mining clearings in the UAV image are characterised by:
      - Low NDVI (< 0.25) — bare soil, cleared vegetation
      - High brightness (> 0.55) — exposed ground / water reflectance

    This produces pseudo-labels when no manual field annotations exist.

    Args:
        uav_data: Preprocessed UAV dict with 'data_10m' [B, H', W'].
        meta: Satellite metadata (H, W to match).
        ndvi_threshold: Pixels below this NDVI are candidate mining.
        brightness_threshold: Pixels above this brightness support mining.

    Returns:
        Binary mask [H, W] aligned to satellite metadata.
    """
    import cv2

    uav_10m = uav_data["data_10m"]
    n_bands = uav_10m.shape[0]

    H_target = meta["height"]
    W_target = meta["width"]

    # Compute UAV-NDVI (requires at least 4 bands: R, G, B, NIR)
    if n_bands >= 4:
        red = uav_10m[0]
        nir = uav_10m[3]
        ndvi = (nir - red) / (nir + red + 1e-8)
    elif n_bands >= 3:
        # Use Green - Red ratio as proxy
        green, red = uav_10m[1], uav_10m[0]
        ndvi = (green - red) / (green + red + 1e-8)
    else:
        ndvi = uav_10m[0]

    # Brightness
    if n_bands >= 3:
        brightness = uav_10m[:3].mean(axis=0)
    else:
        brightness = uav_10m[0]

    # Binary thresholding
    mining_mask = (
        (ndvi < ndvi_threshold) & (brightness > brightness_threshold)
    ).astype(np.uint8)

    # Morphological cleanup
    kernel = np.ones((3, 3), dtype=np.uint8)
    mining_mask = cv2.morphologyEx(mining_mask, cv2.MORPH_CLOSE, kernel)
    mining_mask = cv2.morphologyEx(mining_mask, cv2.MORPH_OPEN, kernel)

    # Resize to satellite dimensions
    mining_mask_resized = cv2.resize(
        mining_mask.astype(np.float32), (W_target, H_target),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)

    n_positive = mining_mask_resized.sum()
    logger.info(
        f"UAV-derived mask: positive pixels={n_positive} "
        f"({100*n_positive/(H_target*W_target):.3f}%)"
    )
    return mining_mask_resized


# =============================================================================
# Per-tile mask creation
# =============================================================================

def create_tile_masks(
    full_mask: np.ndarray,
    tile_coords: List[Dict[str, int]],
    tile_ids: List[str],
    tile_size: int,
    output_dir: str,
    min_positive_fraction: float = 0.001,
) -> Dict[str, Any]:
    """
    Extract per-tile masks from the full-scene binary mask.

    Args:
        full_mask: Binary mask [H, W] — 1 = mining, 0 = background.
        tile_coords: List of coordinate dicts (from tiler.tile_dataset).
        tile_ids: List of tile ID strings matching tile_coords.
        tile_size: Tile size in pixels.
        output_dir: Where to save mask .npy files.
        min_positive_fraction: Tiles with fewer than this fraction of
                               positive pixels are saved as negatives
                               (still needed for balanced training).

    Returns:
        Dict with:
            - 'n_positive_tiles': tiles containing mining sites
            - 'n_negative_tiles': background-only tiles
            - 'tile_info': list of dicts with tile metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    H, W = full_mask.shape

    tile_info = []
    n_positive = 0
    n_negative = 0

    for tile_id, coord in zip(tile_ids, tile_coords):
        rs = coord["row_start"]
        cs = coord["col_start"]
        re = coord["row_end"]
        ce = coord["col_end"]

        # Extract mask patch
        patch = full_mask[rs:re, cs:ce]

        # Pad if tile is on image boundary
        pad_r = tile_size - patch.shape[0]
        pad_c = tile_size - patch.shape[1]
        if pad_r > 0 or pad_c > 0:
            patch = np.pad(patch, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0)

        positive_frac = patch.mean()
        is_positive = positive_frac >= min_positive_fraction

        # Save mask
        mask_path = os.path.join(output_dir, f"{tile_id}_mask.npy")
        np.save(mask_path, patch)

        if is_positive:
            n_positive += 1
        else:
            n_negative += 1

        tile_info.append({
            "tile_id": tile_id,
            "mask_path": mask_path,
            "positive_fraction": float(positive_frac),
            "is_positive": bool(is_positive),
            "row_start": rs, "col_start": cs,
        })

    logger.info(
        f"Tile masks created: {n_positive} positive, {n_negative} negative "
        f"→ {output_dir}"
    )

    summary = {
        "n_positive_tiles": n_positive,
        "n_negative_tiles": n_negative,
        "total_tiles": n_positive + n_negative,
        "positive_rate": n_positive / (n_positive + n_negative + 1e-8),
        "tile_info": tile_info,
    }
    save_json(summary, os.path.join(output_dir, "mask_summary.json"))
    return summary


# =============================================================================
# Mask loader (prioritised)
# =============================================================================

def load_training_masks(config: Dict[str, Any], uav_available: bool = False) -> Optional[np.ndarray]:
    """
    Load the best available training mask, in priority order:

    1. Auto-generated UAV masks (generate_uav_masks.py output) — preferred
    2. GPS point CSV converted to raster mask
    3. Raises FileNotFoundError if neither exists

    This function is called by the orchestrator; it returns a numpy array
    or raises so the caller can decide how to proceed.

    Args:
        config:        Loaded config dict.
        uav_available: Whether UAV data was detected by data_loader.

    Returns:
        Binary numpy array [H, W] — 1 = mining, 0 = background.
    """
    masks_dir = config["paths"]["masks"]
    uav_mask_path = os.path.join(masks_dir, "uav_mining_mask_10m.npy")
    gt_csv_path   = os.path.join(
        config["paths"].get("ground_truth", "data/ground_truth"),
        "mining_sites.csv",
    )

    # --- Priority 1: auto-generated UAV mask ---
    if os.path.exists(uav_mask_path):
        logger.info(
            "Using auto-generated UAV masks for training "
            f"(source: {uav_mask_path})"
        )
        return np.load(uav_mask_path)

    # --- Priority 2: GPS points CSV ---
    if os.path.exists(gt_csv_path):
        logger.info(
            "Auto-generated UAV mask not found. "
            "Falling back to GPS points CSV."
        )
        return None  # signal to orchestrator to run gps_to_mask

    # --- Nothing available ---
    raise FileNotFoundError(
        "No training labels found. "
        "Either run `python main.py --step generate_masks` to auto-generate "
        "masks from the UAV orthomosaic, or supply GPS waypoints at "
        f"{gt_csv_path}."
    )


# =============================================================================
# Orchestrator
# =============================================================================

def prepare_ground_truth(
    data_dict: Dict[str, Any],
    preprocessed: Dict[str, Any],
    tiling_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full ground-truth preparation pipeline.

    Priority order for mask generation:
      1. Auto-generated UAV masks from generate_uav_masks.py  ← NEW (highest)
      2. Field-collected GPS points
      3. UAV-derived spectral pseudo-labels (legacy fallback)
      4. No masks (inference-only mode)

    Args:
        data_dict: Output from data_loader.load_all_data().
        preprocessed: Output from preprocessor.preprocess_all().
        tiling_results: Output from tiler.run_tiling().
        config: Config dict.

    Returns:
        Dict with mask summary and mask metadata.
    """
    paths = config["paths"]
    masks_dir = paths["masks"]
    os.makedirs(masks_dir, exist_ok=True)

    gt_df = data_dict.get("ground_truth")
    uav_available = data_dict.get("uav_available", False)
    uav_data = preprocessed.get("uav")

    # Determine spatial reference (use Sentinel-2 if available)
    if "sentinel2" in preprocessed:
        meta = preprocessed["sentinel2"]["meta"]
    elif "sentinel1" in preprocessed:
        meta = preprocessed["sentinel1"]["meta"]
    else:
        raise RuntimeError("No preprocessed satellite data available for mask generation.")

    # --- Generate / load full-scene binary mask ---
    full_mask = None
    mask_source = "none"

    # Priority 1 — auto-generated UAV mask (generate_uav_masks.py output)
    uav_auto_mask_path = os.path.join(masks_dir, "uav_mining_mask_10m.npy")
    if os.path.exists(uav_auto_mask_path):
        logger.info(
            "Auto-generated UAV mask found — loading directly. "
            "Skipping GPS / spectral pseudo-label generation."
        )
        mask_loaded = np.load(uav_auto_mask_path)
        # Resize to satellite reference if dimensions differ
        H_target = meta["height"]
        W_target = meta["width"]
        if mask_loaded.shape != (H_target, W_target):
            import cv2
            mask_loaded = cv2.resize(
                mask_loaded.astype(np.float32),
                (W_target, H_target),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        full_mask = mask_loaded
        mask_source = "uav_auto_mask"

    elif os.path.exists(os.path.join(paths["ground_truth"], "mining_polygons.geojson")):
        logger.info("Using Google Earth Pro polygons for training masks.")
        full_mask = polygons_to_mask(
            os.path.join(paths["ground_truth"], "mining_polygons.geojson"),
            meta
        )
        mask_source = "polygons_google_earth"

    # Priority 2 — GPS field points
    elif gt_df is not None and len(gt_df) > 0:
        logger.info(f"Creating masks from {len(gt_df)} GPS ground-truth points.")
        full_mask = gps_to_mask(
            gt_df, meta,
            buffer_radius_m=20.0,
        )
        mask_source = "gps_points"

    # Priority 3 — Legacy UAV spectral pseudo-labels
    elif uav_available and uav_data is not None:
        logger.info("No GPS data — creating masks from UAV spectral thresholding.")
        full_mask = uav_polygons_to_mask(uav_data, meta)
        mask_source = "uav_pseudolabels"

    else:
        logger.warning(
            "No ground-truth data available. "
            "Run `python main.py --step generate_masks` to auto-generate "
            "masks, or supply GPS waypoints. "
            "Pipeline will continue in inference-only mode."
        )
        return {"mask_source": "none", "n_positive_tiles": 0, "tile_info": []}

    # Save full mask
    full_mask_path = os.path.join(masks_dir, "full_scene_mask.npy")
    np.save(full_mask_path, full_mask)
    logger.info(f"Full-scene mask saved → {full_mask_path}")

    # --- Create per-tile masks ---
    if "sentinel2" in tiling_results:
        tile_meta = tiling_results["sentinel2"]["meta"]
        tile_ids = tile_meta["tile_ids"]
        tile_coords = tile_meta["coords"]
        tile_size = tile_meta["tile_size"]
    else:
        logger.warning("No satellite tiles found — cannot create tile masks.")
        return {"mask_source": mask_source, "full_mask": full_mask, "n_positive_tiles": 0}

    summary = create_tile_masks(
        full_mask, tile_coords, tile_ids,
        tile_size=tile_size,
        output_dir=masks_dir,
    )
    summary["mask_source"] = mask_source
    summary["full_mask_path"] = full_mask_path

    logger.info(
        f"Ground-truth preparation complete: source={mask_source}, "
        f"positive_rate={summary['positive_rate']:.3f}"
    )
    return summary


def polygons_to_mask(
    geojson_path: str,
    meta: Dict[str, Any],
) -> np.ndarray:
    """
    Convert GeoJSON polygons (from Google Earth Pro) to binary raster mask.

    Args:
        geojson_path: Path to GeoJSON file with mining site polygons.
        meta: Rasterio metadata dict (crs, transform, height, width).

    Returns:
        Binary mask array [H, W] — 1 = mining, 0 = background.
    """
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"Polygon GeoJSON not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    H = meta["height"]
    W = meta["width"]
    crs = meta["crs"]
    transform = meta["transform"]

    # Project to raster CRS
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # Filter invalid geometries
    gdf = gdf[gdf.geometry.is_valid]

    if gdf.empty:
        logger.warning("No valid polygons — returning empty mask.")
        return np.zeros((H, W), dtype=np.uint8)

    shapes = [(mapping(geom), 1) for geom in gdf.geometry]

    mask = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )

    n_positive = mask.sum()
    logger.info(
        f"Polygon mask: {len(gdf)} polygons, "
        f"positive pixels={n_positive} ({100*n_positive/(H*W):.3f}%)"
    )
    return mask