"""
data_loader.py - Data Loading Module
=====================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Loads all input data (Sentinel-2, Sentinel-1, Landsat, UAV, ground-truth)
from user-provided folders, clips to the Atewa boundary, and returns a
unified data dictionary for downstream processing.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import geopandas as gpd
from shapely.geometry import mapping

from scripts.utils import get_logger, load_config, raster_to_numpy

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Boundary loading
# ---------------------------------------------------------------------------

def load_boundary(boundary_path: str) -> gpd.GeoDataFrame:
    """
    Load Atewa Forest Reserve boundary from GeoJSON.

    Args:
        boundary_path: Path to atewa_boundary.geojson.

    Returns:
        GeoDataFrame with a single polygon in WGS84 (EPSG:4326).

    Raises:
        FileNotFoundError: If boundary file is missing.
    """
    if not os.path.exists(boundary_path):
        raise FileNotFoundError(
            f"Boundary file not found: {boundary_path}\n"
            "Please place atewa_boundary.geojson in data/boundary/"
        )
    gdf = gpd.read_file(boundary_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")
    logger.info(f"Boundary loaded: {len(gdf)} polygon(s), CRS={gdf.crs}")
    return gdf


# ---------------------------------------------------------------------------
# Sentinel-2 loading
# ---------------------------------------------------------------------------

def load_sentinel2(
    sentinel2_dir: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Optional[Dict[str, Any]]:
    """
    Load Sentinel-2 Level-2A imagery from .SAFE folders or GeoTIFF files.

    Searches for:
      1. Processed GeoTIFF stacks (data/raw/sentinel2/*.tif)
      2. Sentinel-2 .SAFE folders (opens band TIFFs inside)

    Args:
        sentinel2_dir: Path to data/raw/sentinel2/
        boundary_gdf: Atewa boundary GeoDataFrame (WGS84).

    Returns:
        Dict with keys:
            - 'data': np.ndarray of shape [dates, bands, H, W]
            - 'dates': list of acquisition date strings
            - 'meta': rasterio metadata dict
            - 'filepath': source path string
        Returns None if no files found (prints warning).
    """
    if not os.path.isdir(sentinel2_dir):
        logger.warning(f"Sentinel-2 directory not found: {sentinel2_dir}")
        return None

    # Try pre-processed GeoTIFFs first
    tif_files = sorted(glob.glob(os.path.join(sentinel2_dir, "*.tif")) +
                       glob.glob(os.path.join(sentinel2_dir, "*.TIF")))
    safe_dirs = sorted(glob.glob(os.path.join(sentinel2_dir, "*.SAFE")))

    if not tif_files and not safe_dirs:
        logger.warning(
            "No Sentinel-2 files found. "
            "Download .SAFE files from Copernicus Open Access Hub "
            "and place them in data/raw/sentinel2/"
        )
        return None

    all_stacks = []
    all_dates = []

    # -- Load from GeoTIFFs --
    for tif_path in tif_files:
        try:
            arr, meta = _load_and_clip_raster(tif_path, boundary_gdf)
            all_stacks.append(arr)
            date_str = _extract_date_from_filename(tif_path)
            all_dates.append(date_str)
            logger.info(f"Sentinel-2 GeoTIFF loaded: {tif_path} → shape {arr.shape}")
        except Exception as exc:
            logger.error(f"Failed to load {tif_path}: {exc}")

    # -- Load from .SAFE directories --
    for safe_path in safe_dirs:
        try:
            arr, meta, date_str = _load_sentinel2_safe(safe_path, boundary_gdf)
            all_stacks.append(arr)
            all_dates.append(date_str)
            logger.info(f"Sentinel-2 SAFE loaded: {safe_path} → shape {arr.shape}")
        except Exception as exc:
            logger.error(f"Failed to load SAFE {safe_path}: {exc}")

    if not all_stacks:
        logger.warning("All Sentinel-2 loading attempts failed.")
        return None

    # Stack time dimension
    data_4d = np.stack(all_stacks, axis=0)  # [T, B, H, W]

    return {
        "data": data_4d,
        "dates": all_dates,
        "meta": meta,
        "filepath": sentinel2_dir,
        "type": "sentinel2",
    }


def _load_sentinel2_safe(
    safe_path: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Tuple[np.ndarray, Dict, str]:
    """
    Load bands from a Sentinel-2 .SAFE directory structure.

    Targets bands: B02, B03, B04, B08, B11, B12 (10m and 20m).

    Args:
        safe_path: Path to .SAFE folder.
        boundary_gdf: Boundary for clipping.

    Returns:
        Tuple of (array[bands, H, W], metadata, date_string).
    """
    # Band file patterns within SAFE structure
    band_names = ["B02", "B03", "B04", "B08", "B11", "B12"]
    band_files = {}

    for band in band_names:
        # Search in GRANULE subdirectory
        pattern = os.path.join(safe_path, "GRANULE", "**", f"*{band}*.jp2")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            # Try TIF fallback
            pattern = os.path.join(safe_path, "GRANULE", "**", f"*{band}*.tif")
            matches = glob.glob(pattern, recursive=True)
        if matches:
            band_files[band] = matches[0]

    if len(band_files) < 3:
        raise ValueError(f"Fewer than 3 bands found in {safe_path}: {list(band_files.keys())}")

    # Extract date from SAFE name: S2A_MSIL2A_YYYYMMDD...
    safe_name = os.path.basename(safe_path)
    parts = safe_name.split("_")
    date_str = parts[2][:8] if len(parts) > 2 else "unknown"

    arrays = []
    meta = None
    for band in band_names:
        if band in band_files:
            arr, m = _load_and_clip_raster(band_files[band], boundary_gdf)
            if meta is None:
                meta = m
                ref_shape = arr.shape[1:]
            else:
                # Resample to match first band if shape differs
                if arr.shape[1:] != ref_shape:
                    arr = _resample_array(arr, ref_shape)
            arrays.append(arr[0])  # single band
        else:
            # Fill missing band with zeros
            logger.warning(f"Band {band} not found in {safe_path}; filling with zeros.")
            if meta is not None:
                arrays.append(np.zeros(ref_shape, dtype=np.float32))

    stacked = np.stack(arrays, axis=0)  # [bands, H, W]
    return stacked, meta, date_str


# ---------------------------------------------------------------------------
# Sentinel-1 loading
# ---------------------------------------------------------------------------

def load_sentinel1(
    sentinel1_dir: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Optional[Dict[str, Any]]:
    """
    Load Sentinel-1 GRD data (VV and VH polarization) from .SAFE or GeoTIFF.

    Args:
        sentinel1_dir: Path to data/raw/sentinel1/
        boundary_gdf: Atewa boundary GeoDataFrame.

    Returns:
        Dict with keys: data [T, 2, H, W], dates, meta, filepath, type.
        Returns None if no files found.
    """
    if not os.path.isdir(sentinel1_dir):
        logger.warning(f"Sentinel-1 directory not found: {sentinel1_dir}")
        return None

    tif_files = sorted(glob.glob(os.path.join(sentinel1_dir, "*.tif")) +
                       glob.glob(os.path.join(sentinel1_dir, "*.TIF")))
    safe_dirs = sorted(glob.glob(os.path.join(sentinel1_dir, "*.SAFE")))

    if not tif_files and not safe_dirs:
        logger.warning(
            "No Sentinel-1 files found. "
            "Download .SAFE files from Copernicus Open Access Hub "
            "and place them in data/raw/sentinel1/"
        )
        return None

    all_stacks = []
    all_dates = []
    meta = None

    for tif_path in tif_files:
        try:
            arr, m = _load_and_clip_raster(tif_path, boundary_gdf)
            all_stacks.append(arr)
            all_dates.append(_extract_date_from_filename(tif_path))
            if meta is None:
                meta = m
            logger.info(f"Sentinel-1 TIF loaded: {tif_path} → {arr.shape}")
        except Exception as exc:
            logger.error(f"S1 load failed for {tif_path}: {exc}")

    for safe_path in safe_dirs:
        try:
            arr, m, date_str = _load_sentinel1_safe(safe_path, boundary_gdf)
            all_stacks.append(arr)
            all_dates.append(date_str)
            if meta is None:
                meta = m
            logger.info(f"Sentinel-1 SAFE loaded: {safe_path} → {arr.shape}")
        except Exception as exc:
            logger.error(f"S1 SAFE load failed for {safe_path}: {exc}")

    if not all_stacks:
        logger.warning("No Sentinel-1 data loaded successfully.")
        return None

    data_4d = np.stack(all_stacks, axis=0)
    return {
        "data": data_4d,
        "dates": all_dates,
        "meta": meta,
        "filepath": sentinel1_dir,
        "type": "sentinel1",
    }


def _load_sentinel1_safe(
    safe_path: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Tuple[np.ndarray, Dict, str]:
    """
    Load VV and VH bands from a Sentinel-1 .SAFE directory.

    Args:
        safe_path: Path to .SAFE folder.
        boundary_gdf: Boundary for clipping.

    Returns:
        Tuple of (array[2, H, W], metadata, date_string).
    """
    safe_name = os.path.basename(safe_path)
    parts = safe_name.split("_")
    date_str = parts[4][:8] if len(parts) > 4 else "unknown"

    vv_pat = os.path.join(safe_path, "measurement", "*vv*.tiff")
    vh_pat = os.path.join(safe_path, "measurement", "*vh*.tiff")

    vv_files = glob.glob(vv_pat)
    vh_files = glob.glob(vh_pat)

    if not vv_files or not vh_files:
        raise ValueError(f"VV or VH band missing in {safe_path}")

    vv_arr, meta = _load_and_clip_raster(vv_files[0], boundary_gdf)
    vh_arr, _ = _load_and_clip_raster(vh_files[0], boundary_gdf)

    stacked = np.concatenate([vv_arr, vh_arr], axis=0)  # [2, H, W]
    return stacked, meta, date_str


# ---------------------------------------------------------------------------
# Landsat loading
# ---------------------------------------------------------------------------

def load_landsat(
    landsat_dir: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Optional[Dict[str, Any]]:
    """
    Load Landsat 8/9 Collection-2 data (optional fallback sensor).

    Expects band TIF files in standard Landsat naming: *_B2.TIF, *_B3.TIF, etc.

    Args:
        landsat_dir: Path to data/raw/landsat/
        boundary_gdf: Atewa boundary GeoDataFrame.

    Returns:
        Dict with data, dates, meta, filepath, type. None if unavailable.
    """
    if not os.path.isdir(landsat_dir):
        logger.info("Landsat directory not found – skipping (optional sensor).")
        return None

    # Find scene directories
    scene_dirs = [
        d for d in glob.glob(os.path.join(landsat_dir, "LC0*"))
        if os.path.isdir(d)
    ]
    # Also look for flat TIF files
    tif_files = glob.glob(os.path.join(landsat_dir, "*_B2.TIF"))

    if not scene_dirs and not tif_files:
        logger.info("No Landsat data found – skipping (optional sensor).")
        return None

    all_stacks = []
    all_dates = []
    meta = None
    band_nums = [2, 3, 4, 5, 6, 7]  # Blue, Green, Red, NIR, SWIR1, SWIR2

    source_list = scene_dirs if scene_dirs else [landsat_dir]

    for scene_dir in source_list:
        try:
            arrays = []
            date_str = "unknown"
            for b in band_nums:
                pattern = os.path.join(scene_dir, f"*_B{b}.TIF")
                band_files = glob.glob(pattern)
                if band_files:
                    arr, m = _load_and_clip_raster(band_files[0], boundary_gdf)
                    if meta is None:
                        meta = m
                    arrays.append(arr[0])
                    # Extract date from filename
                    name = os.path.basename(band_files[0])
                    if len(name) > 17:
                        date_str = name[17:25]
            if arrays:
                stacked = np.stack(arrays, axis=0)
                all_stacks.append(stacked)
                all_dates.append(date_str)
                logger.info(f"Landsat scene loaded: {scene_dir} → {stacked.shape}")
        except Exception as exc:
            logger.error(f"Landsat scene load failed: {scene_dir}: {exc}")

    if not all_stacks:
        return None

    return {
        "data": np.stack(all_stacks, axis=0),
        "dates": all_dates,
        "meta": meta,
        "filepath": landsat_dir,
        "type": "landsat",
    }


# ---------------------------------------------------------------------------
# UAV loading
# ---------------------------------------------------------------------------

def load_uav(
    uav_dir: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Optional[Dict[str, Any]]:
    """
    Load UAV orthomosaic GeoTIFF processed in Agisoft Metashape.

    Expects a single (or multiple) GeoTIFF file(s) with RGB (or RGB+NIR) bands.

    Args:
        uav_dir: Path to data/raw/uav/
        boundary_gdf: Atewa boundary GeoDataFrame.

    Returns:
        Dict with:
            - 'data': np.ndarray [bands, H, W] (float32, 0–1 normalised)
            - 'meta': rasterio metadata
            - 'resolution_m': ground sampling distance in metres
            - 'filepath': path to the GeoTIFF
        Returns None if no UAV file found.
    """
    if not os.path.isdir(uav_dir):
        logger.info("UAV directory not found – UAV fusion disabled.")
        return None

    tif_files = sorted(
        glob.glob(os.path.join(uav_dir, "*.tif")) +
        glob.glob(os.path.join(uav_dir, "*.TIF")) +
        glob.glob(os.path.join(uav_dir, "*.tiff")) +
        glob.glob(os.path.join(uav_dir, "*.TIFF"))
    )

    if not tif_files:
        logger.info("No UAV orthomosaic found in data/raw/uav/ – UAV fusion disabled.")
        return None

    # Take the largest file (likely the full orthomosaic)
    tif_files_sorted = sorted(tif_files, key=os.path.getsize, reverse=True)
    uav_path = tif_files_sorted[0]

    logger.info(f"Loading UAV orthomosaic: {uav_path}")

    try:
        arr, meta = _load_and_clip_raster(uav_path, boundary_gdf)

        # Estimate GSD from transform
        gsd_x = abs(meta["transform"].a)
        gsd_y = abs(meta["transform"].e)
        gsd_m = (gsd_x + gsd_y) / 2.0

        # Normalise to [0, 1]
        arr = arr.astype(np.float32)
        for b in range(arr.shape[0]):
            band = arr[b]
            p2, p98 = np.percentile(band[np.isfinite(band)], [2, 98])
            arr[b] = np.clip((band - p2) / (p98 - p2 + 1e-8), 0, 1)

        logger.info(
            f"UAV loaded: shape={arr.shape}, GSD≈{gsd_m:.4f}m, "
            f"area≈{arr.shape[1]*gsd_m/1000:.2f}km × {arr.shape[2]*gsd_m/1000:.2f}km"
        )

        return {
            "data": arr,
            "meta": meta,
            "resolution_m": gsd_m,
            "filepath": uav_path,
            "type": "uav",
        }
    except Exception as exc:
        logger.error(f"UAV loading failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(
    gt_path: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Optional[pd.DataFrame]:
    """
    Load field-collected ground-truth GPS points from CSV.

    Expected CSV columns:
        latitude, longitude, site_type, active_status, size_m2, notes

    Args:
        gt_path: Path to data/ground_truth/mining_sites.csv
        boundary_gdf: Atewa boundary for spatial filtering.

    Returns:
        Filtered DataFrame of points within Atewa boundary,
        or None if file not found.
    """
    if not os.path.exists(gt_path):
        logger.warning(
            f"Ground-truth CSV not found: {gt_path}\n"
            "Pipeline will use UAV-derived masks if available, "
            "or skip supervised training."
        )
        return None

    try:
        df = pd.read_csv(gt_path)
        required_cols = {"latitude", "longitude"}
        if not required_cols.issubset(df.columns):
            logger.error(
                f"Ground-truth CSV missing required columns: "
                f"{required_cols - set(df.columns)}"
            )
            return None

        # Add default columns if missing
        for col, default in [
            ("site_type", "galamsey"),
            ("active_status", "unknown"),
            ("size_m2", 0),
            ("notes", ""),
        ]:
            if col not in df.columns:
                df[col] = default

        # Spatial filter to Atewa boundary
        gdf_gt = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326"
        )
        boundary_union = boundary_gdf.union_all() if hasattr(boundary_gdf, 'union_all') else boundary_gdf.unary_union
        inside = gdf_gt[gdf_gt.within(boundary_union)]

        logger.info(
            f"Ground-truth loaded: {len(df)} total, "
            f"{len(inside)} within Atewa boundary"
        )
        return inside.drop(columns="geometry").reset_index(drop=True)

    except Exception as exc:
        logger.error(f"Ground-truth loading failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_all_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load all available data sources and return a unified data dictionary.

    This is the primary function called by main.py. It:
      1. Loads the Atewa boundary.
      2. Loads Sentinel-2, Sentinel-1, Landsat (optional), UAV (optional).
      3. Loads ground-truth CSV (optional).
      4. Sets uav_available flag.

    Args:
        config: Configuration dictionary from config.yaml.

    Returns:
        Dict with keys:
            sentinel2, sentinel1, landsat, uav, ground_truth, boundary,
            uav_available (bool)
    """
    paths = config["paths"]

    # 1. Boundary (required)
    boundary = load_boundary(paths["boundary"])

    # 2. Sentinel-2
    s2 = load_sentinel2(
        os.path.join(paths["raw_data"], "sentinel2"),
        boundary
    )

    # 3. Sentinel-1
    s1 = load_sentinel1(
        os.path.join(paths["raw_data"], "sentinel1"),
        boundary
    )

    # 4. Landsat (optional)
    landsat = load_landsat(
        os.path.join(paths["raw_data"], "landsat"),
        boundary
    )

    # 5. UAV (optional)
    uav = load_uav(
        os.path.join(paths["raw_data"], "uav"),
        boundary
    )

    # 6. Ground-truth (optional)
    gt = load_ground_truth(paths["ground_truth"], boundary)

    uav_available = uav is not None

    logger.info(
        f"Data loading complete | "
        f"S2={'✓' if s2 else '✗'} "
        f"S1={'✓' if s1 else '✗'} "
        f"Landsat={'✓' if landsat else '✗'} "
        f"UAV={'✓' if uav_available else '✗'} "
        f"GT={'✓' if gt is not None else '✗'}"
    )

    return {
        "sentinel2": s2,
        "sentinel1": s1,
        "landsat": landsat,
        "uav": uav,
        "ground_truth": gt,
        "boundary": boundary,
        "uav_available": uav_available,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_clip_raster(
    filepath: str,
    boundary_gdf: gpd.GeoDataFrame
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Open a raster file, reproject boundary if needed, and clip to it.

    Args:
        filepath: Path to raster file.
        boundary_gdf: Boundary GeoDataFrame (WGS84).

    Returns:
        Tuple of (clipped array [bands, H, W], metadata dict).
    """
    with rasterio.open(filepath) as src:
        # Reproject boundary to raster CRS
        if src.crs != boundary_gdf.crs:
            bd_reproj = boundary_gdf.to_crs(src.crs)
        else:
            bd_reproj = boundary_gdf

        geom = [mapping(geom) for geom in bd_reproj.geometry]

        try:
            out_image, out_transform = rio_mask(src, geom, crop=True, filled=True, fill_value=0)
        except Exception:
            # If clip fails (e.g., boundary outside raster), return full raster
            logger.warning(f"Clip failed for {filepath}; reading full extent.")
            out_image = src.read().astype(np.float32)
            out_transform = src.transform

        meta = {
            "crs": src.crs,
            "transform": out_transform,
            "width": out_image.shape[2],
            "height": out_image.shape[1],
            "count": out_image.shape[0],
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "bounds": rasterio.transform.array_bounds(
                out_image.shape[1], out_image.shape[2], out_transform
            ),
        }

    return out_image.astype(np.float32), meta


def _extract_date_from_filename(filepath: str) -> str:
    """
    Attempt to extract YYYYMMDD date string from a filename.

    Falls back to file modification timestamp if no date found.

    Args:
        filepath: Path to a raster file.

    Returns:
        Date string in YYYYMMDD format.
    """
    import re
    name = os.path.basename(filepath)
    # Match 8-digit sequences that look like a date (YYYYMMDD)
    matches = re.findall(r"\b(20\d{6})\b", name)
    if matches:
        return matches[0]
    # Fallback: file modification time
    mtime = os.path.getmtime(filepath)
    import datetime
    return datetime.datetime.fromtimestamp(mtime).strftime("%Y%m%d")


def _resample_array(
    array: np.ndarray,
    target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resample a [bands, H, W] array to a target spatial shape using bilinear interpolation.

    Args:
        array: Input array [bands, H, W].
        target_shape: (H, W) target shape.

    Returns:
        Resampled array [bands, target_H, target_W].
    """
    import cv2
    bands = array.shape[0]
    out = np.zeros((bands, target_shape[0], target_shape[1]), dtype=np.float32)
    for b in range(bands):
        out[b] = cv2.resize(
            array[b], (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    return out
