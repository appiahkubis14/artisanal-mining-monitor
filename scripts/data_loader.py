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

    # Align all scenes to the largest scene's shape before stacking.
    # Different S2 tiles (e.g. T30NYM vs T30NYN) clip to different shapes
    # over the same boundary. Use the most common / largest shape as reference.
    shapes = [a.shape[1:] for a in all_stacks]
    from collections import Counter
    ref_shape = Counter(shapes).most_common(1)[0][0]  # most frequent shape
    aligned = []
    for i, arr in enumerate(all_stacks):
        if arr.shape[1:] != ref_shape:
            logger.info(
                f"Resampling S2 scene {i} ({all_dates[i]}) "
                f"from {arr.shape[1:]} to {ref_shape}"
            )
            arr = _resample_array(arr, ref_shape)
        aligned.append(arr)

    # Stack time dimension
    data_4d = np.stack(aligned, axis=0)  # [T, B, H, W]

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
    Load bands from a Sentinel-2 L2A .SAFE directory structure.

    Band / resolution mapping (L2A):
        B02, B03, B04  → R10m  (Blue, Green, Red)
        B08            → R10m  (NIR broad)
        B11, B12       → R20m  (SWIR1, SWIR2)  — resampled to 10m

    All bands are clipped to the Atewa boundary and resampled to a
    common 10m grid before stacking.

    Args:
        safe_path    : Path to .SAFE folder.
        boundary_gdf : Boundary for clipping (any CRS — reprojected internally).

    Returns:
        Tuple of (array[6, H, W] float32, metadata dict, date_string YYYYMMDD).
    """
    # Preferred resolution per band
    band_res = {
        "B02": "R10m",
        "B03": "R10m",
        "B04": "R10m",
        "B08": "R10m",
        "B11": "R20m",
        "B12": "R20m",
    }
    band_files: Dict[str, str] = {}

    for band, res in band_res.items():
        # Search preferred resolution first
        pattern_pref = os.path.join(
            safe_path, "GRANULE", "**", res, f"*_{band}_{res[1:]}*.jp2"
        )
        matches = glob.glob(pattern_pref, recursive=True)

        if not matches:
            # Fallback: any resolution
            pattern_any = os.path.join(
                safe_path, "GRANULE", "**", f"*_{band}_*.jp2"
            )
            matches = glob.glob(pattern_any, recursive=True)

        if not matches:
            # Last resort: loose match
            pattern_loose = os.path.join(
                safe_path, "GRANULE", "**", f"*{band}*.jp2"
            )
            matches = sorted(glob.glob(pattern_loose, recursive=True))
            # Prefer the file whose path contains the preferred resolution
            pref_matches = [m for m in matches if res in m]
            matches = pref_matches if pref_matches else matches

        if matches:
            band_files[band] = matches[0]
        else:
            logger.warning(f"Band {band} not found in {safe_path}")

    if len(band_files) < 3:
        raise ValueError(
            f"Fewer than 3 bands found in {safe_path}: "
            f"{list(band_files.keys())}"
        )

    # Extract date: S2X_MSIL2A_YYYYMMDDTHHMMSS_...
    safe_name = os.path.basename(safe_path)
    parts = safe_name.split("_")
    date_str = parts[2][:8] if len(parts) > 2 else "unknown"

    logger.debug(f"S2 band files for {date_str}: {band_files}")

    # Load and clip all bands; resample 20m bands to first-band (10m) shape
    arrays = []
    meta = None
    ref_shape = None

    for band in ["B02", "B03", "B04", "B08", "B11", "B12"]:
        if band not in band_files:
            logger.warning(
                f"Band {band} missing — filling with zeros."
            )
            if ref_shape is not None:
                arrays.append(np.zeros(ref_shape, dtype=np.float32))
            continue

        arr, m = _load_and_clip_raster(band_files[band], boundary_gdf)

        if meta is None:
            meta = m
            ref_shape = arr.shape[1:]   # (H, W) of first loaded band
        elif arr.shape[1:] != ref_shape:
            arr = _resample_array(arr, ref_shape)

        arrays.append(arr[0].astype(np.float32))

    if not arrays:
        raise ValueError(f"No bands loaded from {safe_path}")

    stacked = np.stack(arrays, axis=0)   # [6, H, W]
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

    # Resample all scenes to the reference shape of the first scene.
    # Slight shape differences (±1 pixel) arise from GCP warping rounding.
    ref_shape = all_stacks[0].shape[1:]   # (H, W) of first scene
    aligned = []
    for i, arr in enumerate(all_stacks):
        if arr.shape[1:] != ref_shape:
            logger.info(
                f"Resampling S1 scene {i} from {arr.shape[1:]} "
                f"to {ref_shape} to align with reference scene."
            )
            arr = _resample_array(arr, ref_shape)
        aligned.append(arr)

    data_4d = np.stack(aligned, axis=0)
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
    Load VV and VH bands from a Sentinel-1 GRD .SAFE directory.

    Sentinel-1 GRD TIFFs store pixel coordinates only — no embedded CRS or
    geotransform. Geolocation is in the annotation XML as a grid of GCPs
    (line, pixel, lat, lon). This function:
      1. Reads GCPs from the annotation XML
      2. Warps each band to EPSG:32630 (UTM 30N) using the GCPs
      3. Clips the warped raster to the Atewa boundary
      4. Returns stacked [2, H, W] float32 array

    Args:
        safe_path    : Path to .SAFE directory.
        boundary_gdf : Boundary GeoDataFrame for clipping.

    Returns:
        Tuple of (array[2, H, W], metadata dict, date_string YYYYMMDD).
    """
    import xml.etree.ElementTree as ET
    import tempfile
    from rasterio.crs import CRS

    # Extract date from SAFE name
    safe_name = os.path.basename(safe_path)
    parts = safe_name.split("_")
    date_str = parts[4][:8] if len(parts) > 4 else "unknown"

    TARGET_CRS = CRS.from_epsg(32630)   # UTM 30N — covers Ghana
    TARGET_RES = 10.0                   # metres — match Sentinel-2

    def _warp_band_to_utm(tiff_path: str, xml_path: str) -> Tuple[np.ndarray, Dict]:
        """
        Georeference S1 GRD band using GCPs from annotation XML, warp to UTM.

        Two-step approach:
          1. gdal_translate: embed GCPs into a VRT (no pixel modification)
          2. gdalwarp: reproject the VRT to UTM 30N at 10m

        This is the standard GDAL workflow for GCP-based georeferencing.
        """
        import subprocess
        import xml.etree.ElementTree as ET

        # --- Read GCPs from annotation XML ---
        tree   = ET.parse(xml_path)
        root   = tree.getroot()
        gcp_els = root.findall(
            ".//geolocationGridPointList/geolocationGridPoint"
        )
        if not gcp_els:
            raise ValueError(f"No GCPs found in {xml_path}")

        # Subsample — 25 evenly-spaced GCPs is plenty for polynomial warp
        step    = max(1, len(gcp_els) // 25)
        sampled = gcp_els[::step]

        # Build -gcp args for gdal_translate: pixel line lon lat elev
        gcp_args = []
        for el in sampled:
            col = el.find("pixel").text
            row = el.find("line").text
            lon = el.find("longitude").text
            lat = el.find("latitude").text
            gcp_args += ["-gcp", col, row, lon, lat, "0"]

        # Temp files
        with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as f:
            vrt_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            warped_path = f.name

        try:
            # Step 1 — embed GCPs into a VRT using gdal_translate
            cmd1 = (
                ["gdal_translate", "-of", "VRT", "-a_srs", "EPSG:4326"]
                + gcp_args
                + [tiff_path, vrt_path]
            )
            r1 = subprocess.run(cmd1, capture_output=True, text=True)
            if r1.returncode != 0:
                raise RuntimeError(
                    f"gdal_translate (GCP embed) failed:\n{r1.stderr[-300:]}"
                )

            # Step 2 — warp VRT to UTM 30N at 10m resolution
            cmd2 = [
                "gdalwarp",
                "-overwrite",
                "-t_srs", "EPSG:32630",
                "-tr",    str(TARGET_RES), str(TARGET_RES),
                "-r",     "bilinear",
                "-of",    "GTiff",
                vrt_path, warped_path,
            ]
            r2 = subprocess.run(cmd2, capture_output=True, text=True)
            if r2.returncode != 0:
                raise RuntimeError(
                    f"gdalwarp (UTM reproject) failed:\n{r2.stderr[-300:]}"
                )

            # Step 3 — clip to Atewa boundary
            arr, meta = _load_and_clip_raster(warped_path, boundary_gdf)

        finally:
            for p in (vrt_path, warped_path):
                if os.path.exists(p):
                    os.unlink(p)

        return arr, meta

    # Find VV and VH TIFFs and their matching annotation XMLs
    vv_tiffs = sorted(glob.glob(
        os.path.join(safe_path, "measurement", "*vv*.tiff")
    ))
    vh_tiffs = sorted(glob.glob(
        os.path.join(safe_path, "measurement", "*vh*.tiff")
    ))
    vv_xmls  = sorted(glob.glob(
        os.path.join(safe_path, "annotation", "*vv*.xml")
    ))
    vh_xmls  = sorted(glob.glob(
        os.path.join(safe_path, "annotation", "*vh*.xml")
    ))

    if not vv_tiffs or not vh_tiffs:
        raise ValueError(f"VV or VH band missing in {safe_path}")
    if not vv_xmls or not vh_xmls:
        raise ValueError(f"Annotation XML missing in {safe_path}")

    logger.info(f"Warping S1 VV band to UTM for {date_str}...")
    vv_arr, meta = _warp_band_to_utm(vv_tiffs[0], vv_xmls[0])

    logger.info(f"Warping S1 VH band to UTM for {date_str}...")
    vh_arr, _    = _warp_band_to_utm(vh_tiffs[0], vh_xmls[0])

    # Ensure VH matches VV shape
    if vh_arr.shape != vv_arr.shape:
        vh_arr = _resample_array(vh_arr, vv_arr.shape[1:])

    stacked = np.concatenate([vv_arr, vh_arr], axis=0)   # [2, H, W]
    logger.info(
        f"S1 SAFE loaded: {os.path.basename(safe_path)} "
        f"date={date_str} shape={stacked.shape}"
    )
    return stacked, meta, date_str


def _get_s1_epsg(safe_path: str) -> int:
    """
    Determine the UTM EPSG code for a Sentinel-1 SAFE file.

    Strategy:
    1. Parse the scene centre latitude/longitude from the manifest.safe
       to compute the UTM zone EPSG automatically.
    2. Fall back to EPSG:32630 (UTM 30N — covers Ghana) if parsing fails.

    Args:
        safe_path: Path to .SAFE directory.

    Returns:
        Integer EPSG code (e.g. 32630).
    """
    import xml.etree.ElementTree as ET

    # Try to read centre coordinates from manifest
    manifest = os.path.join(safe_path, "manifest.safe")
    if os.path.exists(manifest):
        try:
            tree = ET.parse(manifest)
            root = tree.getroot()

            # Look for footprint coordinates in manifest
            ns = {"safe": "http://www.esa.int/safe/sentinel-1.0"}
            coords_text = None

            # Try different XML paths where coordinates appear
            for tag in [
                ".//safe:frameSet/safe:frame/safe:footPrint/gml:coordinates",
                ".//gml:coordinates",
                ".//safe:coordinates",
            ]:
                for ns_prefix in [{"gml": "http://www.opengis.net/gml"}, {}]:
                    try:
                        el = root.find(tag, {**ns, **ns_prefix})
                        if el is not None and el.text:
                            coords_text = el.text.strip()
                            break
                    except Exception:
                        continue
                if coords_text:
                    break

            if coords_text:
                # Parse "lat,lon lat,lon ..." or "lon,lat ..."
                pairs = coords_text.split()
                lats, lons = [], []
                for pair in pairs:
                    parts = pair.split(",")
                    if len(parts) == 2:
                        try:
                            a, b = float(parts[0]), float(parts[1])
                            # GML uses lat,lon; determine which is which
                            if abs(a) <= 90 and abs(b) <= 180:
                                lats.append(a)
                                lons.append(b)
                        except ValueError:
                            continue

                if lats and lons:
                    centre_lat = sum(lats) / len(lats)
                    centre_lon = sum(lons) / len(lons)
                    epsg = _latlon_to_utm_epsg(centre_lat, centre_lon)
                    logger.info(
                        f"S1 EPSG derived from manifest: EPSG:{epsg} "
                        f"(centre {centre_lat:.2f}°N, {centre_lon:.2f}°E)"
                    )
                    return epsg

        except Exception as exc:
            logger.debug(f"Manifest parse failed for {safe_path}: {exc}")

    # Fallback: Ghana is in UTM zone 30N
    logger.warning(
        f"CRS not found in manifest for {safe_path}, "
        "defaulting to EPSG:32630 (UTM 30N — Ghana)"
    )
    return 32630


def _latlon_to_utm_epsg(lat: float, lon: float) -> int:
    """Compute UTM zone EPSG from latitude/longitude."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + zone   # Northern hemisphere
    else:
        return 32700 + zone   # Southern hemisphere


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

    # Fast-path: if processed files already exist, load from them directly.
    # This avoids re-warping Sentinel-1 GRD (which takes ~10 min) on every step.
    processed_dir = paths.get("processed_data", "data/processed")
    s2_processed  = os.path.join(processed_dir, "sentinel2", "sentinel2_processed.tif")
    s1_processed  = os.path.join(processed_dir, "sentinel1", "sentinel1_processed.tif")
    ls_processed  = os.path.join(processed_dir, "landsat",   "landsat_processed.tif")

    def _load_processed_tif(path, sensor_name):
        """Load a processed GeoTIFF and return a minimal data dict."""
        import rasterio
        if not os.path.exists(path):
            return None
        try:
            with rasterio.open(path) as src:
                arr  = src.read().astype(np.float32)
                meta = {
                    "crs":       src.crs,
                    "transform": src.transform,
                    "width":     src.width,
                    "height":    src.height,
                    "count":     src.count,
                    "dtype":     str(src.dtypes[0]),
                    "nodata":    src.nodata,
                    "bounds":    src.bounds,
                }
            # Wrap in the same dict structure the preprocessor expects
            logger.info(
                f"[FAST-LOAD] {sensor_name} from processed file → shape {arr.shape}"
            )
            return {
                "data":      arr[np.newaxis],   # add time dim: [1, B, H, W]
                "composite": arr,               # [B, H, W]
                "dates":     ["processed"],
                "meta":      meta,
                "filepath":  path,
                "type":      f"{sensor_name}_processed",
            }
        except Exception as exc:
            logger.warning(f"Could not load processed {sensor_name}: {exc}")
            return None

    all_processed = (
        os.path.exists(s2_processed) and
        os.path.exists(s1_processed)
    )

    if all_processed:
        logger.info(
            "Processed files found — loading from checkpoints "
            "(skipping raw SAFE loading and S1 warping)."
        )
        s2      = _load_processed_tif(s2_processed, "sentinel2")
        s1      = _load_processed_tif(s1_processed, "sentinel1")
        landsat = _load_processed_tif(ls_processed, "landsat") if os.path.exists(ls_processed) else None
        uav     = None  # UAV processed separately
    else:
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

    # 5. UAV (optional) — always check raw folder regardless of fast-path
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
    boundary_gdf: gpd.GeoDataFrame,
    force_epsg: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Open a raster file, reproject boundary to raster CRS, and clip to it.

    Args:
        filepath    : Path to raster file.
        boundary_gdf: Boundary GeoDataFrame (any CRS).
        force_epsg  : Override raster CRS with this EPSG (for S1 files
                      that lack an embedded CRS).

    Returns:
        Tuple of (clipped array [bands, H, W] float32, metadata dict).
    """
    from rasterio.crs import CRS as RioCRS

    with rasterio.open(filepath) as src:

        # Resolve raster CRS
        if force_epsg is not None:
            raster_crs = RioCRS.from_epsg(force_epsg)
            logger.info(f"Setting CRS EPSG:{force_epsg} for {filepath}")
        else:
            raster_crs = src.crs

        if raster_crs is None:
            raise ValueError(
                f"Raster has no CRS and force_epsg not provided: {filepath}"
            )

        # Always reproject boundary using EPSG code strings to avoid
        # CRS object comparison issues between pyproj/rasterio versions
        raster_epsg = raster_crs.to_epsg()
        if raster_epsg:
            bd_reproj = boundary_gdf.to_crs(epsg=raster_epsg)
        else:
            # Fall back to proj4 string if EPSG not available
            bd_reproj = boundary_gdf.to_crs(raster_crs.to_proj4())

        geom = [mapping(g) for g in bd_reproj.geometry]

        try:
            out_image, out_transform = rio_mask(
                src, geom, crop=True, filled=True
            )
        except Exception as e:
            logger.warning(
                f"Clip failed for {filepath} ({e}); reading full extent."
            )
            out_image     = src.read().astype(np.float32)
            out_transform = src.transform

        meta = {
            "crs":       raster_crs,
            "transform": out_transform,
            "width":     out_image.shape[2],
            "height":    out_image.shape[1],
            "count":     out_image.shape[0],
            "dtype":     str(src.dtypes[0]),
            "nodata":    src.nodata,
            "bounds":    rasterio.transform.array_bounds(
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