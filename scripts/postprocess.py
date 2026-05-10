"""
postprocess.py
==============
Convert raw U-Net probability maps into actionable mining site intelligence.

Steps
-----
1. Threshold probability map → binary mask
2. Connected-component analysis → candidate polygons
3. Filter small polygons (<50 m²)
4. IoU-based deduplication across overlapping tiles
5. Per-polygon attribute computation (area, confidence, water proximity, activity)
6. Prioritisation scoring
7. Export GeoJSON + CSV

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_prob_map(prob_tif: Path) -> tuple[np.ndarray, Any]:
    """Load a probability GeoTIFF and return (array, rasterio profile)."""
    with rasterio.open(prob_tif) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
        transform = src.transform
        crs = src.crs
    return arr, profile, transform, crs


def _binary_to_polygons(
    binary: np.ndarray,
    transform,
    crs,
    min_area_m2: float = 50.0,
) -> gpd.GeoDataFrame:
    """Convert binary mask to filtered GeoDataFrame of polygons.

    Parameters
    ----------
    binary    : uint8 array (0/1)
    transform : affine transform matching the array
    crs       : rasterio CRS
    min_area_m2 : discard polygons smaller than this

    Returns
    -------
    GeoDataFrame with columns: geometry, pixel_count
    """
    # rasterio.features.shapes yields (geojson_geom, value) pairs
    results = [
        (shape(geom), val)
        for geom, val in shapes(binary.astype(np.uint8), transform=transform)
        if val == 1
    ]
    if not results:
        log.warning("No mining polygons found in binary mask.")
        return gpd.GeoDataFrame(columns=["geometry", "pixel_count"], crs=crs)

    geoms, _ = zip(*results)
    gdf = gpd.GeoDataFrame(geometry=list(geoms), crs=crs)

    # Reproject to UTM for metric area calculation
    gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["area_m2"] = gdf_utm.geometry.area
    gdf["pixel_count"] = (gdf["area_m2"] / (abs(transform.a) ** 2)).round().astype(int)

    # Filter by minimum area
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()
    gdf = gdf.reset_index(drop=True)
    log.info(f"Retained {len(gdf)} polygons after area filter (>={min_area_m2} m²).")
    return gdf


def _iou_dedup(gdf: gpd.GeoDataFrame, iou_thresh: float = 0.5) -> gpd.GeoDataFrame:
    """Merge overlapping polygons with IoU above threshold.

    Greedy algorithm: keep largest polygon, suppress overlapping smaller ones.

    Parameters
    ----------
    gdf        : GeoDataFrame with geometry and area_m2
    iou_thresh : merge if IoU > this value

    Returns
    -------
    Deduplicated GeoDataFrame
    """
    if len(gdf) <= 1:
        return gdf

    gdf = gdf.sort_values("area_m2", ascending=False).reset_index(drop=True)
    keep = [True] * len(gdf)

    for i in range(len(gdf)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(gdf)):
            if not keep[j]:
                continue
            a = gdf.geometry.iloc[i]
            b = gdf.geometry.iloc[j]
            if not a.intersects(b):
                continue
            inter = a.intersection(b).area
            union = a.union(b).area
            if union == 0:
                continue
            iou = inter / union
            if iou >= iou_thresh:
                keep[j] = False

    result = gdf[keep].copy().reset_index(drop=True)
    log.info(f"IoU dedup: {len(gdf)} → {len(result)} polygons.")
    return result


def _compute_confidence(
    gdf: gpd.GeoDataFrame,
    prob_arr: np.ndarray,
    transform,
    crs,
) -> gpd.GeoDataFrame:
    """Compute mean probability for each polygon from the probability map.

    Parameters
    ----------
    gdf      : polygon GeoDataFrame
    prob_arr : 2-D probability array
    transform: affine transform matching prob_arr
    crs      : CRS of gdf (must match prob_arr)

    Returns
    -------
    GeoDataFrame with 'confidence' column added
    """
    from rasterio.features import rasterize

    h, w = prob_arr.shape
    confidences = []
    for geom in gdf.geometry:
        mask = rasterize(
            [(mapping(geom), 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        pixels = prob_arr[mask == 1]
        conf = float(pixels.mean()) if len(pixels) > 0 else 0.0
        confidences.append(conf)
    gdf = gdf.copy()
    gdf["confidence"] = confidences
    return gdf


def _compute_water_proximity(
    gdf: gpd.GeoDataFrame,
    water_vector: Path | None = None,
    ndwi_tif: Path | None = None,
    buffer_m: float = 200.0,
) -> gpd.GeoDataFrame:
    """Compute binary water-proximity flag (1 if within buffer_m of water).

    Falls back to NDWI-derived water if no vector is provided.

    Parameters
    ----------
    gdf          : polygon GeoDataFrame (geographic CRS)
    water_vector : optional GeoJSON/Shapefile of water bodies
    ndwi_tif     : optional NDWI GeoTIFF (values >0 = water)
    buffer_m     : proximity radius in metres

    Returns
    -------
    GeoDataFrame with 'water_proximity' column (0–1 normalised distance score)
    """
    gdf = gdf.copy()

    if water_vector and Path(water_vector).exists():
        water = gpd.read_file(water_vector)
        water_union = unary_union(water.to_crs(gdf.estimate_utm_crs()).geometry)
        gdf_utm = gdf.to_crs(gdf.estimate_utm_crs())
        dist = gdf_utm.geometry.centroid.distance(water_union)
        gdf["water_proximity"] = np.clip(1.0 - dist / buffer_m, 0, 1)
    elif ndwi_tif and Path(ndwi_tif).exists():
        # Derive water proximity from NDWI raster
        with rasterio.open(ndwi_tif) as src:
            ndwi = src.read(1)
            water_mask = (ndwi > 0).astype(np.uint8)
            dist_px = ndimage.distance_transform_edt(1 - water_mask)
            pixel_size_m = abs(src.transform.a)
            dist_m = dist_px * pixel_size_m
        # Sample distance at polygon centroids
        from rasterio.sample import sample_gen
        import json as _json
        centroids = gdf.to_crs(src.crs).geometry.centroid
        with rasterio.open(ndwi_tif) as src:
            prox = []
            for pt in centroids:
                row, col = src.index(pt.x, pt.y)
                row = max(0, min(row, dist_m.shape[0] - 1))
                col = max(0, min(col, dist_m.shape[1] - 1))
                d = dist_m[row, col]
                prox.append(float(np.clip(1.0 - d / buffer_m, 0, 1)))
        gdf["water_proximity"] = prox
    else:
        log.warning("No water reference found – water_proximity set to 0.5.")
        gdf["water_proximity"] = 0.5

    return gdf


def _compute_activity(
    gdf: gpd.GeoDataFrame,
    equipment_geojson: Path | None = None,
    expansion_threshold: float = 0.20,
) -> gpd.GeoDataFrame:
    """Assign activity status based on equipment detections or area expansion.

    Parameters
    ----------
    gdf                  : mining polygon GeoDataFrame
    equipment_geojson    : UAV equipment detections (if available)
    expansion_threshold  : fraction area increase to flag as active

    Returns
    -------
    GeoDataFrame with 'activity' column and 'equipment_count' column
    """
    gdf = gdf.copy()
    gdf["equipment_count"] = 0
    gdf["activity"] = "UNKNOWN"

    if equipment_geojson and Path(equipment_geojson).exists():
        equip = gpd.read_file(equipment_geojson).to_crs(gdf.crs)
        for idx, row in gdf.iterrows():
            matches = equip[equip.geometry.within(row.geometry)]
            gdf.at[idx, "equipment_count"] = len(matches)
            if len(matches) > 0:
                gdf.at[idx, "activity"] = "ACTIVE"
            else:
                gdf.at[idx, "activity"] = "INACTIVE"
    else:
        # Heuristic: high confidence → likely active
        gdf["activity"] = np.where(gdf["confidence"] >= 0.6, "LIKELY_ACTIVE", "UNKNOWN")

    return gdf


def _compute_priority_score(
    gdf: gpd.GeoDataFrame,
    weights: dict,
) -> gpd.GeoDataFrame:
    """Compute normalised priority score for ranger dispatch.

    Score = w_area * area_norm
           + w_equipment * equipment_factor
           + w_water * water_proximity
           + w_conf * confidence
           + w_activity * activity_factor

    Parameters
    ----------
    gdf     : polygon GeoDataFrame with required columns
    weights : dict with keys area, equipment, water_proximity, confidence, activity

    Returns
    -------
    GeoDataFrame with 'priority_score' and 'priority_rank' columns
    """
    gdf = gdf.copy()

    # Normalise area to [0,1] using log scale (handles large range)
    area_vals = gdf["area_m2"].values
    if area_vals.max() > area_vals.min():
        area_norm = (np.log1p(area_vals) - np.log1p(area_vals.min())) / (
            np.log1p(area_vals.max()) - np.log1p(area_vals.min()) + 1e-8
        )
    else:
        area_norm = np.ones(len(gdf))

    # Equipment factor: sigmoid on count
    eq_count = gdf["equipment_count"].values.astype(float)
    equipment_factor = 1.0 / (1.0 + np.exp(-eq_count + 1))  # sigmoid centred at 1

    # Activity factor
    activity_map = {"ACTIVE": 1.0, "LIKELY_ACTIVE": 0.7, "INACTIVE": 0.2, "UNKNOWN": 0.5}
    activity_factor = gdf["activity"].map(activity_map).fillna(0.5).values

    score = (
        weights.get("area", 0.30) * area_norm
        + weights.get("equipment", 0.25) * equipment_factor
        + weights.get("water_proximity", 0.20) * gdf["water_proximity"].values
        + weights.get("confidence", 0.15) * gdf["confidence"].values
        + weights.get("activity", 0.10) * activity_factor
    )

    gdf["priority_score"] = score
    gdf["priority_rank"] = (
        gdf["priority_score"].rank(ascending=False, method="first").astype(int)
    )
    return gdf


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

@timer
def postprocess_detections(
    prob_tif: Path,
    config: dict,
    output_dir: Path,
    equipment_geojson: Path | None = None,
    water_vector: Path | None = None,
    ndwi_tif: Path | None = None,
    date_str: str = "unknown",
) -> gpd.GeoDataFrame:
    """Full postprocessing pipeline for a single probability map.

    Parameters
    ----------
    prob_tif           : path to U-Net probability GeoTIFF
    config             : loaded config dict
    output_dir         : directory for output files
    equipment_geojson  : UAV equipment detections (optional)
    water_vector       : water body vector (optional)
    ndwi_tif           : NDWI raster for water proximity (optional)
    date_str           : date label for output filenames

    Returns
    -------
    GeoDataFrame with all mining site attributes
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inf_cfg = config.get("inference", {})
    thresholds = inf_cfg.get("thresholds", [0.3, 0.5, 0.7])
    min_area = inf_cfg.get("min_polygon_area_m2", 50.0)
    weights = config.get("prioritization", {}).get("weights", {
        "area": 0.30, "equipment": 0.25, "water_proximity": 0.20,
        "confidence": 0.15, "activity": 0.10,
    })

    # Use medium confidence threshold as primary
    primary_threshold = thresholds[1] if len(thresholds) > 1 else 0.5

    log.info(f"Loading probability map: {prob_tif}")
    prob_arr, profile, transform, crs = _load_prob_map(prob_tif)

    log.info(f"Thresholding at {primary_threshold}...")
    binary = (prob_arr >= primary_threshold).astype(np.uint8)

    # Morphological cleanup
    binary = ndimage.binary_opening(binary, structure=np.ones((3, 3))).astype(np.uint8)
    binary = ndimage.binary_closing(binary, structure=np.ones((5, 5))).astype(np.uint8)

    log.info("Converting to polygons...")
    gdf = _binary_to_polygons(binary, transform, crs, min_area_m2=min_area)

    if len(gdf) == 0:
        log.warning("No mining polygons detected after filtering.")
        return gdf

    log.info("Deduplicating overlapping polygons...")
    gdf = _iou_dedup(gdf, iou_thresh=0.5)

    log.info("Computing confidence scores...")
    gdf = _compute_confidence(gdf, prob_arr, transform, crs)

    log.info("Computing water proximity...")
    gdf = _compute_water_proximity(gdf, water_vector, ndwi_tif)

    log.info("Computing activity status...")
    gdf = _compute_activity(gdf, equipment_geojson)

    log.info("Computing priority scores...")
    gdf = _compute_priority_score(gdf, weights)

    # Add metadata columns
    gdf["area_ha"] = (gdf["area_m2"] / 10000).round(4)
    gdf["date"] = date_str
    gdf["site_id"] = [f"ATEWA_{date_str}_{i:04d}" for i in gdf.index]
    gdf["detection_source"] = "satellite_unet"

    # Confidence tier
    def _tier(c):
        if c >= 0.7:
            return "HIGH"
        if c >= 0.5:
            return "MEDIUM"
        return "LOW"
    gdf["confidence_tier"] = gdf["confidence"].map(_tier)

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    geojson_path = output_dir / f"mining_detections_{date_str}.geojson"
    csv_path = output_dir / f"mining_summary_{date_str}.csv"
    priority_path = output_dir / f"priority_list_{date_str}.csv"

    # GeoJSON
    gdf_save = gdf.copy()
    gdf_save.to_file(geojson_path, driver="GeoJSON")
    log.info(f"Saved GeoJSON: {geojson_path}")

    # CSV summary
    csv_cols = [
        "site_id", "date", "area_m2", "area_ha", "confidence",
        "confidence_tier", "activity", "equipment_count",
        "water_proximity", "priority_score", "priority_rank",
    ]
    gdf[csv_cols].to_csv(csv_path, index=False)
    log.info(f"Saved CSV summary: {csv_path}")

    # Priority list (top sites sorted)
    priority_df = gdf[csv_cols].sort_values("priority_rank")
    priority_df.to_csv(priority_path, index=False)
    log.info(f"Saved priority list: {priority_path}")

    # Summary stats
    summary = {
        "date": date_str,
        "total_sites": len(gdf),
        "total_area_ha": float(gdf["area_ha"].sum().round(2)),
        "high_confidence": int((gdf["confidence_tier"] == "HIGH").sum()),
        "medium_confidence": int((gdf["confidence_tier"] == "MEDIUM").sum()),
        "low_confidence": int((gdf["confidence_tier"] == "LOW").sum()),
        "active_sites": int((gdf["activity"] == "ACTIVE").sum()),
        "top_priority_site": gdf.sort_values("priority_rank").iloc[0]["site_id"]
        if len(gdf) > 0 else None,
    }
    summary_path = output_dir / f"detection_summary_{date_str}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"\n{'='*50}\n"
        f"POSTPROCESSING COMPLETE — {date_str}\n"
        f"  Total sites   : {summary['total_sites']}\n"
        f"  Total area    : {summary['total_area_ha']} ha\n"
        f"  HIGH conf     : {summary['high_confidence']}\n"
        f"  ACTIVE sites  : {summary['active_sites']}\n"
        f"{'='*50}"
    )
    return gdf


@timer
def run_postprocess(config: dict) -> dict[str, gpd.GeoDataFrame]:
    """Postprocess all probability maps found in outputs directory.

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    Dict mapping date_str → GeoDataFrame
    """
    output_dir = Path(config["paths"]["outputs"])
    prob_maps = sorted(output_dir.glob("prob_map_*.tif"))

    if not prob_maps:
        log.error(
            "No probability maps found in outputs/. "
            "Run inference_satellite.py first."
        )
        return {}

    equipment_geojson = output_dir / "equipment_detections.geojson"
    ndwi_tif = Path(config["paths"]["features"]) / "ndwi.tif"

    results = {}
    for prob_tif in prob_maps:
        # Extract date from filename: prob_map_YYYY-MM-DD.tif
        stem = prob_tif.stem  # e.g. prob_map_2024-03-15
        date_str = stem.replace("prob_map_", "")
        log.info(f"\n--- Postprocessing: {date_str} ---")

        gdf = postprocess_detections(
            prob_tif=prob_tif,
            config=config,
            output_dir=output_dir,
            equipment_geojson=equipment_geojson if equipment_geojson.exists() else None,
            ndwi_tif=ndwi_tif if ndwi_tif.exists() else None,
            date_str=date_str,
        )
        results[date_str] = gdf

    log.info(f"Postprocessed {len(results)} detection maps.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Postprocess mining detection maps.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--prob_tif",
        default=None,
        help="Single probability TIF to process (optional; default: all in outputs/)",
    )
    parser.add_argument("--date", default="manual", help="Date label for output files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging = get_logger  # alias

    if args.prob_tif:
        postprocess_detections(
            prob_tif=Path(args.prob_tif),
            config=cfg,
            output_dir=Path(cfg["paths"]["outputs"]),
            date_str=args.date,
        )
    else:
        run_postprocess(cfg)
