"""
fusion.py
=========
Fuse satellite U-Net detections with UAV equipment detections into a unified
site intelligence layer.

Workflow
--------
1. Load satellite postprocessed polygons (GeoJSON)
2. Load UAV equipment detections (GeoJSON) if available
3. RANSAC-based coordinate alignment between sensors
4. Spatial join: assign equipment detections to satellite polygons
5. Activity classification (ACTIVE / LIKELY_ACTIVE / INACTIVE / UNKNOWN)
6. Compute fused confidence score
7. Export fused GeoJSON with all attributes

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
from shapely.geometry import box, mapping, shape, Point
from shapely.ops import unary_union

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)

# Equipment classes that confirm active mining
ACTIVE_EQUIPMENT = {"excavator", "truck", "water_pump"}
INFRASTRUCTURE_EQUIPMENT = {"settling_pond", "pit"}


# ---------------------------------------------------------------------------
# RANSAC coordinate alignment
# ---------------------------------------------------------------------------

def _ransac_align(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    n_iter: int = 1000,
    inlier_thresh_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate translation offset between sensor coordinates using RANSAC.

    Assumes small translation only (sensors roughly co-registered).

    Parameters
    ----------
    src_points      : (N, 2) array of [lon, lat] source points (UAV)
    dst_points      : (N, 2) array of [lon, lat] destination points (satellite)
    n_iter          : RANSAC iterations
    inlier_thresh_m : inlier distance threshold in metres

    Returns
    -------
    (best_offset, inlier_mask) where best_offset is [dx, dy] in degrees
    """
    if len(src_points) < 3:
        log.warning("Too few control points for RANSAC – skipping alignment.")
        return np.array([0.0, 0.0]), np.ones(len(src_points), dtype=bool)

    # Convert threshold from metres to degrees (approximate)
    thresh_deg = inlier_thresh_m / 111_320.0

    best_offset = np.array([0.0, 0.0])
    best_n_inliers = 0
    best_mask = np.zeros(len(src_points), dtype=bool)

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        # Sample one pair
        idx = rng.integers(0, len(src_points))
        offset = dst_points[idx] - src_points[idx]
        # Evaluate
        residuals = np.linalg.norm((src_points + offset) - dst_points, axis=1)
        mask = residuals < thresh_deg
        n_inliers = mask.sum()
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_mask = mask
            # Refine with all inliers
            if n_inliers >= 2:
                best_offset = (dst_points[mask] - src_points[mask]).mean(axis=0)
            else:
                best_offset = offset

    log.info(
        f"RANSAC alignment: offset=[{best_offset[0]:.6f}°, {best_offset[1]:.6f}°], "
        f"inliers={best_n_inliers}/{len(src_points)}"
    )
    return best_offset, best_mask


def _apply_offset_to_gdf(gdf: gpd.GeoDataFrame, offset: np.ndarray) -> gpd.GeoDataFrame:
    """Translate all geometries in a GeoDataFrame by (dx, dy) in CRS units."""
    from shapely.affinity import translate
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: translate(g, xoff=float(offset[0]), yoff=float(offset[1]))
    )
    return gdf


# ---------------------------------------------------------------------------
# Control point extraction
# ---------------------------------------------------------------------------

def _extract_control_points(
    sat_gdf: gpd.GeoDataFrame,
    uav_gdf: gpd.GeoDataFrame,
    max_dist_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract matched control point pairs for RANSAC alignment.

    Matches UAV equipment centroids to nearest satellite polygon centroids.

    Parameters
    ----------
    sat_gdf    : satellite mining polygons
    uav_gdf    : UAV equipment detections
    max_dist_m : maximum match distance in metres

    Returns
    -------
    (src_points, dst_points) arrays of shape (N, 2) in WGS84 lon/lat
    """
    if len(sat_gdf) == 0 or len(uav_gdf) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # Work in projected CRS for distance calculation
    utm_crs = sat_gdf.estimate_utm_crs()
    sat_utm = sat_gdf.to_crs(utm_crs)
    uav_utm = uav_gdf.to_crs(utm_crs)

    sat_centroids = sat_utm.geometry.centroid
    uav_centroids = uav_utm.geometry.centroid

    src_pts, dst_pts = [], []
    for uav_c, uav_row in zip(uav_centroids, uav_gdf.itertuples()):
        dists = sat_centroids.distance(uav_c)
        nearest_idx = dists.idxmin()
        if dists[nearest_idx] <= max_dist_m:
            sat_c = sat_centroids[nearest_idx]
            # Back to WGS84
            uav_wgs = uav_gdf.to_crs("EPSG:4326").geometry.iloc[
                uav_gdf.index.get_loc(uav_row.Index)
            ].centroid
            sat_wgs = sat_gdf.to_crs("EPSG:4326").geometry.iloc[
                sat_gdf.index.get_loc(nearest_idx)
            ].centroid
            src_pts.append([uav_wgs.x, uav_wgs.y])
            dst_pts.append([sat_wgs.x, sat_wgs.y])

    return np.array(src_pts), np.array(dst_pts)


# ---------------------------------------------------------------------------
# Spatial join and attribute fusion
# ---------------------------------------------------------------------------

def _spatial_join_equipment(
    sat_gdf: gpd.GeoDataFrame,
    uav_gdf: gpd.GeoDataFrame,
    buffer_m: float = 30.0,
) -> gpd.GeoDataFrame:
    """Join UAV equipment detections to satellite polygons.

    Equipment points that fall within a satellite polygon (or within buffer_m)
    are attributed to that polygon.

    Parameters
    ----------
    sat_gdf  : satellite mining polygons
    uav_gdf  : UAV equipment detections (point or polygon)
    buffer_m : buffer around satellite polygons for matching

    Returns
    -------
    sat_gdf with additional columns: equipment_types, equipment_total,
    active_equipment_count, infrastructure_count
    """
    sat_gdf = sat_gdf.copy()
    sat_gdf["equipment_types"] = [[] for _ in range(len(sat_gdf))]
    sat_gdf["equipment_total"] = 0
    sat_gdf["active_equipment_count"] = 0
    sat_gdf["infrastructure_count"] = 0

    if len(uav_gdf) == 0:
        return sat_gdf

    utm_crs = sat_gdf.estimate_utm_crs()
    sat_utm = sat_gdf.to_crs(utm_crs)
    uav_utm = uav_gdf.to_crs(utm_crs)

    # Buffer satellite polygons
    sat_buffered = sat_utm.copy()
    sat_buffered["geometry"] = sat_utm.geometry.buffer(buffer_m)

    # Spatial join
    uav_centroids = uav_utm.copy()
    uav_centroids["geometry"] = uav_utm.geometry.centroid

    joined = gpd.sjoin(uav_centroids, sat_buffered, how="left", predicate="within")

    for sat_idx, sat_row in sat_gdf.iterrows():
        matches = joined[joined["index_right"] == sat_idx]
        if len(matches) == 0:
            continue
        classes = matches.get("class_name", pd.Series(dtype=str)).tolist()
        sat_gdf.at[sat_idx, "equipment_types"] = classes
        sat_gdf.at[sat_idx, "equipment_total"] = len(classes)
        sat_gdf.at[sat_idx, "active_equipment_count"] = sum(
            1 for c in classes if c in ACTIVE_EQUIPMENT
        )
        sat_gdf.at[sat_idx, "infrastructure_count"] = sum(
            1 for c in classes if c in INFRASTRUCTURE_EQUIPMENT
        )

    # Serialise list to string for GeoJSON compatibility
    sat_gdf["equipment_types"] = sat_gdf["equipment_types"].apply(
        lambda x: ",".join(x) if x else ""
    )
    return sat_gdf


def _classify_activity(row: pd.Series) -> str:
    """Classify mining site activity from fused attributes."""
    active_eq = row.get("active_equipment_count", 0)
    infra = row.get("infrastructure_count", 0)
    conf = row.get("confidence", 0.0)

    if active_eq >= 2:
        return "ACTIVE"
    if active_eq == 1 or (active_eq == 0 and infra >= 1 and conf >= 0.5):
        return "LIKELY_ACTIVE"
    if infra >= 1 or conf >= 0.6:
        return "INACTIVE_INFRASTRUCTURE"
    return "UNKNOWN"


def _fused_confidence(row: pd.Series, config: dict) -> float:
    """Compute fused confidence from satellite and UAV signals.

    Weighted combination:
      - satellite confidence (base)
      - UAV equipment corroboration bonus
      - cross-sensor consistency bonus
    """
    sat_conf = float(row.get("confidence", 0.5))
    active_eq = int(row.get("active_equipment_count", 0))
    total_eq = int(row.get("equipment_total", 0))

    fusion_cfg = config.get("fusion", {})
    sat_weight = fusion_cfg.get("satellite_weight", 0.6)
    uav_weight = fusion_cfg.get("uav_weight", 0.4)

    # UAV confidence: normalised equipment count
    uav_conf = min(1.0, total_eq / 3.0)  # saturates at 3 pieces of equipment

    # If no UAV data, rely fully on satellite
    if total_eq == 0:
        return sat_conf

    fused = sat_weight * sat_conf + uav_weight * uav_conf

    # Active equipment bonus
    if active_eq >= 1:
        fused = min(1.0, fused + 0.05 * active_eq)

    return round(float(fused), 4)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@timer
def run_fusion(
    config: dict,
    date_str: str | None = None,
) -> gpd.GeoDataFrame:
    """Fuse satellite and UAV detections for one or all dates.

    Parameters
    ----------
    config   : loaded config dict
    date_str : specific date to process; None = process all available

    Returns
    -------
    Fused GeoDataFrame (also saved to outputs/)
    """
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Discover satellite detection files ----
    if date_str:
        sat_files = list(output_dir.glob(f"mining_detections_{date_str}.geojson"))
    else:
        sat_files = sorted(output_dir.glob("mining_detections_*.geojson"))

    if not sat_files:
        log.error(
            "No satellite detection GeoJSONs found. Run postprocess.py first."
        )
        return gpd.GeoDataFrame()

    # ---- Load UAV equipment detections ----
    equip_path = output_dir / "equipment_detections.geojson"
    uav_available = equip_path.exists()
    if uav_available:
        uav_gdf = gpd.read_file(equip_path)
        log.info(f"Loaded {len(uav_gdf)} UAV equipment detections.")
    else:
        uav_gdf = gpd.GeoDataFrame(columns=["geometry", "class_name", "confidence"])
        log.warning("No UAV equipment detections found – proceeding satellite-only.")

    all_fused = []

    for sat_file in sat_files:
        file_date = sat_file.stem.replace("mining_detections_", "")
        log.info(f"\n--- Fusing: {file_date} ---")

        sat_gdf = gpd.read_file(sat_file)
        if len(sat_gdf) == 0:
            log.warning(f"No satellite detections for {file_date}, skipping.")
            continue

        # Ensure consistent CRS
        sat_gdf = sat_gdf.to_crs("EPSG:4326")

        if uav_available and len(uav_gdf) > 0:
            uav_crs_gdf = uav_gdf.to_crs("EPSG:4326")

            # RANSAC alignment
            src_pts, dst_pts = _extract_control_points(sat_gdf, uav_crs_gdf)
            if len(src_pts) >= 3:
                offset, _ = _ransac_align(src_pts, dst_pts)
                uav_crs_gdf = _apply_offset_to_gdf(uav_crs_gdf, offset)
                log.info("Applied RANSAC coordinate alignment to UAV detections.")
            else:
                log.info("Insufficient control points – using raw UAV coordinates.")

            # Spatial join
            sat_gdf = _spatial_join_equipment(sat_gdf, uav_crs_gdf)
        else:
            sat_gdf["equipment_types"] = ""
            sat_gdf["equipment_total"] = 0
            sat_gdf["active_equipment_count"] = 0
            sat_gdf["infrastructure_count"] = 0

        # Activity classification
        sat_gdf["activity_fused"] = sat_gdf.apply(_classify_activity, axis=1)

        # Fused confidence
        sat_gdf["confidence_fused"] = sat_gdf.apply(
            lambda row: _fused_confidence(row, config), axis=1
        )

        # Data source flag
        sat_gdf["fusion_source"] = (
            "satellite+uav" if uav_available else "satellite_only"
        )

        # Update priority score with fused confidence
        weights = config.get("prioritization", {}).get("weights", {
            "area": 0.30, "equipment": 0.25, "water_proximity": 0.20,
            "confidence": 0.15, "activity": 0.10,
        })
        activity_map = {
            "ACTIVE": 1.0, "LIKELY_ACTIVE": 0.7,
            "INACTIVE_INFRASTRUCTURE": 0.4, "INACTIVE": 0.2, "UNKNOWN": 0.5,
        }
        area_vals = sat_gdf["area_m2"].values
        if area_vals.max() > area_vals.min():
            area_norm = (np.log1p(area_vals) - np.log1p(area_vals.min())) / (
                np.log1p(area_vals.max()) - np.log1p(area_vals.min()) + 1e-8
            )
        else:
            area_norm = np.ones(len(sat_gdf))

        eq_factor = sat_gdf["active_equipment_count"].apply(
            lambda x: 1.0 / (1.0 + np.exp(-float(x) + 1))
        ).values
        act_factor = sat_gdf["activity_fused"].map(activity_map).fillna(0.5).values

        sat_gdf["priority_score_fused"] = (
            weights.get("area", 0.30) * area_norm
            + weights.get("equipment", 0.25) * eq_factor
            + weights.get("water_proximity", 0.20) * sat_gdf["water_proximity"].fillna(0.5).values
            + weights.get("confidence", 0.15) * sat_gdf["confidence_fused"].values
            + weights.get("activity", 0.10) * act_factor
        )
        sat_gdf["priority_rank_fused"] = (
            sat_gdf["priority_score_fused"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        # Save fused GeoJSON
        fused_path = output_dir / f"fused_detections_{file_date}.geojson"
        sat_gdf.to_file(fused_path, driver="GeoJSON")
        log.info(f"Saved fused detections: {fused_path}")

        # Save fused CSV
        csv_cols = [
            "site_id", "date", "area_ha", "confidence", "confidence_fused",
            "activity_fused", "equipment_total", "active_equipment_count",
            "equipment_types", "water_proximity", "priority_score_fused",
            "priority_rank_fused", "fusion_source",
        ]
        existing_cols = [c for c in csv_cols if c in sat_gdf.columns]
        csv_path = output_dir / f"fused_summary_{file_date}.csv"
        sat_gdf[existing_cols].sort_values("priority_rank_fused").to_csv(
            csv_path, index=False
        )
        log.info(f"Saved fused summary: {csv_path}")

        # Summary
        n_active = (sat_gdf["activity_fused"] == "ACTIVE").sum()
        n_likely = (sat_gdf["activity_fused"] == "LIKELY_ACTIVE").sum()
        log.info(
            f"\n{'='*50}\n"
            f"FUSION COMPLETE — {file_date}\n"
            f"  Sites         : {len(sat_gdf)}\n"
            f"  ACTIVE        : {n_active}\n"
            f"  LIKELY_ACTIVE : {n_likely}\n"
            f"  Fusion source : {'satellite+uav' if uav_available else 'satellite_only'}\n"
            f"{'='*50}"
        )
        all_fused.append(sat_gdf)

    if not all_fused:
        return gpd.GeoDataFrame()

    combined = pd.concat(all_fused, ignore_index=True)
    combined_path = output_dir / "fused_detections_all.geojson"
    gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326").to_file(
        combined_path, driver="GeoJSON"
    )
    log.info(f"Saved combined fused detections: {combined_path}")
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fuse satellite + UAV detections.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--date", default=None, help="Date string (e.g. 2024-03-15)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_fusion(cfg, date_str=args.date)
