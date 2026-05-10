"""
change_detection.py
===================
Track mining sites across multiple dates to detect expansion, new sites,
and cessation of activity.

Workflow
--------
1. Load fused detection GeoJSONs for all dates (chronological order)
2. Match sites across dates using centroid proximity + IoU
3. Compute per-site change attributes (area delta, expansion rate)
4. Kalman filter smoothing of area time-series
5. RANSAC outlier rejection on expansion rates
6. Generate change report JSON + CSV
7. Export per-site time-series GeoJSON

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import mapping

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Kalman filter (1-D constant-velocity model for area time-series)
# ---------------------------------------------------------------------------

class KalmanFilter1D:
    """Simple 1-D Kalman filter for scalar time-series smoothing.

    State: [position, velocity]  (position = area in ha)
    """

    def __init__(
        self,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
        initial_value: float = 0.0,
    ):
        self.x = np.array([[initial_value], [0.0]])   # state [pos, vel]
        self.P = np.eye(2) * 1.0                      # covariance
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])  # transition
        self.H = np.array([[1.0, 0.0]])               # observation
        self.Q = np.eye(2) * process_noise            # process noise
        self.R = np.array([[measurement_noise]])      # measurement noise

    def predict(self) -> float:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0])

    def update(self, measurement: float) -> float:
        z = np.array([[measurement]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return float(self.x[0, 0])

    def step(self, measurement: float) -> float:
        self.predict()
        return self.update(measurement)


def smooth_area_series(areas: list[float], **kf_kwargs) -> list[float]:
    """Apply Kalman smoothing to a list of area measurements."""
    if len(areas) == 0:
        return []
    kf = KalmanFilter1D(initial_value=areas[0], **kf_kwargs)
    return [areas[0]] + [kf.step(a) for a in areas[1:]]


# ---------------------------------------------------------------------------
# RANSAC outlier rejection for expansion rates
# ---------------------------------------------------------------------------

def _ransac_expansion_rate(
    dates_ordinal: np.ndarray,
    areas: np.ndarray,
    n_iter: int = 200,
    inlier_thresh: float = 0.5,
) -> tuple[float, float]:
    """Estimate robust linear expansion rate (ha/month) using RANSAC.

    Parameters
    ----------
    dates_ordinal : ordinal day numbers
    areas         : area measurements in ha
    n_iter        : RANSAC iterations
    inlier_thresh : inlier residual threshold in ha

    Returns
    -------
    (slope_ha_per_day, intercept)
    """
    n = len(dates_ordinal)
    if n < 2:
        return 0.0, float(areas[0]) if n == 1 else 0.0

    best_slope, best_intercept = 0.0, float(np.mean(areas))
    best_n_inliers = 0

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n, size=2, replace=False)
        dx = dates_ordinal[idx[1]] - dates_ordinal[idx[0]]
        if dx == 0:
            continue
        slope = (areas[idx[1]] - areas[idx[0]]) / dx
        intercept = areas[idx[0]] - slope * dates_ordinal[idx[0]]
        residuals = np.abs(areas - (slope * dates_ordinal + intercept))
        mask = residuals < inlier_thresh
        n_inliers = mask.sum()
        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            if mask.sum() >= 2:
                # OLS refine on inliers
                x = dates_ordinal[mask]
                y = areas[mask]
                coeffs = np.polyfit(x, y, 1)
                best_slope, best_intercept = coeffs[0], coeffs[1]
            else:
                best_slope, best_intercept = slope, intercept

    return float(best_slope), float(best_intercept)


# ---------------------------------------------------------------------------
# Site matching across dates
# ---------------------------------------------------------------------------

@dataclass
class SiteTrack:
    """Accumulates observations of a single mining site across dates."""
    track_id: str
    dates: list[str] = field(default_factory=list)
    areas_ha: list[float] = field(default_factory=list)
    areas_ha_smoothed: list[float] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    equipment_counts: list[int] = field(default_factory=list)
    site_ids: list[str] = field(default_factory=list)
    geometry_latest: Any = None  # shapely geometry of latest observation

    @property
    def n_observations(self) -> int:
        return len(self.dates)

    @property
    def first_seen(self) -> str:
        return self.dates[0] if self.dates else "unknown"

    @property
    def last_seen(self) -> str:
        return self.dates[-1] if self.dates else "unknown"

    @property
    def area_latest_ha(self) -> float:
        return self.areas_ha_smoothed[-1] if self.areas_ha_smoothed else 0.0

    @property
    def area_initial_ha(self) -> float:
        return self.areas_ha[0] if self.areas_ha else 0.0

    @property
    def total_expansion_ha(self) -> float:
        if len(self.areas_ha) < 2:
            return 0.0
        return self.areas_ha_smoothed[-1] - self.areas_ha[0]

    def compute_expansion_rate(self) -> float:
        """Robust expansion rate in ha/month using RANSAC."""
        if len(self.dates) < 2:
            return 0.0
        try:
            ordinals = np.array([
                pd.Timestamp(d).toordinal() for d in self.dates
            ], dtype=float)
            areas = np.array(self.areas_ha, dtype=float)
            slope_per_day, _ = _ransac_expansion_rate(ordinals, areas)
            return float(slope_per_day * 30.44)  # ha/month
        except Exception:
            return 0.0

    def status(self) -> str:
        """Classify site status from trajectory."""
        if len(self.areas_ha) < 2:
            return "NEW"
        delta = self.total_expansion_ha
        rate = self.compute_expansion_rate()
        latest_activity = self.activities[-1] if self.activities else "UNKNOWN"

        if "ACTIVE" in latest_activity and rate > 0.01:
            return "EXPANDING"
        if delta > 0.05:
            return "GROWN"
        if delta < -0.05:
            return "REDUCED"
        if latest_activity in ("INACTIVE", "INACTIVE_INFRASTRUCTURE"):
            return "STABLE_INACTIVE"
        return "STABLE"


def _iou_geom(a, b) -> float:
    """Compute IoU between two shapely geometries."""
    try:
        inter = a.intersection(b).area
        union = a.union(b).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _match_sites(
    tracks: list[SiteTrack],
    new_gdf: gpd.GeoDataFrame,
    dist_thresh_m: float = 150.0,
    iou_thresh: float = 0.1,
) -> tuple[dict[int, str], list[int]]:
    """Match new detections to existing tracks.

    Parameters
    ----------
    tracks       : existing site tracks
    new_gdf      : GeoDataFrame of new detections (WGS84)
    dist_thresh_m: maximum centroid distance to consider a match
    iou_thresh   : minimum IoU for match confirmation

    Returns
    -------
    (matched, unmatched_indices)
    matched          : dict {new_gdf_index -> track_id}
    unmatched_indices: indices in new_gdf with no matching track
    """
    if len(tracks) == 0:
        return {}, list(new_gdf.index)

    matched = {}
    used_tracks = set()

    utm_crs = new_gdf.estimate_utm_crs()
    new_utm = new_gdf.to_crs(utm_crs)

    for new_idx, new_row in new_utm.iterrows():
        new_c = new_row.geometry.centroid
        best_track_id = None
        best_score = -1.0

        for track in tracks:
            if track.track_id in used_tracks:
                continue
            if track.geometry_latest is None:
                continue
            track_geom_utm = gpd.GeoSeries(
                [track.geometry_latest], crs="EPSG:4326"
            ).to_crs(utm_crs).iloc[0]
            dist = new_c.distance(track_geom_utm.centroid)
            if dist > dist_thresh_m:
                continue
            iou = _iou_geom(new_row.geometry, track_geom_utm)
            # Combined score: favour IoU, penalise distance
            score = iou * 0.7 + (1.0 - dist / dist_thresh_m) * 0.3
            if score > best_score and (iou >= iou_thresh or dist < 50):
                best_score = score
                best_track_id = track.track_id

        if best_track_id is not None:
            matched[new_idx] = best_track_id
            used_tracks.add(best_track_id)

    unmatched = [i for i in new_gdf.index if i not in matched]
    return matched, unmatched


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@timer
def run_change_detection(config: dict) -> pd.DataFrame:
    """Detect and characterise changes across all fused detection dates.

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    DataFrame with one row per tracked site and all change attributes
    """
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cd_cfg = config.get("change_detection", {})
    dist_thresh = cd_cfg.get("distance_threshold_m", 150.0)
    iou_thresh = cd_cfg.get("iou_threshold", 0.1)

    # ---- Load and sort fused detection files ----
    fused_files = sorted(output_dir.glob("fused_detections_[0-9]*.geojson"))
    if not fused_files:
        # Fall back to postprocessed files
        fused_files = sorted(output_dir.glob("mining_detections_[0-9]*.geojson"))

    if not fused_files:
        log.error(
            "No detection GeoJSONs found. Run postprocess.py and fusion.py first."
        )
        return pd.DataFrame()

    log.info(f"Processing {len(fused_files)} detection epochs.")

    # ---- Tracking loop ----
    tracks: list[SiteTrack] = []
    track_counter = 0

    for fused_file in fused_files:
        stem = fused_file.stem
        # Extract date from filename
        for prefix in ("fused_detections_", "mining_detections_"):
            stem = stem.replace(prefix, "")
        date_str = stem

        log.info(f"Processing epoch: {date_str}")
        gdf = gpd.read_file(fused_file).to_crs("EPSG:4326")
        if len(gdf) == 0:
            log.warning(f"No detections for {date_str}.")
            continue

        # Match to existing tracks
        matched, unmatched_idxs = _match_sites(tracks, gdf, dist_thresh, iou_thresh)

        # Update matched tracks
        track_map = {t.track_id: t for t in tracks}
        for new_idx, track_id in matched.items():
            row = gdf.loc[new_idx]
            track = track_map[track_id]
            track.dates.append(date_str)
            track.areas_ha.append(float(row.get("area_ha", 0.0)))
            track.confidences.append(float(row.get("confidence_fused", row.get("confidence", 0.5))))
            track.activities.append(str(row.get("activity_fused", row.get("activity", "UNKNOWN"))))
            track.equipment_counts.append(int(row.get("equipment_total", 0)))
            track.site_ids.append(str(row.get("site_id", "")))
            track.geometry_latest = row.geometry

        # Create new tracks for unmatched detections
        for new_idx in unmatched_idxs:
            row = gdf.loc[new_idx]
            track_counter += 1
            track_id = f"TRACK_{track_counter:04d}"
            track = SiteTrack(track_id=track_id)
            track.dates.append(date_str)
            track.areas_ha.append(float(row.get("area_ha", 0.0)))
            track.confidences.append(float(row.get("confidence_fused", row.get("confidence", 0.5))))
            track.activities.append(str(row.get("activity_fused", row.get("activity", "UNKNOWN"))))
            track.equipment_counts.append(int(row.get("equipment_total", 0)))
            track.site_ids.append(str(row.get("site_id", "")))
            track.geometry_latest = row.geometry
            tracks.append(track)

        log.info(
            f"  {date_str}: {len(matched)} matched, {len(unmatched_idxs)} new tracks. "
            f"Total tracks: {len(tracks)}"
        )

    if not tracks:
        log.warning("No site tracks created.")
        return pd.DataFrame()

    # ---- Smooth area time-series and compute change attributes ----
    log.info("Smoothing area time-series with Kalman filter...")
    for track in tracks:
        track.areas_ha_smoothed = smooth_area_series(track.areas_ha)

    # ---- Build output DataFrame ----
    rows = []
    for track in tracks:
        rate = track.compute_expansion_rate()
        rows.append({
            "track_id": track.track_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "n_observations": track.n_observations,
            "area_initial_ha": round(track.area_initial_ha, 4),
            "area_latest_ha": round(track.area_latest_ha, 4),
            "total_expansion_ha": round(track.total_expansion_ha, 4),
            "expansion_rate_ha_per_month": round(rate, 4),
            "status": track.status(),
            "latest_activity": track.activities[-1] if track.activities else "UNKNOWN",
            "max_equipment_count": max(track.equipment_counts) if track.equipment_counts else 0,
            "mean_confidence": round(np.mean(track.confidences), 4) if track.confidences else 0.0,
            "geometry": track.geometry_latest,
            "area_series_json": json.dumps(
                {"dates": track.dates, "areas_ha": track.areas_ha_smoothed}
            ),
        })

    df = pd.DataFrame(rows)

    # ---- Export change report CSV ----
    csv_cols = [c for c in df.columns if c not in ("geometry", "area_series_json")]
    csv_path = output_dir / "change_report.csv"
    df[csv_cols].sort_values("total_expansion_ha", ascending=False).to_csv(
        csv_path, index=False
    )
    log.info(f"Saved change report CSV: {csv_path}")

    # ---- Export change report GeoJSON ----
    gdf_out = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    geojson_path = output_dir / "change_report.geojson"
    gdf_out.drop(columns=["area_series_json"]).to_file(
        geojson_path, driver="GeoJSON"
    )
    log.info(f"Saved change report GeoJSON: {geojson_path}")

    # ---- Export time-series JSON ----
    ts_data = []
    for track in tracks:
        ts_data.append({
            "track_id": track.track_id,
            "dates": track.dates,
            "areas_ha_raw": track.areas_ha,
            "areas_ha_smoothed": track.areas_ha_smoothed,
            "confidences": track.confidences,
            "activities": track.activities,
            "equipment_counts": track.equipment_counts,
        })
    ts_path = output_dir / "site_time_series.json"
    with open(ts_path, "w") as f:
        json.dump(ts_data, f, indent=2)
    log.info(f"Saved site time-series JSON: {ts_path}")

    # ---- Summary statistics ----
    n_expanding = (df["status"] == "EXPANDING").sum()
    n_new = (df["status"] == "NEW").sum()
    n_stable = df["status"].str.startswith("STABLE").sum()
    total_area = df["area_latest_ha"].sum()

    summary = {
        "total_tracks": len(tracks),
        "expanding_sites": int(n_expanding),
        "new_sites": int(n_new),
        "stable_sites": int(n_stable),
        "total_active_area_ha": round(float(total_area), 2),
        "dates_processed": [
            f.stem.replace("fused_detections_", "").replace("mining_detections_", "")
            for f in fused_files
        ],
    }
    summary_path = output_dir / "change_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    log.info(
        f"\n{'='*55}\n"
        f"CHANGE DETECTION COMPLETE\n"
        f"  Total tracked sites   : {summary['total_tracks']}\n"
        f"  Expanding             : {summary['expanding_sites']}\n"
        f"  New                   : {summary['new_sites']}\n"
        f"  Stable                : {summary['stable_sites']}\n"
        f"  Total active area     : {summary['total_active_area_ha']} ha\n"
        f"{'='*55}"
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-date mining site change detection.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_change_detection(cfg)
