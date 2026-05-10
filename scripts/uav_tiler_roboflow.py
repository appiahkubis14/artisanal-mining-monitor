"""
uav_tiler_roboflow.py
======================
Tile the UAV orthomosaic into 640×640 JPEG images for upload to Roboflow
and create a precise coordinate mapping so that every YOLO detection can
later be georeferenced back to real-world coordinates.

Workflow
--------
1. Load preprocessed UAV GeoTIFF (from data/processed/uav/)
2. Split into 640×640 tiles with configurable overlap
3. Save tiles as .jpg files in data/roboflow_upload/images/
4. Save a per-tile coordinate mapping JSON (affine + WGS84 bounds)
5. Create empty YOLO placeholder label files in data/roboflow_upload/labels/
6. Write a human-readable README_roboflow.txt with upload instructions

The coordinate mapping JSON is consumed by import_roboflow_annotations.py
and inference_uav.py to convert YOLO pixel coordinates → lon/lat.

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from pyproj import Transformer

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _affine_to_list(t: Affine) -> list[float]:
    """Convert rasterio Affine to a 6-element list [a, b, c, d, e, f]."""
    return [t.a, t.b, t.c, t.d, t.e, t.f]


def _tile_wgs84_bounds(
    col_start: int,
    row_start: int,
    tile_w: int,
    tile_h: int,
    src_transform: Affine,
    src_crs: CRS,
) -> dict[str, float]:
    """Compute WGS84 (lon/lat) bounding box for a tile.

    Parameters
    ----------
    col_start, row_start : top-left pixel offset in the source image
    tile_w, tile_h       : tile dimensions in pixels
    src_transform        : affine transform of the source image
    src_crs              : CRS of the source image

    Returns
    -------
    Dict with min_lon, max_lon, min_lat, max_lat, center_lon, center_lat
    """
    # Four corners of the tile in source CRS
    corners_src = []
    for dr, dc in [(0, 0), (0, tile_w), (tile_h, 0), (tile_h, tile_w)]:
        x = src_transform.c + (col_start + dc) * src_transform.a
        y = src_transform.f + (row_start + dr) * src_transform.e
        corners_src.append((x, y))

    # Transform to WGS84
    if src_crs.to_epsg() == 4326:
        corners_wgs = corners_src
    else:
        transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
        corners_wgs = [transformer.transform(x, y) for x, y in corners_src]

    lons = [c[0] for c in corners_wgs]
    lats = [c[1] for c in corners_wgs]

    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    return {
        "min_lon":    round(min_lon, 8),
        "max_lon":    round(max_lon, 8),
        "min_lat":    round(min_lat, 8),
        "max_lat":    round(max_lat, 8),
        "center_lon": round((min_lon + max_lon) / 2, 8),
        "center_lat": round((min_lat + max_lat) / 2, 8),
    }


def _tile_affine(
    col_start: int,
    row_start: int,
    src_transform: Affine,
) -> Affine:
    """Build the affine transform for a tile given its top-left pixel offset."""
    x0 = src_transform.c + col_start * src_transform.a
    y0 = src_transform.f + row_start * src_transform.e
    return Affine(src_transform.a, src_transform.b, x0,
                  src_transform.d, src_transform.e, y0)


# ---------------------------------------------------------------------------
# Core tiling
# ---------------------------------------------------------------------------

def tile_uav_for_roboflow(
    uav_tif: Path,
    output_dir: Path,
    tile_size: int = 640,
    overlap: float = 0.10,
    jpeg_quality: int = 92,
) -> list[dict[str, Any]]:
    """
    Split a UAV GeoTIFF into 640×640 JPEG tiles for Roboflow annotation.

    Each tile record contains:
      - tile_id, image_file
      - pixel dimensions
      - georeferencing dict (CRS string, WGS84 bounds, affine parameters)
      - source pixel offsets (col_start, row_start, col_end, row_end)
      - actual tile dims (may differ from tile_size at image edges)

    Parameters
    ----------
    uav_tif      : path to preprocessed UAV GeoTIFF
    output_dir   : root output directory (data/roboflow_upload/)
    tile_size    : YOLO-standard tile size in pixels (default 640)
    overlap      : fractional overlap between adjacent tiles (default 0.10)
    jpeg_quality : JPEG compression quality 0–100

    Returns
    -------
    List of tile metadata dicts (also saved to tile_coordinates.json)
    """
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading UAV GeoTIFF: {uav_tif}")
    with rasterio.open(uav_tif) as src:
        n_bands = src.count
        H = src.height
        W = src.width
        src_transform = src.transform
        src_crs = src.crs
        gsd_m = abs(src_transform.a)

        # Read all bands
        img_raw = src.read().astype(np.float32)   # [B, H, W]

    log.info(
        f"UAV image: {n_bands} bands, {H}×{W} px, "
        f"GSD={gsd_m:.4f} m, CRS={src_crs.to_string()}"
    )

    # Normalise to [0, 255] uint8 for JPEG
    if img_raw.max() > 1.0:
        p2  = np.percentile(img_raw,  2)
        p98 = np.percentile(img_raw, 98)
        img_norm = np.clip((img_raw - p2) / (p98 - p2 + 1e-8), 0, 1)
    else:
        img_norm = img_raw.clip(0, 1)

    img_u8 = (img_norm * 255).astype(np.uint8)   # [B, H, W]

    # Build RGB for JPEG (OpenCV expects BGR)
    if n_bands >= 3:
        rgb = img_u8[:3].transpose(1, 2, 0)       # [H, W, 3]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    elif n_bands == 1:
        gray = img_u8[0]
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(img_u8[0], cv2.COLOR_GRAY2BGR)

    stride = max(1, int(tile_size * (1 - overlap)))
    tile_records = []
    tile_counter = 0

    row = 0
    while row < H:
        row_end   = min(row + tile_size, H)
        row_start = max(row_end - tile_size, 0)

        col = 0
        while col < W:
            col_end   = min(col + tile_size, W)
            col_start = max(col_end - tile_size, 0)

            patch = bgr[row_start:row_end, col_start:col_end]

            # Pad to exactly tile_size × tile_size if on an edge
            pad_b = tile_size - patch.shape[0]
            pad_r = tile_size - patch.shape[1]
            if pad_b > 0 or pad_r > 0:
                patch = cv2.copyMakeBorder(
                    patch, 0, pad_b, 0, pad_r,
                    borderType=cv2.BORDER_REFLECT_101,
                )

            # Skip tiles that are entirely black / zero (outside raster extent)
            if patch.max() == 0:
                col += stride
                continue

            tile_id   = f"tile_{tile_counter:06d}"
            img_fname = f"{tile_id}.jpg"
            img_path  = images_dir / img_fname

            # Save JPEG
            cv2.imwrite(
                str(img_path),
                patch,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )

            # Create empty placeholder label file
            (labels_dir / f"{tile_id}.txt").touch()

            # Build coordinate record
            tile_affine = _tile_affine(col_start, row_start, src_transform)
            wgs84       = _tile_wgs84_bounds(
                col_start, row_start,
                col_end - col_start, row_end - row_start,
                src_transform, src_crs,
            )

            actual_h = row_end - row_start
            actual_w = col_end - col_start

            record = {
                "tile_id":     tile_id,
                "image_file":  img_fname,
                "image_width":  tile_size,
                "image_height": tile_size,
                "actual_width_px":  actual_w,   # unpadded pixel extent
                "actual_height_px": actual_h,
                "col_start":   col_start,
                "row_start":   row_start,
                "col_end":     col_end,
                "row_end":     row_end,
                "georeferencing": {
                    "crs":                src_crs.to_string(),
                    "epsg":               src_crs.to_epsg(),
                    **wgs84,
                    "pixel_to_geo_transform": _affine_to_list(tile_affine),
                },
            }
            tile_records.append(record)
            tile_counter += 1

            col += stride
        row += stride

    log.info(
        f"Tiling complete: {len(tile_records)} tiles → {images_dir}"
    )
    return tile_records


# ---------------------------------------------------------------------------
# Save coordinate mapping + README
# ---------------------------------------------------------------------------

def save_coordinate_mapping(
    tile_records: list[dict[str, Any]],
    output_dir: Path,
    uav_tif: Path,
    tile_size: int,
    overlap: float,
) -> Path:
    """Save tile_coordinates.json to output_dir."""
    mapping = {
        "source_tif":  str(uav_tif),
        "tile_size":   tile_size,
        "overlap":     overlap,
        "n_tiles":     len(tile_records),
        "tiles":       tile_records,
    }
    out_path = output_dir / "tile_coordinates.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    log.info(f"Coordinate mapping saved: {out_path}")
    return out_path


def write_roboflow_readme(output_dir: Path, n_tiles: int) -> Path:
    """Write human-readable upload instructions."""
    readme = output_dir / "README_roboflow.txt"
    content = f"""
ATEWA FOREST RESERVE — ROBOFLOW ANNOTATION PACKAGE
====================================================
Generated by: uav_tiler_roboflow.py
Tiles:        {n_tiles}
Tile size:    640 × 640 px (YOLO standard)

UPLOAD INSTRUCTIONS
-------------------
1. Go to https://roboflow.com and create a new project
   Project type : Object Detection
   Annotation format : YOLO v8

2. Upload the contents of the images/ folder
   (or zip the entire images/ folder and drag-and-drop)

3. Annotate the following equipment classes:
   Class 0 — excavator
   Class 1 — truck
   Class 2 — water_pump
   Class 3 — settling_pond
   Class 4 — pit

4. When annotation is complete, click "Generate Dataset" and
   export in "YOLOv8" format.

5. Download the export ZIP and place the contents in:
       data/roboflow_export/
   The folder should contain at minimum:
       data/roboflow_export/labels/*.txt   (one per tile)

6. Run the import step:
       python main.py --step import_roboflow

7. Then train the equipment detector:
       python main.py --step train_yolo

COORDINATE MAPPING
------------------
tile_coordinates.json maps every tile back to real-world coordinates.
Do NOT delete or rename this file — it is required for georeferencing
detections in the fusion step.

Each entry contains:
  - WGS84 bounding box (min/max lon/lat)
  - Affine transform parameters (pixel → CRS coordinates)
  - Original pixel offsets in the full UAV orthomosaic
"""
    readme.write_text(content.lstrip())
    log.info(f"README written: {readme}")
    return readme


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@timer
def run_uav_tiler_roboflow(config: dict) -> dict[str, Any]:
    """
    Full Roboflow tiling pipeline.

    Reads config from config.yaml → roboflow section.

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    Dict with output paths and tile count
    """
    rb_cfg   = config.get("roboflow", {})
    tile_size = int(rb_cfg.get("tile_size",        640))
    overlap   = float(rb_cfg.get("overlap_fraction", 0.10))
    out_dir   = Path(rb_cfg.get("output_dir", "data/roboflow_upload"))

    # Locate preprocessed UAV orthomosaic
    uav_proc_dir = Path(config["paths"].get("processed_data", config["paths"].get("processed", "data/processed"))) / "uav"
    candidates   = list(uav_proc_dir.glob("*.tif"))
    if not candidates:
        raise FileNotFoundError(
            f"No UAV GeoTIFF in {uav_proc_dir}. "
            "Run `python main.py --step preprocess` first."
        )
    uav_tif = next(
        (p for p in candidates if "highres" in p.name or "ortho" in p.name.lower()),
        candidates[0],
    )
    log.info(f"Source UAV GeoTIFF: {uav_tif}")

    # Tile
    tile_records = tile_uav_for_roboflow(
        uav_tif=uav_tif,
        output_dir=out_dir,
        tile_size=tile_size,
        overlap=overlap,
    )

    # Save coordinate mapping
    coord_json = save_coordinate_mapping(
        tile_records, out_dir, uav_tif, tile_size, overlap
    )

    # Write README
    readme = write_roboflow_readme(out_dir, len(tile_records))

    log.info(
        f"\n{'='*60}\n"
        f"ROBOFLOW PACKAGE READY\n"
        f"  Tiles        : {len(tile_records)}\n"
        f"  Images dir   : {out_dir / 'images'}\n"
        f"  Labels dir   : {out_dir / 'labels'} (empty placeholders)\n"
        f"  Coord map    : {coord_json}\n"
        f"  Instructions : {readme}\n"
        f"\nNext step: upload images/ to Roboflow, annotate, export YOLO v8,\n"
        f"place export in data/roboflow_export/, then run:\n"
        f"  python main.py --step import_roboflow\n"
        f"{'='*60}"
    )

    return {
        "n_tiles":      len(tile_records),
        "images_dir":   str(out_dir / "images"),
        "labels_dir":   str(out_dir / "labels"),
        "coord_json":   str(coord_json),
        "readme":       str(readme),
        "output_dir":   str(out_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tile UAV orthomosaic for Roboflow annotation."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_uav_tiler_roboflow(cfg)