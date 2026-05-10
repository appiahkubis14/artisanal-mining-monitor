"""
inference_uav.py - UAV Equipment Detection Inference
======================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Runs YOLOv8 on the full-resolution UAV orthomosaic in tiles,
detects mining equipment, and exports detections to GeoJSON.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import geopandas as gpd
from shapely.geometry import box as shapely_box
from rasterio.transform import xy as rasterio_xy

from scripts.tiler import tile_image
from scripts.utils import get_logger, timer, pixel_to_latlon

logger = get_logger(__name__)

EQUIPMENT_CLASSES = ["excavator", "truck", "water_pump", "settling_pond", "pit"]
# Colours for visualisation
CLASS_COLORS = {
    "excavator":     (255, 50, 50),
    "truck":         (50, 255, 50),
    "water_pump":    (50, 100, 255),
    "settling_pond": (255, 200, 0),
    "pit":           (180, 0, 255),
}


@timer
def run_uav_inference(
    uav_data: Dict[str, Any],
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Run YOLOv8 equipment detection on the UAV orthomosaic.

    Process:
      1. Tile the full-resolution UAV array into 640×640 tiles.
      2. Run YOLO inference on each tile.
      3. Project bounding boxes back to geographic coordinates.
      4. Apply Non-Maximum Suppression across tile boundaries.
      5. Export to GeoJSON.

    Args:
        uav_data: Preprocessed UAV dict (from preprocessor.preprocess_uav).
                  Must contain 'data_highres' and 'meta_highres'.
        config: Config dict.
        model_path: Path to best YOLO weights (data/models/yolo_best.pt).
        conf_threshold: Minimum confidence for a detection.

    Returns:
        Dict with:
            - 'detections': list of detection dicts
            - 'geojson_path': path to GeoJSON output
            - 'summary': equipment counts by class
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required. Install with: pip install ultralytics"
        )

    paths = config["paths"]
    infer_cfg = config["inference"]
    os.makedirs(paths["outputs"], exist_ok=True)

    if model_path is None:
        model_path = os.path.join(paths["models"], "yolo_best.pt")

    if not os.path.exists(model_path):
        logger.warning(
            f"YOLO model not found at {model_path}. "
            "Inference skipped. Train YOLO first."
        )
        return {"detections": [], "geojson_path": None, "summary": {}}

    # Load YOLO model
    model = YOLO(model_path)
    logger.info(f"YOLOv8 loaded: {model_path}")

    uav_highres = uav_data["data_highres"]   # [B, H, W] float32 0–1
    uav_meta = uav_data["meta_highres"]
    transform = uav_meta["transform"]
    crs = uav_meta["crs"]

    # Tile the UAV image
    tile_size = config["training_yolo"].get("image_size", 640)
    tiles, coords = tile_image(
        uav_highres, tile_size=tile_size, overlap=0.15,
        min_valid_fraction=0.01, max_cloud_fraction=1.0,
    )

    logger.info(f"UAV inference: {len(tiles)} tiles at {tile_size}×{tile_size}px")

    all_detections = []

    for tile_idx, (tile, coord) in enumerate(zip(tiles, coords)):
        # Convert tile to BGR uint8 for YOLO
        n_bands = min(3, tile.shape[0])
        rgb = (tile[:n_bands].transpose(1, 2, 0) * 255).astype(np.uint8)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Run YOLO
        results = model.predict(
            source=bgr,
            conf=conf_threshold,
            iou=0.45,
            imgsz=tile_size,
            verbose=False,
            save=False,
        )

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]

                # Pixel coords within tile → pixel coords in full image
                tile_rs = coord["row_start"]
                tile_cs = coord["col_start"]
                global_x1 = tile_cs + x1
                global_y1 = tile_rs + y1
                global_x2 = tile_cs + x2
                global_y2 = tile_rs + y2

                # Convert to geographic coordinates
                try:
                    lon1, lat1 = _pixel_to_lonlat(global_x1, global_y1, transform, crs)
                    lon2, lat2 = _pixel_to_lonlat(global_x2, global_y2, transform, crs)
                except Exception:
                    continue

                class_name = (EQUIPMENT_CLASSES[cls_id]
                              if cls_id < len(EQUIPMENT_CLASSES) else f"class_{cls_id}")

                all_detections.append({
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(conf, 4),
                    "bbox_pixel": [global_x1, global_y1, global_x2, global_y2],
                    "bbox_geo": [
                        min(lon1, lon2), min(lat1, lat2),
                        max(lon1, lon2), max(lat1, lat2)
                    ],
                    "centroid": [
                        (lon1 + lon2) / 2, (lat1 + lat2) / 2
                    ],
                })

        if (tile_idx + 1) % 50 == 0:
            logger.debug(f"UAV inference: {tile_idx+1}/{len(tiles)} tiles")

    # Cross-tile NMS (remove duplicate detections near tile boundaries)
    all_detections = _cross_tile_nms(all_detections, iou_threshold=0.45)

    logger.info(f"UAV detections after NMS: {len(all_detections)}")

    # --- Export to GeoJSON ---
    geojson_path = _save_detections_geojson(all_detections, paths["outputs"])

    # --- Summary ---
    summary = {}
    for det in all_detections:
        cls = det["class_name"]
        summary[cls] = summary.get(cls, 0) + 1

    logger.info(f"Equipment detected: {summary}")
    logger.info(f"Detections saved → {geojson_path}")

    return {
        "detections": all_detections,
        "geojson_path": geojson_path,
        "summary": summary,
        "n_detections": len(all_detections),
    }


# =============================================================================
# Helpers
# =============================================================================

def _pixel_to_lonlat(
    col: float,
    row: float,
    transform: Any,
    crs: Any,
) -> Tuple[float, float]:
    """
    Convert raster pixel (col, row) to geographic lon/lat.

    Args:
        col: Column (x) in the raster.
        row: Row (y) in the raster.
        transform: Affine transform.
        crs: Raster CRS.

    Returns:
        (longitude, latitude) in WGS84.
    """
    from pyproj import Transformer

    x, y = rasterio_xy(transform, row, col)

    if str(crs) != "EPSG:4326":
        proj = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = proj.transform(x, y)
    else:
        lon, lat = x, y

    return float(lon), float(lat)


def _cross_tile_nms(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """
    Apply Non-Maximum Suppression across tile boundaries.

    Removes duplicate detections that span tile overlap regions.

    Args:
        detections: List of detection dicts with 'bbox_geo' and 'confidence'.
        iou_threshold: IoU above which the lower-confidence box is suppressed.

    Returns:
        Filtered list of detections.
    """
    if not detections:
        return []

    # Sort by confidence descending
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    from scripts.utils import compute_iou

    kept = []
    suppressed = set()

    for i, det in enumerate(detections):
        if i in suppressed:
            continue
        kept.append(det)
        box_i = tuple(det["bbox_geo"])

        for j in range(i + 1, len(detections)):
            if j in suppressed:
                continue
            # Only suppress same class
            if detections[j]["class_name"] != det["class_name"]:
                continue
            box_j = tuple(detections[j]["bbox_geo"])
            if compute_iou(box_i, box_j) > iou_threshold:
                suppressed.add(j)

    return kept


def _save_detections_geojson(
    detections: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Save detections to a GeoJSON file with bounding box geometries.

    Args:
        detections: List of detection dicts.
        output_dir: Output directory.

    Returns:
        Path to saved GeoJSON.
    """
    features = []
    for i, det in enumerate(detections):
        lon1, lat1, lon2, lat2 = det["bbox_geo"]
        geom = shapely_box(lon1, lat1, lon2, lat2).__geo_interface__

        features.append({
            "type": "Feature",
            "id": i,
            "geometry": geom,
            "properties": {
                "detection_id": i,
                "class_name": det["class_name"],
                "class_id": det["class_id"],
                "confidence": det["confidence"],
                "centroid_lon": det["centroid"][0],
                "centroid_lat": det["centroid"][1],
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "total_detections": len(detections),
            "classes": list({d["class_name"] for d in detections}),
        },
    }

    out_path = os.path.join(output_dir, "equipment_detections.geojson")
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)

    return out_path
