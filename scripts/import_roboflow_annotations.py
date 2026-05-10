"""
import_roboflow_annotations.py
================================
Import Roboflow-exported YOLO annotations back into the pipeline after
manual equipment labelling on https://roboflow.com.

Annotation source — checked in this order
------------------------------------------
1. Roboflow API  (if api_key, workspace, project are set in config.yaml)
   Downloads the dataset directly; no manual ZIP handling required.
2. Manual export (if data/roboflow_export/labels/ already contains .txt files)
   Place the Roboflow "YOLOv8" export there and run this step.

Both paths feed into the same validation + import pipeline:

Workflow
--------
1. Download via API  OR  scan data/roboflow_export/labels/ for .txt files
2. Load tile_coordinates.json (produced by uav_tiler_roboflow.py)
3. Verify each annotation file matches a known tile in the coordinate map
4. Validate bounding boxes (in-range, valid class IDs)
5. Copy validated annotations to data/tiles/uav/labels/ (pipeline working dir)
6. Enrich each annotation with real-world centroid coordinates and save a
   georeferenced GeoJSON of all detections (for use in fusion step)
7. Save equipment_annotation_stats.json (class counts, tile coverage)

After this step, run:
    python main.py --step train_yolo

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

import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)

# Valid class IDs and their names (must match config.yaml → roboflow.classes)
DEFAULT_CLASSES = {
    0: "excavator",
    1: "truck",
    2: "water_pump",
    3: "settling_pond",
    4: "pit",
}


# ---------------------------------------------------------------------------
# Step 1 — Load coordinate mapping
# ---------------------------------------------------------------------------

def load_tile_coordinates(coord_json: Path) -> dict[str, dict]:
    """Load tile_coordinates.json and index by tile_id.

    Parameters
    ----------
    coord_json : path to tile_coordinates.json

    Returns
    -------
    Dict mapping tile_id -> tile record dict
    """
    if not coord_json.exists():
        raise FileNotFoundError(
            f"Tile coordinate mapping not found: {coord_json}\n"
            "Run `python main.py --step prep_roboflow` first."
        )
    with open(coord_json) as f:
        data = json.load(f)

    tile_index = {rec["tile_id"]: rec for rec in data.get("tiles", [])}
    log.info(
        f"Loaded coordinate mapping: {len(tile_index)} tiles "
        f"from {coord_json}"
    )
    return tile_index


# ---------------------------------------------------------------------------
# Step 2 — Scan Roboflow export directory
# ---------------------------------------------------------------------------

def scan_export_labels(export_labels_dir: Path) -> list[Path]:
    """Return list of all .txt annotation files in the Roboflow export.

    Parameters
    ----------
    export_labels_dir : data/roboflow_export/labels/

    Returns
    -------
    Sorted list of .txt file paths
    """
    if not export_labels_dir.exists():
        raise FileNotFoundError(
            f"Roboflow export labels directory not found: {export_labels_dir}\n"
            "Download the YOLO v8 export from Roboflow and place it in "
            "data/roboflow_export/."
        )
    label_files = sorted(export_labels_dir.glob("*.txt"))
    log.info(f"Found {len(label_files)} annotation files in {export_labels_dir}")
    return label_files


# ---------------------------------------------------------------------------
# Step 2a — Roboflow API download (alternative to manual export)
# ---------------------------------------------------------------------------

def download_via_api(rb_cfg: dict, export_dir: Path) -> bool:
    """Download the annotated dataset from Roboflow using the Python SDK.

    Called automatically when api_key, workspace, and project are all set
    in config.yaml → roboflow. Skipped silently when any field is blank.

    Parameters
    ----------
    rb_cfg     : the roboflow section of config.yaml
    export_dir : destination directory (data/roboflow_export/)

    Returns
    -------
    True  — download succeeded; labels are now in export_dir/labels/
    False — credentials missing or download failed (caller falls back to manual)
    """
    api_key   = rb_cfg.get("api_key",   "").strip()
    workspace = rb_cfg.get("workspace", "").strip()
    project   = rb_cfg.get("project",   "").strip()
    version   = int(rb_cfg.get("version", 1))

    # If any credential is blank, skip API download silently
    if not all([api_key, workspace, project]):
        log.info(
            "Roboflow API credentials not set in config.yaml "
            "(api_key / workspace / project). "
            "Falling back to manual export in data/roboflow_export/."
        )
        return False

    try:
        from roboflow import Roboflow
    except ImportError:
        log.warning(
            "roboflow package not installed — cannot use API download.\n"
            "Install with: pip install roboflow\n"
            "Falling back to manual export."
        )
        return False

    try:
        log.info(
            f"Connecting to Roboflow API — "
            f"workspace={workspace}, project={project}, version={version}"
        )
        rf      = Roboflow(api_key=api_key)
        proj    = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download(
            "yolov8",
            location=str(export_dir),
            overwrite=True,
        )
        log.info(f"Roboflow API download complete → {export_dir}")

        # The SDK may nest files under a version subdirectory, e.g.
        # data/roboflow_export/<project>-<version>/labels/
        # Normalise so labels are always at export_dir/labels/
        labels_direct = export_dir / "labels"
        if not labels_direct.exists():
            # Search one level down for a labels/ subfolder
            subdirs = [d for d in export_dir.iterdir() if d.is_dir()]
            for sub in subdirs:
                candidate = sub / "labels"
                if candidate.exists():
                    log.info(
                        f"SDK placed labels in subdirectory {sub.name}/ — "
                        f"moving to {labels_direct}"
                    )
                    candidate.rename(labels_direct)
                    # Also move images/ if present
                    candidate_img = sub / "images"
                    if candidate_img.exists():
                        candidate_img.rename(export_dir / "images")
                    break

        if not (export_dir / "labels").exists():
            log.error(
                "API download succeeded but no labels/ folder found. "
                "Check that the Roboflow project has been annotated and "
                "a dataset version has been generated."
            )
            return False

        n_labels = len(list((export_dir / "labels").glob("*.txt")))
        log.info(f"API download: {n_labels} label files available.")
        return True

    except Exception as exc:
        log.error(
            f"Roboflow API download failed: {exc}\n"
            "Falling back to manual export in data/roboflow_export/."
        )
        return False


# ---------------------------------------------------------------------------
# Step 3 — Verify tile correspondence
# ---------------------------------------------------------------------------

def verify_correspondence(
    label_files: list[Path],
    tile_index: dict[str, dict],
) -> tuple[list[Path], list[Path]]:
    """Check which exported labels match known tiles.

    Parameters
    ----------
    label_files : list of exported .txt files
    tile_index  : mapping of tile_id -> tile record

    Returns
    -------
    (matched, unmatched) lists of label file paths
    """
    matched   = []
    unmatched = []

    for lf in label_files:
        tile_id = lf.stem   # e.g. "tile_000042"
        if tile_id in tile_index:
            matched.append(lf)
        else:
            unmatched.append(lf)
            log.warning(f"Tile ID not in coordinate map: {tile_id} ({lf.name})")

    log.info(
        f"Tile correspondence: {len(matched)} matched, "
        f"{len(unmatched)} unmatched"
    )
    if len(unmatched) > 0:
        log.warning(
            f"{len(unmatched)} exported files have no matching tile in "
            "tile_coordinates.json. They will be skipped."
        )
    return matched, unmatched


# ---------------------------------------------------------------------------
# Step 4 — Validate YOLO annotations
# ---------------------------------------------------------------------------

def _parse_yolo_line(line: str) -> tuple[int, float, float, float, float] | None:
    """Parse one YOLO annotation line.

    Format: class_id cx cy w h  (all floats, space-separated)
    Returns None if the line is blank or malformed.
    """
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        return int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None


def validate_annotations(
    label_file: Path,
    valid_class_ids: set[int],
) -> tuple[list[str], list[str]]:
    """Validate all annotations in a single YOLO label file.

    Checks:
    - Bounding box coordinates in [0, 1]
    - Class ID is in valid_class_ids

    Parameters
    ----------
    label_file      : path to .txt label file
    valid_class_ids : set of permitted class IDs

    Returns
    -------
    (valid_lines, issues) — valid_lines are clean annotation strings,
    issues are human-readable problem descriptions
    """
    valid_lines = []
    issues      = []

    raw = label_file.read_text().strip()
    if not raw:
        return [], []   # empty = no annotations = valid (background tile)

    for lineno, line in enumerate(raw.splitlines(), start=1):
        parsed = _parse_yolo_line(line)
        if parsed is None:
            issues.append(f"Line {lineno}: malformed → '{line}'")
            continue

        class_id, cx, cy, w, h = parsed

        if class_id not in valid_class_ids:
            issues.append(
                f"Line {lineno}: invalid class_id={class_id} "
                f"(valid: {sorted(valid_class_ids)})"
            )
            continue

        coords_ok = all(0.0 <= v <= 1.0 for v in (cx, cy, w, h))
        if not coords_ok:
            issues.append(
                f"Line {lineno}: bbox out of range "
                f"cx={cx:.4f} cy={cy:.4f} w={w:.4f} h={h:.4f}"
            )
            continue

        # Clip to [0, 1] just in case of floating-point rounding
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        valid_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return valid_lines, issues


# ---------------------------------------------------------------------------
# Step 5 — Copy to pipeline working directory
# ---------------------------------------------------------------------------

def copy_to_pipeline(
    matched_files: list[Path],
    tile_index: dict[str, dict],
    valid_class_ids: set[int],
    dest_labels_dir: Path,
) -> dict[str, Any]:
    """Validate and copy annotations to data/tiles/uav/labels/.

    Parameters
    ----------
    matched_files    : verified label files from Roboflow export
    tile_index       : tile ID -> tile record mapping
    valid_class_ids  : set of valid YOLO class IDs
    dest_labels_dir  : pipeline destination (data/tiles/uav/labels/)

    Returns
    -------
    Dict with copy statistics
    """
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    n_copied   = 0
    n_empty    = 0
    n_invalid  = 0
    all_issues = []

    for lf in matched_files:
        valid_lines, issues = validate_annotations(lf, valid_class_ids)

        if issues:
            all_issues.extend([f"{lf.name}: {iss}" for iss in issues])
            n_invalid += 1
            if not valid_lines:
                log.warning(
                    f"All annotations invalid in {lf.name} — "
                    "writing empty label."
                )

        dest = dest_labels_dir / lf.name
        dest.write_text("\n".join(valid_lines) + ("\n" if valid_lines else ""))

        if valid_lines:
            n_copied += 1
        else:
            n_empty += 1

    log.info(
        f"Copied to pipeline: {n_copied} annotated, "
        f"{n_empty} background, {n_invalid} had validation issues"
    )
    if all_issues:
        log.warning(f"Validation issues ({len(all_issues)}):")
        for iss in all_issues[:20]:
            log.warning(f"  {iss}")
        if len(all_issues) > 20:
            log.warning(f"  … and {len(all_issues)-20} more (see logs)")

    return {
        "n_annotated_tiles": n_copied,
        "n_background_tiles": n_empty,
        "n_validation_issues": n_invalid,
        "validation_messages": all_issues,
    }


# ---------------------------------------------------------------------------
# Step 6 — Georeferenced GeoJSON of all detections
# ---------------------------------------------------------------------------

def _yolo_to_lonlat(
    cx: float, cy: float,
    tile_rec: dict,
) -> tuple[float, float]:
    """Convert YOLO normalised centre coordinates to WGS84 lon/lat.

    Parameters
    ----------
    cx, cy   : YOLO normalised centre (0–1) relative to the padded tile
    tile_rec : tile record from tile_coordinates.json

    Returns
    -------
    (lon, lat) in WGS84
    """
    geo = tile_rec["georeferencing"]
    transform_params = geo["pixel_to_geo_transform"]  # [a, b, c, d, e, f]
    a, b, c, d, e, f = transform_params

    tile_w = tile_rec["image_width"]
    tile_h = tile_rec["image_height"]

    # Pixel coordinate of bbox centre (within tile, 0-based)
    px = cx * tile_w
    py = cy * tile_h

    # Apply affine: x = c + px*a + py*b;  y = f + px*d + py*e
    x_src = c + px * a + py * b
    y_src = f + px * d + py * e

    # Convert to WGS84 if needed
    epsg = geo.get("epsg")
    if epsg and epsg != 4326:
        transformer = Transformer.from_crs(
            f"EPSG:{epsg}", "EPSG:4326", always_xy=True
        )
        lon, lat = transformer.transform(x_src, y_src)
    else:
        lon, lat = x_src, y_src

    return round(lon, 8), round(lat, 8)


def build_georeferenced_geojson(
    matched_files: list[Path],
    tile_index: dict[str, dict],
    valid_class_ids: set[int],
    class_names: dict[int, str],
    dest_labels_dir: Path,
) -> dict[str, Any]:
    """Build a GeoJSON FeatureCollection of all annotated equipment.

    Each feature is a Point at the WGS84 centroid of the bounding box,
    with properties: tile_id, class_id, class_name, confidence (null),
    bbox_norm (cx, cy, w, h).

    Parameters
    ----------
    matched_files    : validated label files
    tile_index       : tile ID -> tile record
    valid_class_ids  : set of valid class IDs
    class_names      : dict mapping class_id -> name
    dest_labels_dir  : directory where validated labels were saved

    Returns
    -------
    GeoJSON FeatureCollection dict
    """
    features = []

    for lf in matched_files:
        tile_id  = lf.stem
        tile_rec = tile_index[tile_id]

        # Read validated labels from dest_labels_dir (already cleaned)
        dest_lf = dest_labels_dir / lf.name
        if not dest_lf.exists():
            continue

        raw = dest_lf.read_text().strip()
        if not raw:
            continue

        for line in raw.splitlines():
            parsed = _parse_yolo_line(line)
            if parsed is None:
                continue
            class_id, cx, cy, w, h = parsed
            if class_id not in valid_class_ids:
                continue

            lon, lat = _yolo_to_lonlat(cx, cy, tile_rec)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat],
                },
                "properties": {
                    "tile_id":    tile_id,
                    "class_id":   class_id,
                    "class_name": class_names.get(class_id, f"class_{class_id}"),
                    "confidence": None,   # set by YOLO inference, not annotation
                    "bbox_cx":    round(cx, 6),
                    "bbox_cy":    round(cy, 6),
                    "bbox_w":     round(w, 6),
                    "bbox_h":     round(h, 6),
                    "source":     "roboflow_annotation",
                },
            }
            features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    log.info(f"Built georeferenced GeoJSON: {len(features)} equipment instances")
    return geojson


# ---------------------------------------------------------------------------
# Step 7 — Equipment annotation statistics
# ---------------------------------------------------------------------------

def compute_annotation_stats(
    matched_files: list[Path],
    dest_labels_dir: Path,
    class_names: dict[int, str],
) -> dict[str, Any]:
    """Count annotated instances per equipment class.

    Parameters
    ----------
    matched_files    : verified label files
    dest_labels_dir  : directory with validated labels
    class_names      : dict mapping class_id -> name

    Returns
    -------
    Stats dict suitable for JSON serialisation
    """
    class_counts: dict[str, int] = {name: 0 for name in class_names.values()}
    n_annotated_tiles  = 0
    n_background_tiles = 0
    total_instances    = 0

    for lf in matched_files:
        dest_lf = dest_labels_dir / lf.name
        if not dest_lf.exists():
            continue

        raw = dest_lf.read_text().strip()
        if not raw:
            n_background_tiles += 1
            continue

        n_annotated_tiles += 1
        for line in raw.splitlines():
            parsed = _parse_yolo_line(line)
            if parsed is None:
                continue
            class_id = parsed[0]
            name = class_names.get(class_id, f"class_{class_id}")
            class_counts[name] = class_counts.get(name, 0) + 1
            total_instances += 1

    stats = {
        "total_instances":      total_instances,
        "annotated_tiles":      n_annotated_tiles,
        "background_tiles":     n_background_tiles,
        "instances_per_class":  class_counts,
        "annotation_rate_pct":  round(
            100 * n_annotated_tiles / max(1, n_annotated_tiles + n_background_tiles), 2
        ),
    }
    log.info(
        f"Annotation stats: {total_instances} instances across "
        f"{n_annotated_tiles} tiles | "
        + ", ".join(f"{k}={v}" for k, v in class_counts.items() if v > 0)
    )
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@timer
def run_import_roboflow(config: dict) -> dict[str, Any]:
    """
    Full Roboflow annotation import pipeline.

    Annotation source priority
    --------------------------
    1. Roboflow API  — if api_key, workspace, project set in config.yaml
    2. Manual export — labels already in data/roboflow_export/labels/
    3. Error         — neither source available

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    Dict with import statistics and output paths
    """
    rb_cfg     = config.get("roboflow", {})
    upload_dir = Path(rb_cfg.get("output_dir", "data/roboflow_upload"))
    export_dir = Path(rb_cfg.get("export_dir", "data/roboflow_export"))
    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    # Build class map from config (fallback to defaults)
    classes_cfg = rb_cfg.get("classes", {})
    if classes_cfg:
        class_names = {int(v): k for k, v in classes_cfg.items()}
    else:
        class_names = DEFAULT_CLASSES
    valid_class_ids = set(class_names.keys())

    dest_labels_dir = Path(config["paths"]["tiles"]) / "uav" / "labels"

    # ------------------------------------------------------------------ #
    # Step 1 — Load coordinate mapping
    # ------------------------------------------------------------------ #
    coord_json = upload_dir / "tile_coordinates.json"
    tile_index = load_tile_coordinates(coord_json)

    # ------------------------------------------------------------------ #
    # Step 2 — Obtain export labels: API first, manual fallback
    # ------------------------------------------------------------------ #
    export_labels_dir = export_dir / "labels"

    api_used = download_via_api(rb_cfg, export_dir)

    if api_used:
        log.info("Using Roboflow API download as annotation source.")
    else:
        manual_exists = (
            export_labels_dir.exists()
            and any(export_labels_dir.glob("*.txt"))
        )
        if manual_exists:
            log.info(
                f"Using manual Roboflow export from {export_labels_dir}."
            )
        else:
            raise FileNotFoundError(
                "No annotations found. Choose one of:\n\n"
                "  Option A — Roboflow API (fill in config.yaml → roboflow):\n"
                "    api_key:   YOUR_KEY\n"
                "    workspace: YOUR_WORKSPACE_SLUG\n"
                "    project:   YOUR_PROJECT_NAME\n"
                "    version:   1\n"
                "  Then re-run: python main.py --step import_roboflow\n\n"
                "  Option B — Manual download:\n"
                "    1. Roboflow → your project → Generate Dataset\n"
                "    2. Export → YOLOv8 format → download ZIP\n"
                "    3. Extract and place the labels/ folder at:\n"
                f"       {export_labels_dir}\n"
                "    4. Re-run: python main.py --step import_roboflow"
            )

    label_files = scan_export_labels(export_labels_dir)

    # ------------------------------------------------------------------ #
    # Step 3 — Verify correspondence
    # ------------------------------------------------------------------ #
    matched, unmatched = verify_correspondence(label_files, tile_index)
    if not matched:
        raise RuntimeError(
            "No exported annotation files match any known tile. "
            "Check that tile IDs in the Roboflow export match "
            f"those in {coord_json}."
        )

    # ------------------------------------------------------------------ #
    # Steps 4 & 5 — Validate + copy to pipeline working dir
    # ------------------------------------------------------------------ #
    copy_stats = copy_to_pipeline(
        matched, tile_index, valid_class_ids, dest_labels_dir
    )

    # ------------------------------------------------------------------ #
    # Step 6 — Georeferenced GeoJSON
    # ------------------------------------------------------------------ #
    geojson = build_georeferenced_geojson(
        matched, tile_index, valid_class_ids, class_names, dest_labels_dir
    )
    geojson_path = output_dir / "annotated_equipment.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)
    log.info(f"Georeferenced equipment GeoJSON: {geojson_path}")

    # ------------------------------------------------------------------ #
    # Step 7 — Annotation statistics
    # ------------------------------------------------------------------ #
    stats = compute_annotation_stats(matched, dest_labels_dir, class_names)
    stats_path = output_dir / "equipment_annotation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Equipment annotation stats: {stats_path}")

    result = {
        **copy_stats,
        **stats,
        "unmatched_files":    len(unmatched),
        "geojson_path":       str(geojson_path),
        "stats_path":         str(stats_path),
        "dest_labels_dir":    str(dest_labels_dir),
    }

    n_inst = stats["total_instances"]
    n_ann  = stats["annotated_tiles"]
    log.info(
        f"\n{'='*60}\n"
        f"ROBOFLOW IMPORT COMPLETE\n"
        f"  Equipment instances : {n_inst}\n"
        f"  Annotated tiles    : {n_ann}\n"
        f"  Background tiles   : {stats['background_tiles']}\n"
        f"  Validation issues  : {copy_stats['n_validation_issues']}\n"
        f"  GeoJSON            : {geojson_path}\n"
        f"  Stats              : {stats_path}\n"
        f"\nNext step: python main.py --step train_yolo\n"
        f"{'='*60}"
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Import Roboflow-exported YOLO annotations into the pipeline."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_import_roboflow(cfg)