"""
train_yolo.py - YOLOv8 Equipment Detection Training
=====================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Trains a YOLOv8 model to detect mining equipment (excavators, trucks,
water pumps, settling ponds, pits) in high-resolution UAV imagery.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

import numpy as np
import yaml

from scripts.utils import get_logger, set_seeds, timer, get_device

logger = get_logger(__name__)

# Equipment class definitions
EQUIPMENT_CLASSES = ["excavator", "truck", "water_pump", "settling_pond", "pit"]


# =============================================================================
# Dataset YAML generation
# =============================================================================

def create_yolo_dataset_yaml(
    tiles_dir: str,
    output_path: str = "data/yolo_dataset.yaml",
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> str:
    """
    Generate a YOLO-format dataset YAML file from UAV tiles.

    Splits tile images into train/val/test sets and writes the YAML
    pointing to each split's image directory.

    Args:
        tiles_dir: Directory with 'images/' and 'labels/' subdirectories
                   (output from tiler.tile_uav_for_yolo).
        output_path: Where to write the dataset YAML.
        train_fraction: Fraction of images for training.
        val_fraction: Fraction for validation.
        seed: Random seed.

    Returns:
        Path to the generated YAML file.
    """
    images_dir = os.path.join(tiles_dir, "images")
    labels_dir = os.path.join(tiles_dir, "labels")

    if not os.path.exists(images_dir):
        raise FileNotFoundError(
            f"YOLO images directory not found: {images_dir}\n"
            "Run the tiling step first (python main.py --step preprocess)."
        )

    # List all image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"YOLO dataset: {len(image_files)} images found.")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(image_files))
    rng.shuffle(indices)

    n_train = int(len(image_files) * train_fraction)
    n_val = int(len(image_files) * val_fraction)

    train_files = [image_files[i] for i in indices[:n_train]]
    val_files = [image_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [image_files[i] for i in indices[n_train + n_val:]]

    # Create split directories
    splits_base = os.path.join(tiles_dir, "splits")
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_img_dir = os.path.join(splits_base, split_name, "images")
        split_lbl_dir = os.path.join(splits_base, split_name, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for fname in files:
            src_img = os.path.join(images_dir, fname)
            dst_img = os.path.join(split_img_dir, fname)
            if not os.path.exists(dst_img):
                shutil.copy2(src_img, dst_img)

            # Copy label (empty if not annotated)
            lbl_name = os.path.splitext(fname)[0] + ".txt"
            src_lbl = os.path.join(labels_dir, lbl_name)
            dst_lbl = os.path.join(split_lbl_dir, lbl_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, "w").close()

    # Write YAML
    abs_splits = os.path.abspath(splits_base)
    dataset_yaml = {
        "path": abs_splits,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(EQUIPMENT_CLASSES),
        "names": EQUIPMENT_CLASSES,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)

    logger.info(
        f"YOLO dataset YAML: {output_path} | "
        f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
    )
    return output_path


# =============================================================================
# Annotation helper
# =============================================================================

def generate_pseudo_annotations(
    tiles_dir: str,
    gt_df: Any,
    uav_meta: Dict[str, Any],
    tile_coords: Any,
    tile_ids: Any,
) -> int:
    """
    Generate YOLO-format pseudo-annotations from GPS ground-truth.

    For each GPS mining site that falls within a UAV tile, creates a
    YOLO label file with a bounding box annotation.

    Note: These are approximate annotations (point → small box).
    For production use, manual annotation with a tool like LabelImg
    or Roboflow is strongly recommended.

    Args:
        tiles_dir: UAV tiles directory.
        gt_df: Ground-truth DataFrame with lat/lon columns.
        uav_meta: UAV metadata with transform and CRS.
        tile_coords: List of tile coordinate dicts.
        tile_ids: List of tile ID strings.

    Returns:
        Number of annotations generated.
    """
    from pyproj import Transformer
    from rasterio.transform import rowcol

    labels_dir = os.path.join(tiles_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    transform = uav_meta.get("transform")
    crs = uav_meta.get("crs")

    if transform is None or crs is None or gt_df is None or len(gt_df) == 0:
        logger.warning("Cannot generate pseudo-annotations: missing metadata or ground-truth.")
        return 0

    proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    n_annotations = 0

    for tile_id, coord in zip(tile_ids, tile_coords):
        lbl_path = os.path.join(labels_dir, f"{tile_id}.txt")
        annotations = []

        tile_rs = coord["row_start"]
        tile_cs = coord["col_start"]
        tile_re = coord["row_end"]
        tile_ce = coord["col_end"]
        tile_h = tile_re - tile_rs
        tile_w = tile_ce - tile_cs

        for _, row in gt_df.iterrows():
            x_proj, y_proj = proj.transform(row["longitude"], row["latitude"])
            r, c = rowcol(transform, x_proj, y_proj)

            # Check if point falls in this tile
            if tile_rs <= r < tile_re and tile_cs <= c < tile_ce:
                # Relative position within tile [0, 1]
                cx = (c - tile_cs) / tile_w
                cy = (r - tile_rs) / tile_h
                # Box size: 10% of tile (approximate)
                bw = 0.10
                bh = 0.10

                # Default class: excavator (0) — user should update
                class_id = 0
                annotations.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_annotations += 1

        with open(lbl_path, "w") as f:
            f.write("\n".join(annotations))

    logger.info(f"Generated {n_annotations} pseudo-annotations from GPS points.")
    return n_annotations


# =============================================================================
# Training
# =============================================================================

@timer
def train_yolo(
    uav_tiling_results: Dict[str, Any],
    config: Dict[str, Any],
    gt_df: Any = None,
    uav_meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Full YOLOv8 training pipeline for equipment detection.

    Annotation priority
    -------------------
    1. Roboflow-imported annotations in data/tiles/uav/labels/
       (produced by import_roboflow_annotations.py — highest quality)
    2. GPS-derived pseudo-annotations (approximate point → small box)
    3. No annotations → training skipped with guidance message

    Args:
        uav_tiling_results: Output from tiler.tile_uav_for_yolo().
        config: Config dict from config.yaml.
        gt_df: Optional ground-truth DataFrame for pseudo-annotation.
        uav_meta: UAV metadata for coordinate projection.

    Returns:
        Path to best YOLOv8 weights file, or empty string if skipped.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "YOLOv8 requires the ultralytics package.\n"
            "Install with: pip install ultralytics"
        )

    cfg = config["training_yolo"]
    paths = config["paths"]
    seed = config.get("random_seed", 42)
    set_seeds(seed)

    tiles_dir  = uav_tiling_results["output_dir"]
    labels_dir = os.path.join(tiles_dir, "labels")

    # ------------------------------------------------------------------
    # Annotation source priority check
    # ------------------------------------------------------------------
    rb_export_dir = Path(
        config.get("roboflow", {}).get("export_dir", "data/roboflow_export")
    ) / "labels"

    # Priority 1 — Roboflow-imported annotations already in pipeline dir
    n_labelled = 0
    if os.path.isdir(labels_dir):
        n_labelled = sum(
            1 for f in os.listdir(labels_dir)
            if f.endswith(".txt") and
            os.path.getsize(os.path.join(labels_dir, f)) > 0
        )

    if n_labelled > 0:
        logger.info(
            f"Using imported Roboflow annotations: "
            f"{n_labelled} annotated tiles in {labels_dir}"
        )

    # Priority 2 — Roboflow export present but not yet imported
    elif rb_export_dir.exists() and any(rb_export_dir.glob("*.txt")):
        logger.error(
            "Roboflow export detected in data/roboflow_export/labels/ "
            "but annotations have not been imported yet.\n"
            "Run: python main.py --step import_roboflow\n"
            "Then re-run: python main.py --step train_yolo"
        )
        return ""

    # Priority 3 — GPS pseudo-annotations
    elif gt_df is not None and len(gt_df) > 0:
        logger.info(
            "No Roboflow annotations found. "
            "Generating pseudo-annotations from GPS ground-truth..."
        )
        os.makedirs(labels_dir, exist_ok=True)
        generate_pseudo_annotations(
            tiles_dir=tiles_dir,
            gt_df=gt_df,
            uav_meta=uav_meta or {},
            tile_coords=uav_tiling_results["coords"],
            tile_ids=uav_tiling_results["tile_ids"],
        )
        n_labelled = sum(
            1 for f in os.listdir(labels_dir)
            if f.endswith(".txt") and
            os.path.getsize(os.path.join(labels_dir, f)) > 0
        )

    total = uav_tiling_results["num_tiles"]
    logger.info(
        f"YOLO training setup: {total} tiles total, "
        f"{n_labelled} annotated ({100*n_labelled/max(total,1):.1f}%)"
    )

    if n_labelled == 0:
        logger.warning(
            "No YOLO annotations found. Equipment detection training skipped.\n"
            "\nTo annotate equipment:\n"
            "  Option A (Roboflow — recommended):\n"
            "    python main.py --step prep_roboflow\n"
            "    → Upload data/roboflow_upload/images/ to https://roboflow.com\n"
            "    → Annotate, export YOLO v8, place in data/roboflow_export/\n"
            "    python main.py --step import_roboflow\n"
            "    python main.py --step train_yolo\n"
            "\n  Option B (LabelImg — offline):\n"
            "    pip install labelImg\n"
            f"    labelImg {os.path.join(tiles_dir, 'images')} {labels_dir}\n"
            "    python main.py --step train_yolo"
        )
        return ""

    # Create dataset YAML
    data_yaml = create_yolo_dataset_yaml(
        tiles_dir=tiles_dir,
        output_path=os.path.join(paths["models"], "yolo_dataset.yaml"),
        seed=seed,
    )

    # Load pretrained model — or resume from interrupted training run
    model_name  = cfg.get("model", "yolov8m.pt")
    yolo_run_dir = Path(paths["models"]) / "yolo_mining"
    last_weights = yolo_run_dir / "weights" / "last.pt"

    if last_weights.exists():
        logger.info(
            f"[RESUME] Interrupted YOLO training detected → {last_weights}. "
            "Resuming from last checkpoint."
        )
        model = YOLO(str(last_weights))
        resume_flag = True
    else:
        logger.info(f"Loading YOLOv8 pretrained: {model_name}")
        model = YOLO(model_name)
        resume_flag = False

    # Train
    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Training YOLOv8: epochs={cfg['epochs']}, "
        f"imgsz={cfg['image_size']}, batch={cfg['batch_size']}"
    )

    results = model.train(
        data=data_yaml,
        epochs=cfg["epochs"],
        imgsz=cfg["image_size"],
        batch=cfg["batch_size"],
        patience=cfg.get("early_stopping_patience", 20),
        project=paths["models"],
        name="yolo_mining",
        device=device,
        exist_ok=True,
        resume=resume_flag,
        save=True,
        val=True,
        augment=cfg.get("augment", True),
        lr0=cfg.get("learning_rate", 0.01),
        seed=seed,
        plots=True,
    )

    # Copy best model to standard location
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = os.path.join(paths["models"], "yolo_best.pt")

    if best_src.exists():
        shutil.copy2(str(best_src), best_dst)
        logger.info(f"YOLOv8 best model → {best_dst}")

    # Validate on test set
    logger.info("Running YOLO validation on test split...")
    test_yaml_splits = os.path.join(tiles_dir, "splits")
    val_results = model.val(data=data_yaml, split="test")
    logger.info(f"YOLO test mAP@0.5: {val_results.box.map50:.4f}")

    return best_dst


