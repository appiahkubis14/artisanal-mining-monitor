"""
inference_satellite.py - Satellite U-Net Inference
====================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Runs the trained U-Net on new satellite imagery to produce:
  - Probability map (GeoTIFF, 0–1)
  - Binary detection maps at 3 confidence thresholds
  - Optional multi-date change detection inputs
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scripts.models import build_unet
from scripts.tiler import tile_image, reconstruct_from_tiles
from scripts.utils import (
    get_logger, get_device, load_checkpoint,
    numpy_to_geotiff, timer,
)

logger = get_logger(__name__)


@timer
def run_satellite_inference(
    feature_stack: np.ndarray,
    meta: Dict[str, Any],
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    date_label: str = "latest",
) -> Dict[str, Any]:
    """
    Run U-Net inference on a feature stack and produce probability maps.

    Checkpoint/resume: if prob_map_{date_label}.tif already exists in
    outputs/ the function reloads it from disk and skips inference entirely.

    Args:
        feature_stack: [C, H, W] feature array (from feature_engineering.py).
        meta: Rasterio metadata for the feature stack.
        config: Config dict.
        model_path: Path to model weights (default: data/models/unet_best.pth).
        date_label: Label for this date (used in output filenames).

    Returns:
        Dict with:
            - 'prob_map': np.ndarray [H, W] float32, probabilities [0, 1]
            - 'binary_maps': dict of threshold → binary mask [H, W]
            - 'output_paths': dict of threshold → saved GeoTIFF paths
    """
    paths     = config["paths"]
    infer_cfg = config["inference"]
    os.makedirs(paths["outputs"], exist_ok=True)

    prob_path = os.path.join(paths["outputs"], f"prob_map_{date_label}.tif")

    # --- Checkpoint: reload probability map if already computed ---
    if os.path.exists(prob_path):
        logger.info(
            f"[RESUME] Probability map already exists for {date_label} "
            f"→ {prob_path}. Skipping inference."
        )
        import rasterio as _rio
        with _rio.open(prob_path) as src:
            prob_map = src.read(1).astype(np.float32)

        thresholds   = infer_cfg.get("thresholds", {"low": 0.3, "medium": 0.5, "high": 0.7})
        binary_maps  = {}
        output_paths = {"prob": prob_path}
        for level, thresh in thresholds.items():
            bin_path = os.path.join(paths["outputs"], f"binary_{level}_{date_label}.tif")
            if os.path.exists(bin_path):
                with _rio.open(bin_path) as src:
                    binary_maps[level] = src.read(1)
            else:
                binary_maps[level] = (prob_map >= thresh).astype(np.uint8)
            output_paths[level] = bin_path
        return {
            "prob_map":    prob_map,
            "binary_maps": binary_maps,
            "output_paths": output_paths,
            "meta":        meta,
            "date_label":  date_label,
        }

    if model_path is None:
        model_path = os.path.join(paths["models"], "unet_best.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Run training first: python main.py --step train_unet"
        )

    device = get_device(config.get("device", "cuda"))

    # --- Load model ---
    n_channels = feature_stack.shape[0]
    model = build_unet(
        architecture=config["training_unet"].get("architecture", "unet"),
        in_channels=n_channels,
        pretrained=False,
    )
    ckpt = load_checkpoint(model_path, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    # --- Tile feature stack ---
    tile_cfg  = config["tiling"]
    tile_size = tile_cfg["satellite_tile_size"]
    overlap   = tile_cfg["overlap"]

    tiles, coords = tile_image(
        feature_stack, tile_size, overlap,
        min_valid_fraction=0.01,
        max_cloud_fraction=1.0,
    )

    logger.info(f"Inference: {len(tiles)} tiles, shape={feature_stack.shape}")

    # --- Batch inference ---
    batch_size = infer_cfg.get("batch_size", 32)
    all_preds  = []

    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch        = tiles[i:i + batch_size]
            batch_tensor = torch.from_numpy(np.stack(batch, axis=0)).to(device)

            logits = model(batch_tensor)               # [B, 1, H, W]
            probs  = torch.sigmoid(logits).squeeze(1)  # [B, H, W]
            all_preds.extend(probs.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                logger.debug(f"Inference: {i + len(batch)}/{len(tiles)} tiles")

    # --- Reconstruct probability map ---
    H, W    = feature_stack.shape[1], feature_stack.shape[2]
    prob_map = reconstruct_from_tiles(all_preds, coords, (H, W), tile_size)

    logger.info(
        f"Probability map: min={prob_map.min():.3f}, max={prob_map.max():.3f}, "
        f"mean={prob_map.mean():.3f}"
    )

    # --- Save probability map ---
    numpy_to_geotiff(prob_map, prob_path, meta, dtype="float32")

    # --- Binary maps at multiple thresholds ---
    thresholds  = infer_cfg.get("thresholds", {"low": 0.3, "medium": 0.5, "high": 0.7})
    binary_maps = {}
    output_paths = {"prob": prob_path}

    for level, thresh in thresholds.items():
        binary   = (prob_map >= thresh).astype(np.uint8)
        binary_maps[level] = binary
        bin_path = os.path.join(paths["outputs"], f"binary_{level}_{date_label}.tif")
        numpy_to_geotiff(binary, bin_path, meta, dtype="uint8")
        output_paths[level] = bin_path
        n_pixels = binary.sum()
        area_ha  = n_pixels * (10 * 10) / 10000
        logger.info(
            f"Threshold {level} ({thresh}): {n_pixels} pixels detected "
            f"≈ {area_ha:.1f} ha"
        )

    logger.info(f"Inference outputs saved → {paths['outputs']}")

    return {
        "prob_map":    prob_map,
        "binary_maps": binary_maps,
        "output_paths": output_paths,
        "meta":        meta,
        "date_label":  date_label,
    }


def run_multi_date_inference(
    feature_stacks_by_date: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Run inference on multiple dates for change detection.

    Each date is checkpointed individually — if a probability map already
    exists for a date it is reloaded from disk and that date is skipped.

    Args:
        feature_stacks_by_date: Dict mapping date strings to feature arrays.
        meta: Shared rasterio metadata.
        config: Config dict.

    Returns:
        Dict mapping date strings to inference results.
    """
    results = {}
    total   = len(feature_stacks_by_date)
    for idx, (date_label, feature_stack) in enumerate(feature_stacks_by_date.items(), 1):
        logger.info(f"Inference [{idx}/{total}]: {date_label}")
        results[date_label] = run_satellite_inference(
            feature_stack, meta, config, date_label=date_label
        )
    return results