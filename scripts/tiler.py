"""
tiler.py - Image Tiling and Reconstruction Module
===================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Splits large rasters into overlapping patches for training/inference,
and reconstructs full probability maps from tiled predictions using
weighted average blending in overlap regions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from scripts.utils import get_logger, save_json, load_json

logger = get_logger(__name__)


# =============================================================================
# Tiling
# =============================================================================

def tile_image(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.20,
    min_valid_fraction: float = 0.10,
    max_cloud_fraction: float = 0.90,
) -> Tuple[List[np.ndarray], List[Dict[str, int]]]:
    """
    Split a 2D or 3D image into overlapping square tiles.

    Args:
        image: Array of shape [bands, H, W] or [H, W].
        tile_size: Tile edge length in pixels.
        overlap: Fractional overlap between adjacent tiles (0–1).
        min_valid_fraction: Skip tiles where fewer than this fraction of
                            pixels are finite (non-NaN, non-zero).
        max_cloud_fraction: Skip tiles where more than this fraction of
                            pixels are NaN (cloud-masked).

    Returns:
        Tuple of:
            - tiles: List of arrays, each shape [bands, tile_size, tile_size].
            - coords: List of dicts {row_start, col_start, row_end, col_end}.
    """
    if image.ndim == 2:
        image = image[np.newaxis, ...]  # add band dim

    bands, H, W = image.shape
    stride = int(tile_size * (1 - overlap))
    if stride < 1:
        stride = 1

    tiles = []
    coords = []

    row = 0
    while row < H:
        col = 0
        row_end = min(row + tile_size, H)
        row_start = max(row_end - tile_size, 0)

        while col < W:
            col_end = min(col + tile_size, W)
            col_start = max(col_end - tile_size, 0)

            patch = image[:, row_start:row_end, col_start:col_end]

            # Pad if necessary (edge tiles smaller than tile_size)
            pad_r = tile_size - patch.shape[1]
            pad_c = tile_size - patch.shape[2]
            if pad_r > 0 or pad_c > 0:
                patch = np.pad(
                    patch,
                    ((0, 0), (0, pad_r), (0, pad_c)),
                    mode="reflect"
                )

            # Quality checks
            nan_fraction = np.isnan(patch).mean()
            if nan_fraction > max_cloud_fraction:
                col += stride
                continue

            zero_mask = np.all(patch == 0, axis=0)
            valid_fraction = 1.0 - zero_mask.mean()
            if valid_fraction < min_valid_fraction:
                col += stride
                continue

            tiles.append(patch.astype(np.float32))
            coords.append({
                "row_start": int(row_start),
                "col_start": int(col_start),
                "row_end": int(row_end),
                "col_end": int(col_end),
                "pad_r": int(pad_r),
                "pad_c": int(pad_c),
            })

            col += stride
        row += stride

    return tiles, coords


def tile_dataset(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: float = 0.20,
    output_dir: str = "data/tiles/satellite",
    prefix: str = "tile",
    meta: Optional[Dict[str, Any]] = None,
    resume: bool = True,
    min_valid_fraction: float = 0.10,
    max_cloud_fraction: float = 0.90,
) -> Dict[str, Any]:
    """
    Tile an entire image and save tiles + metadata to disk.

    Supports checkpoint/resume: if output_dir already has tiles from a
    previous run, only missing tiles are generated.

    Args:
        image: Array [bands, H, W] (or [T, B, H, W] — first date used).
        tile_size: Tile size in pixels.
        overlap: Fractional overlap.
        output_dir: Directory to save .npy tile files.
        prefix: Filename prefix for tiles.
        meta: Optional rasterio metadata dict (saved to JSON).
        resume: If True, skip tiles already on disk.
        min_valid_fraction: Skip tiles with insufficient valid pixels.
        max_cloud_fraction: Skip tiles with too many cloud-masked pixels.

    Returns:
        Dict with:
            - 'output_dir': str
            - 'num_tiles': int
            - 'tile_ids': list of strings
            - 'coords': list of coordinate dicts
            - 'meta': metadata dict
    """
    os.makedirs(output_dir, exist_ok=True)
    progress_path = os.path.join(output_dir, "tiling_progress.json")

    # Handle time-stacked input
    if image.ndim == 4:
        logger.info("4D input detected; using temporal median for tiling reference.")
        image = np.nanmedian(image, axis=0)  # [B, H, W]

    tiles, coords = tile_image(
        image, tile_size, overlap, min_valid_fraction, max_cloud_fraction
    )

    # Resume: skip tiles already saved
    progress = load_json(progress_path) if (resume and os.path.exists(progress_path)) else {}
    saved_so_far = progress.get("saved", 0)

    tile_ids = []
    for i, (tile, coord) in enumerate(zip(tiles, coords)):
        tile_id = f"{prefix}_{i:05d}"
        npy_path = os.path.join(output_dir, f"{tile_id}.npy")

        if resume and i < saved_so_far and os.path.exists(npy_path):
            tile_ids.append(tile_id)
            continue

        np.save(npy_path, tile)
        tile_ids.append(tile_id)

        # Save progress every 100 tiles
        if (i + 1) % 100 == 0:
            save_json(
                {"saved": i + 1, "total": len(tiles)},
                progress_path
            )
            logger.debug(f"Tiling progress: {i+1}/{len(tiles)}")

    # Save final metadata
    meta_out = {
        "tile_size": tile_size,
        "overlap": overlap,
        "num_tiles": len(tiles),
        "image_shape": list(image.shape),
        "coords": coords,
        "tile_ids": tile_ids,
        "prefix": prefix,
    }
    if meta:
        # Serialise rasterio metadata
        meta_serialisable = {
            k: str(v) for k, v in meta.items()
        }
        meta_out["raster_meta"] = meta_serialisable

    save_json(meta_out, os.path.join(output_dir, "tiles_metadata.json"))
    save_json({"saved": len(tiles), "total": len(tiles)}, progress_path)

    logger.info(
        f"Tiling complete: {len(tiles)} tiles saved to {output_dir} "
        f"(tile_size={tile_size}, overlap={overlap})"
    )

    return {
        "output_dir": output_dir,
        "num_tiles": len(tiles),
        "tile_ids": tile_ids,
        "coords": coords,
        "meta": meta_out,
    }


def tile_uav_for_yolo(
    uav_array: np.ndarray,
    tile_size: int = 640,
    overlap: float = 0.20,
    output_dir: str = "data/tiles/uav",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create YOLO-compatible tiles from the full-resolution UAV orthomosaic.

    Tiles are saved as both .npy (for model pipeline) and .png (for YOLO).

    Args:
        uav_array: Array [bands, H, W] at original UAV resolution (float32, 0–1).
        tile_size: Tile size for YOLO (typically 640).
        overlap: Fractional overlap.
        output_dir: Output directory.
        meta: Raster metadata.

    Returns:
        Dict with tiling results.
    """
    # import cv2as_cv2
    try:
        import cv2 as cv2
    except ImportError:
        cv2 = None

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    tiles, coords = tile_image(uav_array, tile_size, overlap,
                               min_valid_fraction=0.05, max_cloud_fraction=1.0)

    tile_ids = []
    for i, (tile, coord) in enumerate(zip(tiles, coords)):
        tile_id = f"uav_tile_{i:05d}"
        np.save(os.path.join(output_dir, f"{tile_id}.npy"), tile)

        # Save as PNG for YOLO training
        n_bands = min(3, tile.shape[0])
        rgb = (tile[:n_bands].transpose(1, 2, 0) * 255).astype(np.uint8)
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)

        png_path = os.path.join(images_dir, f"{tile_id}.png")
        if cv2 is not None:
            cv2.imwrite(png_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        else:
            try:
                from PIL import Image
                Image.fromarray(rgb).save(png_path)
            except Exception:
                pass

        # Create empty label file (user fills with annotations)
        with open(os.path.join(labels_dir, f"{tile_id}.txt"), "w") as f:
            pass  # YOLO format: class cx cy w h (normalised)

        tile_ids.append(tile_id)

    meta_out = {
        "tile_size": tile_size,
        "overlap": overlap,
        "num_tiles": len(tiles),
        "coords": coords,
        "tile_ids": tile_ids,
    }
    save_json(meta_out, os.path.join(output_dir, "uav_tiles_metadata.json"))

    logger.info(
        f"UAV tiling complete: {len(tiles)} YOLO tiles → {images_dir}"
    )
    return {
        "output_dir": output_dir,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "num_tiles": len(tiles),
        "tile_ids": tile_ids,
        "coords": coords,
        "meta": meta_out,
    }


# =============================================================================
# Reconstruction
# =============================================================================

def reconstruct_from_tiles(
    predictions: List[np.ndarray],
    coords: List[Dict[str, int]],
    image_shape: Tuple[int, int],
    tile_size: int = 512,
) -> np.ndarray:
    """
    Reconstruct a full probability map from per-tile predictions.

    Uses weighted-average blending in overlap regions: tiles are blended
    with a 2D cosine window to suppress boundary artefacts.

    Args:
        predictions: List of arrays [1, tile_size, tile_size] or [tile_size, tile_size],
                     each containing predicted probabilities in [0, 1].
        coords: List of coordinate dicts from tile_dataset().
        image_shape: (H, W) of the full output map.
        tile_size: Tile edge length in pixels.

    Returns:
        Probability map array of shape [H, W] with values in [0, 1].
    """
    H, W = image_shape
    prob_map = np.zeros((H, W), dtype=np.float64)
    weight_map = np.zeros((H, W), dtype=np.float64)

    # 2D Hanning window for smooth blending
    window_1d = np.hanning(tile_size).astype(np.float64)
    window_2d = np.outer(window_1d, window_1d)
    window_2d = window_2d / (window_2d.max() + 1e-8)  # peak = 1

    for pred, coord in zip(predictions, coords):
        rs = coord["row_start"]
        cs = coord["col_start"]
        re = coord["row_end"]
        ce = coord["col_end"]
        pad_r = coord.get("pad_r", 0)
        pad_c = coord.get("pad_c", 0)

        # Remove pred dim if needed
        if pred.ndim == 3:
            pred = pred[0]

        # Remove padding
        actual_h = re - rs
        actual_w = ce - cs
        pred_crop = pred[:actual_h, :actual_w]
        win_crop = window_2d[:actual_h, :actual_w]

        prob_map[rs:re, cs:ce] += pred_crop * win_crop
        weight_map[rs:re, cs:ce] += win_crop

    # Normalise by accumulated weights
    valid = weight_map > 0
    prob_map[valid] /= weight_map[valid]

    return prob_map.astype(np.float32)


# =============================================================================
# Orchestrator
# =============================================================================

def run_tiling(
    preprocessed: Dict[str, Any],
    config: Dict[str, Any],
    uav_available: bool = False
) -> Dict[str, Any]:
    """
    Tile all preprocessed data sources.

    Args:
        preprocessed: Output from preprocessor.preprocess_all().
        config: Config dict.
        uav_available: Whether UAV data is available.

    Returns:
        Dict with tiling results for each data source.
    """
    paths = config["paths"]
    tiling_cfg = config["tiling"]
    sat_tile_size = tiling_cfg["satellite_tile_size"]
    uav_tile_size = tiling_cfg["uav_tile_size"]
    overlap = tiling_cfg["overlap"]
    max_cloud = tiling_cfg.get("max_cloud_fraction", 1.0 - tiling_cfg.get("min_cloud_free", 0.1))
    min_valid = tiling_cfg.get("min_valid_fraction", 0.10)

    results = {}

    # Satellite tiles (use Sentinel-2 composite as primary)
    if "sentinel2" in preprocessed:
        s2_comp = preprocessed["sentinel2"]["composite"]  # [B, H, W]
        s2_meta = preprocessed["sentinel2"]["meta"]
        logger.info("Tiling Sentinel-2 ...")
        results["sentinel2"] = tile_dataset(
            s2_comp, sat_tile_size, overlap,
            os.path.join(paths["tiles"], "satellite"),
            prefix="s2",
            meta=s2_meta,
            max_cloud_fraction=max_cloud,
            min_valid_fraction=min_valid,
        )

    # SAR tiles
    if "sentinel1" in preprocessed:
        s1_comp = preprocessed["sentinel1"]["composite"]
        s1_meta = preprocessed["sentinel1"]["meta"]
        logger.info("Tiling Sentinel-1 ...")
        results["sentinel1"] = tile_dataset(
            s1_comp, sat_tile_size, overlap,
            os.path.join(paths["tiles"], "sentinel1"),
            prefix="s1",
            meta=s1_meta,
            max_cloud_fraction=1.0,  # SAR has no clouds
            min_valid_fraction=min_valid,
        )

    # UAV tiles (high-res for YOLO)
    if uav_available and "uav" in preprocessed:
        uav_hr = preprocessed["uav"]["data_highres"]
        uav_meta = preprocessed["uav"]["meta_highres"]
        logger.info("Tiling UAV (high-res for YOLO) ...")
        results["uav"] = tile_uav_for_yolo(
            uav_hr, uav_tile_size, overlap,
            os.path.join(paths["tiles"], "uav"),
            meta=uav_meta,
        )

    # ---------------------------------------------------------------
    # Tile the 10 m UAV auto-mask alongside satellite tiles so that
    # each satellite tile has a matching mask tile on disk.
    # Priority: auto-generated UAV mask → full_scene_mask fallback.
    # ---------------------------------------------------------------
    masks_dir = paths["masks"]
    uav_auto_mask_path = os.path.join(masks_dir, "uav_mining_mask_10m.npy")
    full_scene_mask_path = os.path.join(masks_dir, "full_scene_mask.npy")

    mask_npy_path = None
    if os.path.exists(uav_auto_mask_path):
        mask_npy_path = uav_auto_mask_path
        logger.info("Tiling auto-generated UAV 10 m mask alongside satellite tiles ...")
    elif os.path.exists(full_scene_mask_path):
        mask_npy_path = full_scene_mask_path
        logger.info("Tiling full_scene_mask alongside satellite tiles ...")

    if mask_npy_path is not None and "sentinel2" in results:
        mask = np.load(mask_npy_path)                      # [H, W]
        sat_coords   = results["sentinel2"]["coords"]
        sat_tile_ids = results["sentinel2"]["tile_ids"]
        sat_dir      = results["sentinel2"]["output_dir"]

        H_mask, W_mask = mask.shape
        saved_mask_tiles = 0

        for tile_id, coord in zip(sat_tile_ids, sat_coords):
            rs = coord["row_start"]
            cs = coord["col_start"]
            re = coord["row_end"]
            ce = coord["col_end"]

            # Clamp to mask dimensions (mask may differ slightly from satellite)
            rs_m = min(rs, H_mask)
            re_m = min(re, H_mask)
            cs_m = min(cs, W_mask)
            ce_m = min(ce, W_mask)

            patch = mask[rs_m:re_m, cs_m:ce_m]

            # Pad to sat_tile_size if needed
            pad_r = sat_tile_size - patch.shape[0]
            pad_c = sat_tile_size - patch.shape[1]
            if pad_r > 0 or pad_c > 0:
                patch = np.pad(
                    patch, ((0, pad_r), (0, pad_c)),
                    mode="constant", constant_values=0,
                )

            mask_path = os.path.join(sat_dir, f"{tile_id}_mask.npy")
            np.save(mask_path, patch.astype(np.uint8))
            saved_mask_tiles += 1

        logger.info(
            f"Mask tiling complete: {saved_mask_tiles} mask tiles "
            f"saved alongside satellite tiles in {sat_dir}"
        )
        results["mask_tiling"] = {
            "source": mask_npy_path,
            "n_mask_tiles": saved_mask_tiles,
            "output_dir": sat_dir,
        }
    else:
        if mask_npy_path is None:
            logger.info(
                "No mask found during tiling (run generate_masks or "
                "prepare_ground_truth first to create mask tiles)."
            )

    logger.info("All tiling complete.")
    return results