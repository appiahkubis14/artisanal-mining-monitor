"""
utils.py - Shared Utility Functions
====================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Provides reusable helpers for I/O, geospatial transforms, logging,
timing, checkpointing, and spatial cross-validation used across all scripts.
"""

import os
import json
import time
import logging
import functools
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
import torch
import rasterio
from rasterio.transform import from_bounds, rowcol, xy
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_filename: str = "pipeline.log"
) -> logging.Logger:
    """
    Configure root logger to write to console and a rotating file.

    Args:
        log_dir: Directory where log files are stored.
        log_level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_filename: Name of the log file.

    Returns:
        Configured root Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(numeric_level)
    ch.setFormatter(formatter)

    # File handler
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    if not root_logger.handlers:
        root_logger.addHandler(ch)
        root_logger.addHandler(fh)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger by name."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Timer decorator
# ---------------------------------------------------------------------------

def timer(func):
    """
    Decorator that logs wall-clock execution time of any function.

    Usage:
        @timer
        def my_function(): ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger("timer")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(
            f"{func.__name__} completed in "
            f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"
        )
        return result
    return wrapper


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None
) -> None:
    """
    Save a training checkpoint to disk.

    Args:
        state: Dictionary containing model state_dict, optimizer, epoch, metrics.
        filepath: Path to save checkpoint .pth file.
        is_best: If True, also copy to best_filepath.
        best_filepath: Path for best model weights.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        import shutil
        shutil.copyfile(filepath, best_filepath)
        get_logger("checkpoint").info(f"New best model saved → {best_filepath}")


def load_checkpoint(
    filepath: str,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        filepath: Path to the .pth checkpoint file.
        device: Device to map tensors to.

    Returns:
        Dictionary with checkpoint contents.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    get_logger("checkpoint").info(f"Checkpoint loaded from {filepath}")
    return checkpoint


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save arbitrary JSON-serialisable data to a file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(filepath: str) -> Any:
    """Load JSON data from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def raster_to_numpy(
    filepath: str,
    band_indices: Optional[List[int]] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read a raster file into a NumPy array.

    Args:
        filepath: Path to GeoTIFF or other GDAL-readable raster.
        band_indices: 1-based list of bands to read. Reads all if None.

    Returns:
        Tuple of (array with shape [bands, height, width], metadata dict).
    """
    with rasterio.open(filepath) as src:
        if band_indices is not None:
            data = src.read(band_indices).astype(np.float32)
        else:
            data = src.read().astype(np.float32)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "bounds": src.bounds,
        }
    return data, meta


def numpy_to_geotiff(
    array: np.ndarray,
    filepath: str,
    meta: Dict[str, Any],
    dtype: str = "float32",
    nodata: Optional[float] = None,
    compress: str = "lzw"
) -> None:
    """
    Write a NumPy array to a GeoTIFF file.

    Args:
        array: Array of shape [bands, height, width] or [height, width].
        filepath: Output path for GeoTIFF.
        meta: Metadata dict from raster_to_numpy (contains crs, transform).
        dtype: Output data type string (e.g. 'float32', 'uint8').
        nodata: No-data value for the output raster.
        compress: Compression algorithm ('lzw', 'deflate', 'none').
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    bands, height, width = array.shape

    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "width": width,
        "height": height,
        "count": bands,
        "crs": meta["crs"],
        "transform": meta["transform"],
        "compress": compress,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    elif meta.get("nodata") is not None:
        profile["nodata"] = meta["nodata"]

    with rasterio.open(filepath, "w", **profile) as dst:
        dst.write(array.astype(dtype))

    get_logger("raster_io").debug(f"Saved GeoTIFF: {filepath} ({bands} bands, {height}x{width})")


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def latlon_to_pixel(
    lat: float,
    lon: float,
    transform: rasterio.transform.Affine,
    crs: CRS
) -> Tuple[int, int]:
    """
    Convert geographic coordinates (lat/lon WGS84) to pixel row/col.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        transform: Affine transform of the raster.
        crs: Coordinate reference system of the raster.

    Returns:
        Tuple of (row, col) pixel indices.
    """
    import pyproj
    from pyproj import Transformer

    # Project lat/lon (EPSG:4326) → raster CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    row, col = rowcol(transform, x, y)
    return int(row), int(col)


def pixel_to_latlon(
    row: int,
    col: int,
    transform: rasterio.transform.Affine,
    crs: CRS
) -> Tuple[float, float]:
    """
    Convert pixel row/col to geographic coordinates (lat/lon WGS84).

    Args:
        row: Pixel row index.
        col: Pixel column index.
        transform: Affine transform of the raster.
        crs: Coordinate reference system of the raster.

    Returns:
        Tuple of (latitude, longitude) in decimal degrees.
    """
    from pyproj import Transformer

    x, y = xy(transform, row, col)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float]
) -> float:
    """
    Compute Intersection-over-Union for two axis-aligned bounding boxes.

    Args:
        box1: (x_min, y_min, x_max, y_max)
        box2: (x_min, y_min, x_max, y_max)

    Returns:
        IoU score in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute IoU between two binary segmentation masks.

    Args:
        mask1: Binary mask array.
        mask2: Binary mask array (same shape as mask1).

    Returns:
        IoU score in [0, 1].
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Spatial cross-validation split
# ---------------------------------------------------------------------------

def split_geospatial_train_test(
    gdf: gpd.GeoDataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    buffer_km: float = 2.0,
    seed: int = 42
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Split a GeoDataFrame into train/val/test sets with spatial buffering
    to prevent data leakage from spatial autocorrelation.

    Points within `buffer_km` of a split boundary are excluded from
    the opposite split to ensure true spatial independence.

    Args:
        gdf: GeoDataFrame with Point or Polygon geometries in a projected CRS.
        val_fraction: Fraction of data for validation.
        test_fraction: Fraction of data for testing.
        buffer_km: Exclusion buffer in kilometres.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_gdf, val_gdf, test_gdf).
    """
    rng = np.random.RandomState(seed)
    n = len(gdf)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    buffer_m = buffer_km * 1000.0

    test_gdf = gdf.iloc[test_idx].copy()
    val_gdf = gdf.iloc[val_idx].copy()
    train_gdf = gdf.iloc[train_idx].copy()

    # Remove training points within buffer of test/val
    test_union = test_gdf.geometry.union_all() if hasattr(test_gdf.geometry, 'union_all') else test_gdf.geometry.unary_union
    val_union = val_gdf.geometry.union_all() if hasattr(val_gdf.geometry, 'union_all') else val_gdf.geometry.unary_union

    test_buffer = test_union.buffer(buffer_m)
    val_buffer = val_union.buffer(buffer_m)

    train_mask = ~(
        train_gdf.geometry.intersects(test_buffer) |
        train_gdf.geometry.intersects(val_buffer)
    )
    train_gdf = train_gdf[train_mask]

    logger = get_logger("spatial_split")
    logger.info(
        f"Spatial split: train={len(train_gdf)}, val={len(val_gdf)}, "
        f"test={len(test_gdf)} (buffer={buffer_km}km)"
    )
    return train_gdf, val_gdf, test_gdf


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device(preferred: str = "cuda") -> torch.device:
    """
    Return a torch device, falling back to CPU if CUDA unavailable.

    Args:
        preferred: 'cuda' or 'cpu'.

    Returns:
        torch.device instance.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        get_logger("device").info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        get_logger("device").info("Using CPU")
    return device


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def save_progress(progress: Dict[str, Any], filepath: str) -> None:
    """Save pipeline progress state to JSON for checkpoint/resume."""
    save_json(progress, filepath)


def load_progress(filepath: str) -> Dict[str, Any]:
    """Load pipeline progress from JSON, return empty dict if not found."""
    if os.path.exists(filepath):
        return load_json(filepath)
    return {}


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalize_array(
    array: np.ndarray,
    lower_pct: float = 2.0,
    upper_pct: float = 98.0
) -> np.ndarray:
    """
    Percentile-based normalisation of a numpy array to [0, 1].

    Args:
        array: Input array of any shape.
        lower_pct: Lower percentile for clipping.
        upper_pct: Upper percentile for clipping.

    Returns:
        Normalised array with values in [0, 1].
    """
    p_low = np.percentile(array[np.isfinite(array)], lower_pct)
    p_high = np.percentile(array[np.isfinite(array)], upper_pct)
    clipped = np.clip(array, p_low, p_high)
    if p_high - p_low < 1e-10:
        return np.zeros_like(clipped)
    return (clipped - p_low) / (p_high - p_low)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute binary classification metrics.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].
        threshold: Decision threshold for positive class.

    Returns:
        Dict with keys: precision, recall, f1, accuracy, iou.
    """
    pred_binary = (y_pred >= threshold).astype(np.int32)
    true_binary = y_true.astype(np.int32)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "iou": float(iou),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
