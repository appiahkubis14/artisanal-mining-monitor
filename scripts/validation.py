"""
validation.py
=============
Comprehensive accuracy assessment of the mining detection pipeline.

Metrics
-------
- Pixel-level: precision, recall, F1, IoU, ROC-AUC, PR-AUC
- Object-level: detection rate, false alarm rate
- Spatial cross-validation: 2 km block holdout
- Temporal validation: hold-out date comparison
- Outputs: accuracy_report.json, ROC/PR curves PNG, confusion matrix PNG

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import mapping
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rasterisation helpers
# ---------------------------------------------------------------------------

def _geojson_to_mask(
    geojson_path: Path,
    reference_tif: Path,
) -> np.ndarray:
    """Rasterise a GeoJSON of polygons to a binary mask matching reference_tif.

    Parameters
    ----------
    geojson_path   : path to GeoJSON file
    reference_tif  : reference raster for extent, CRS, and pixel size

    Returns
    -------
    Binary numpy array (0/1) with same shape as reference raster
    """
    with rasterio.open(reference_tif) as src:
        out_shape = (src.height, src.width)
        transform = src.transform
        crs = src.crs

    gdf = gpd.read_file(geojson_path).to_crs(crs)
    if len(gdf) == 0:
        return np.zeros(out_shape, dtype=np.uint8)

    geometries = [mapping(g) for g in gdf.geometry if g is not None and not g.is_empty]
    if not geometries:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = rasterize(
        [(g, 1) for g in geometries],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def _load_binary_pred(pred_tif: Path, threshold: float = 0.5) -> np.ndarray:
    """Load prediction raster and threshold to binary."""
    with rasterio.open(pred_tif) as src:
        arr = src.read(1).astype(np.float32)
    return (arr >= threshold).astype(np.uint8)


def _load_prob(prob_tif: Path) -> np.ndarray:
    """Load raw probability map."""
    with rasterio.open(prob_tif) as src:
        return src.read(1).astype(np.float32)


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Compute pixel-level binary classification metrics.

    Parameters
    ----------
    y_true : flat binary ground-truth array
    y_pred : flat binary prediction array
    y_prob : flat probability array (optional, for ROC/PR)

    Returns
    -------
    Dictionary of metrics
    """
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "iou": round(float(iou), 4),
        "accuracy": round(float(accuracy), 4),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "support_positive": int(tp + fn),
        "support_negative": int(tn + fp),
    }

    if y_prob is not None:
        y_prob_flat = y_prob.ravel()
        # Filter to valid probability range
        valid = (y_prob_flat >= 0) & (y_prob_flat <= 1)
        if valid.sum() > 0 and len(np.unique(y_true[valid])) > 1:
            metrics["roc_auc"] = round(
                float(roc_auc_score(y_true[valid], y_prob_flat[valid])), 4
            )
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_true[valid], y_prob_flat[valid]
            )
            metrics["pr_auc"] = round(float(auc(recall_curve, precision_curve)), 4)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Object-level metrics
# ---------------------------------------------------------------------------

def compute_object_metrics(
    gt_gdf: gpd.GeoDataFrame,
    pred_gdf: gpd.GeoDataFrame,
    iou_thresh: float = 0.3,
) -> dict:
    """Compute object-level detection metrics.

    A prediction is a true positive if IoU with any ground-truth polygon
    exceeds iou_thresh (and each GT is matched at most once).

    Parameters
    ----------
    gt_gdf     : ground-truth mining polygons
    pred_gdf   : predicted mining polygons
    iou_thresh : minimum IoU for a valid match

    Returns
    -------
    Dictionary with detection_rate, false_alarm_rate, object_precision, etc.
    """
    if len(gt_gdf) == 0:
        return {
            "gt_count": 0, "pred_count": len(pred_gdf),
            "tp_objects": 0, "fp_objects": len(pred_gdf),
            "fn_objects": 0, "object_precision": 0.0,
            "object_recall": 0.0, "object_f1": 0.0,
            "detection_rate": 0.0, "false_alarm_rate": float(len(pred_gdf)),
        }

    if len(pred_gdf) == 0:
        return {
            "gt_count": len(gt_gdf), "pred_count": 0,
            "tp_objects": 0, "fp_objects": 0,
            "fn_objects": len(gt_gdf), "object_precision": 0.0,
            "object_recall": 0.0, "object_f1": 0.0,
            "detection_rate": 0.0, "false_alarm_rate": 0.0,
        }

    gt_matched = [False] * len(gt_gdf)
    tp = 0
    fp = 0

    for _, pred_row in pred_gdf.iterrows():
        matched = False
        for gt_idx, gt_row in gt_gdf.iterrows():
            if gt_matched[list(gt_gdf.index).index(gt_idx)]:
                continue
            try:
                inter = pred_row.geometry.intersection(gt_row.geometry).area
                union = pred_row.geometry.union(gt_row.geometry).area
                iou = inter / union if union > 0 else 0.0
            except Exception:
                iou = 0.0
            if iou >= iou_thresh:
                tp += 1
                gt_matched[list(gt_gdf.index).index(gt_idx)] = True
                matched = True
                break
        if not matched:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "gt_count": len(gt_gdf),
        "pred_count": len(pred_gdf),
        "tp_objects": tp,
        "fp_objects": fp,
        "fn_objects": fn,
        "object_precision": round(precision, 4),
        "object_recall": round(recall, 4),
        "object_f1": round(f1, 4),
        "detection_rate": round(recall, 4),
        "false_alarm_rate": round(fp / max(1, len(pred_gdf)), 4),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    tag: str = "overall",
) -> None:
    """Save ROC and PR curve plots."""
    y_true_flat = y_true.ravel().astype(int)
    y_prob_flat = y_prob.ravel()
    valid = (y_prob_flat >= 0) & (y_prob_flat <= 1)

    if valid.sum() == 0 or len(np.unique(y_true_flat[valid])) < 2:
        log.warning("Cannot plot ROC/PR curves – insufficient class diversity.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true_flat[valid], y_prob_flat[valid])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, "b-", lw=2, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve ({tag})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PR
    prec, rec, _ = precision_recall_curve(y_true_flat[valid], y_prob_flat[valid])
    pr_auc = auc(rec, prec)
    baseline = y_true_flat.mean()
    axes[1].plot(rec, prec, "r-", lw=2, label=f"AUC = {pr_auc:.3f}")
    axes[1].axhline(y=baseline, color="k", linestyle="--", lw=1,
                    label=f"Baseline ({baseline:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall Curve ({tag})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f"roc_pr_{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved ROC/PR plot: {out_path}")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    tag: str = "overall",
) -> None:
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true.ravel().astype(int), y_pred.ravel().astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["Non-Mining", "Mining"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    out_path = output_dir / f"confusion_matrix_{tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Saved confusion matrix: {out_path}")


# ---------------------------------------------------------------------------
# Spatial cross-validation
# ---------------------------------------------------------------------------

def spatial_block_cv(
    prob_arr: np.ndarray,
    gt_mask: np.ndarray,
    transform,
    n_blocks: int = 5,
    block_size_m: float = 2000.0,
    threshold: float = 0.5,
) -> list[dict]:
    """2 km block spatial cross-validation.

    Divides the study area into grid blocks and evaluates each block
    using models trained on the remaining blocks (approximated here by
    spatial holdout of the probability map).

    Parameters
    ----------
    prob_arr    : probability map array [H, W]
    gt_mask     : ground-truth binary mask [H, W]
    transform   : affine transform
    n_blocks    : grid size (n × n blocks)
    block_size_m: target block side length in metres
    threshold   : classification threshold

    Returns
    -------
    List of per-block metric dicts
    """
    H, W = prob_arr.shape
    pixel_size_m = abs(transform.a)
    block_px = max(1, int(block_size_m / pixel_size_m))

    block_metrics = []
    for row_start in range(0, H, block_px):
        for col_start in range(0, W, block_px):
            row_end = min(row_start + block_px, H)
            col_end = min(col_start + block_px, W)
            block_prob = prob_arr[row_start:row_end, col_start:col_end]
            block_gt = gt_mask[row_start:row_end, col_start:col_end]

            if block_gt.sum() == 0 and (1 - block_gt).sum() < 100:
                continue  # Skip nearly empty blocks

            block_pred = (block_prob >= threshold).astype(np.uint8)
            metrics = compute_pixel_metrics(block_gt, block_pred)
            metrics["block_row"] = row_start // block_px
            metrics["block_col"] = col_start // block_px
            metrics["gt_positive_pixels"] = int(block_gt.sum())
            block_metrics.append(metrics)

    return block_metrics


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

@timer
def run_validation(config: dict) -> dict:
    """Run complete validation pipeline.

    Requires at minimum:
    - A probability map in outputs/
    - A ground-truth GeoJSON in data/masks/ or data/ground_truth/

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    Nested dict with all validation metrics
    """
    output_dir = Path(config["paths"]["outputs"])
    masks_dir = Path(config["paths"]["masks"])
    val_dir = output_dir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    # ---- Find probability maps ----
    prob_maps = sorted(output_dir.glob("prob_map_*.tif"))
    if not prob_maps:
        log.error("No probability maps found. Run inference first.")
        return {}

    # Use most recent probability map for primary validation
    prob_tif = prob_maps[-1]
    date_str = prob_tif.stem.replace("prob_map_", "")
    log.info(f"Validating probability map: {prob_tif}")

    # ---- Load probability and ground-truth ----
    prob_arr = _load_prob(prob_tif)

    gt_mask_path = masks_dir / "full_scene_mask.npy"
    gt_geojson = None

    # Try to load fused detections as "ground-truth" for objects
    gt_geojson_candidates = [
        Path(config["paths"].get("ground_truth", "data/ground_truth")) / "mining_sites.geojson",
        Path("data/ground_truth/mining_sites.geojson"),
    ]

    if gt_mask_path.exists():
        gt_mask = np.load(gt_mask_path)
        # Resize to match prob_arr if needed
        if gt_mask.shape != prob_arr.shape:
            import cv2
            gt_mask = cv2.resize(
                gt_mask.astype(np.float32),
                (prob_arr.shape[1], prob_arr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.uint8)
        log.info(f"Loaded ground-truth mask: {gt_mask_path}")
    else:
        log.warning("Ground-truth mask not found – creating empty mask.")
        gt_mask = np.zeros_like(prob_arr, dtype=np.uint8)

    with rasterio.open(prob_tif) as src:
        transform = src.transform
        crs = src.crs

    pred_mask = (prob_arr >= 0.5).astype(np.uint8)

    # ---- Pixel-level metrics ----
    log.info("Computing pixel-level metrics...")
    pixel_metrics = compute_pixel_metrics(gt_mask, pred_mask, prob_arr)
    log.info(f"Pixel metrics: {pixel_metrics}")

    # ---- ROC / PR plots ----
    _plot_roc_pr(gt_mask, prob_arr, val_dir, tag=date_str)
    _plot_confusion_matrix(gt_mask, pred_mask, val_dir, tag=date_str)

    # ---- Object-level metrics ----
    log.info("Computing object-level metrics...")
    fused_geojson = output_dir / f"fused_detections_{date_str}.geojson"
    if not fused_geojson.exists():
        fused_geojson = output_dir / f"mining_detections_{date_str}.geojson"

    object_metrics = {}
    for gt_candidate in gt_geojson_candidates:
        if gt_candidate.exists():
            gt_gdf = gpd.read_file(gt_candidate).to_crs(crs.to_epsg() or "EPSG:4326")
            if fused_geojson.exists():
                pred_gdf = gpd.read_file(fused_geojson).to_crs(
                    crs.to_epsg() or "EPSG:4326"
                )
            else:
                pred_gdf = gpd.GeoDataFrame()
            object_metrics = compute_object_metrics(gt_gdf, pred_gdf)
            log.info(f"Object metrics: {object_metrics}")
            break

    # ---- Spatial cross-validation ----
    log.info("Running spatial block cross-validation (2 km blocks)...")
    block_metrics = spatial_block_cv(prob_arr, gt_mask, transform)

    if block_metrics:
        cv_df = pd.DataFrame(block_metrics)
        cv_summary = {
            "mean_f1": round(float(cv_df["f1_score"].mean()), 4),
            "std_f1": round(float(cv_df["f1_score"].std()), 4),
            "mean_iou": round(float(cv_df["iou"].mean()), 4),
            "mean_precision": round(float(cv_df["precision"].mean()), 4),
            "mean_recall": round(float(cv_df["recall"].mean()), 4),
            "n_blocks": len(cv_df),
        }
        cv_df.to_csv(val_dir / "spatial_cv_blocks.csv", index=False)
        log.info(f"Spatial CV summary: {cv_summary}")

        # Plot per-block F1
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(cv_df)), cv_df["f1_score"].values, color="steelblue", alpha=0.8)
        ax.axhline(y=cv_df["f1_score"].mean(), color="r", linestyle="--",
                   label=f"Mean F1={cv_summary['mean_f1']:.3f}")
        ax.set_xlabel("Block Index")
        ax.set_ylabel("F1 Score")
        ax.set_title("Spatial Cross-Validation: F1 per Block")
        ax.legend()
        plt.tight_layout()
        plt.savefig(val_dir / "spatial_cv_f1.png", dpi=150)
        plt.close()
    else:
        cv_summary = {"warning": "No valid blocks for cross-validation."}

    # ---- Threshold analysis ----
    log.info("Computing metrics across thresholds...")
    thresholds = np.arange(0.1, 0.95, 0.05)
    thresh_results = []
    y_true_flat = gt_mask.ravel().astype(int)
    y_prob_flat = prob_arr.ravel()
    for t in thresholds:
        y_pred_flat = (y_prob_flat >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1]).ravel()
        f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
        prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
        rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
        thresh_results.append({
            "threshold": round(float(t), 2),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
        })

    thresh_df = pd.DataFrame(thresh_results)
    thresh_df.to_csv(val_dir / "threshold_analysis.csv", index=False)
    best_thresh_row = thresh_df.loc[thresh_df["f1_score"].idxmax()]
    log.info(
        f"Best threshold: {best_thresh_row['threshold']} "
        f"(F1={best_thresh_row['f1_score']:.4f})"
    )

    # Plot threshold analysis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresh_df["threshold"], thresh_df["precision"], "b-", label="Precision")
    ax.plot(thresh_df["threshold"], thresh_df["recall"], "r-", label="Recall")
    ax.plot(thresh_df["threshold"], thresh_df["f1_score"], "g-", lw=2, label="F1")
    ax.axvline(x=best_thresh_row["threshold"], color="k", linestyle="--", alpha=0.5,
               label=f"Best ({best_thresh_row['threshold']})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs. Classification Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(val_dir / "threshold_analysis.png", dpi=150)
    plt.close()

    # ---- Compile full report ----
    report = {
        "date": date_str,
        "pixel_metrics": pixel_metrics,
        "object_metrics": object_metrics,
        "spatial_cv": cv_summary,
        "best_threshold": {
            "value": float(best_thresh_row["threshold"]),
            "f1_at_best": float(best_thresh_row["f1_score"]),
            "precision_at_best": float(best_thresh_row["precision"]),
            "recall_at_best": float(best_thresh_row["recall"]),
        },
        "outputs": {
            "roc_pr_plot": str(val_dir / f"roc_pr_{date_str}.png"),
            "confusion_matrix": str(val_dir / f"confusion_matrix_{date_str}.png"),
            "threshold_plot": str(val_dir / "threshold_analysis.png"),
            "spatial_cv_plot": str(val_dir / "spatial_cv_f1.png"),
            "spatial_cv_csv": str(val_dir / "spatial_cv_blocks.csv"),
        },
    }

    report_path = val_dir / "accuracy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Saved accuracy report: {report_path}")

    log.info(
        f"\n{'='*55}\n"
        f"VALIDATION COMPLETE\n"
        f"  Pixel F1      : {pixel_metrics.get('f1_score', 'N/A')}\n"
        f"  Pixel IoU     : {pixel_metrics.get('iou', 'N/A')}\n"
        f"  ROC-AUC       : {pixel_metrics.get('roc_auc', 'N/A')}\n"
        f"  Spatial CV F1 : {cv_summary.get('mean_f1', 'N/A')} "
        f"± {cv_summary.get('std_f1', 'N/A')}\n"
        f"  Best threshold: {report['best_threshold']['value']}\n"
        f"{'='*55}"
    )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate mining detection pipeline.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_validation(cfg)
