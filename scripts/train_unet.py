"""
train_unet.py - U-Net Training Script
=======================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Trains the U-Net segmentation model with:
  - Focal Loss for class imbalance
  - AdamW optimiser
  - CosineAnnealingLR schedule
  - Automatic Mixed Precision (AMP)
  - Early stopping
  - TensorBoard logging
  - Checkpoint / resume support
"""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from scripts.models import build_unet, freeze_encoder, unfreeze_layers
from scripts.dataset import build_dataloaders
from scripts.utils import (
    get_logger, set_seeds, get_device,
    save_checkpoint, load_checkpoint, compute_binary_metrics, timer,
)

logger = get_logger(__name__)


# =============================================================================
# Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for the minority (positive) class.
        gamma: Focusing parameter — higher values down-weight easy examples.
        reduction: 'mean', 'sum', or 'none'.

    Reference:
        Lin et al. (2017). Focal Loss for Dense Object Detection. ICCV.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw model output [B, 1, H, W] (pre-sigmoid).
            targets: Binary ground truth [B, H, W] (0 or 1, long).

        Returns:
            Scalar loss value.
        """
        targets_float = targets.float().unsqueeze(1)  # [B, 1, H, W]

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_float, reduction="none"
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets_float + (1 - probs) * (1 - targets_float)
        alpha_t = self.alpha * targets_float + (1 - self.alpha) * (1 - targets_float)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CombinedLoss(nn.Module):
    """
    Combined Focal + Dice Loss for robust segmentation training.

    Dice Loss encourages global shape agreement; Focal handles pixel-level
    class imbalance. Combined weight: 0.5 * focal + 0.5 * dice.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets_f = targets.float().unsqueeze(1)
        smooth = 1.0
        intersection = (probs * targets_f).sum()
        dice = (2.0 * intersection + smooth) / (probs.sum() + targets_f.sum() + smooth)
        return 1.0 - dice

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice_loss(logits, targets)


# =============================================================================
# Training loop
# =============================================================================

@timer
def train_unet(
    tile_metadata: Dict[str, Any],
    feature_stack: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full training pipeline for the U-Net segmentation model.

    Args:
        tile_metadata: From tiler.tile_dataset().
        feature_stack: Full-scene feature array [C, H, W] (used to infer channel count).
        config: Config dict from config.yaml.

    Returns:
        Dict with:
            - 'best_model_path': path to best checkpoint
            - 'final_metrics': val metrics at best epoch
            - 'history': training history dict
    """
    cfg = config["training_unet"]
    paths = config["paths"]
    seed = config.get("random_seed", 42)
    set_seeds(seed)

    device = get_device(config.get("device", "cuda"))
    os.makedirs(paths["models"], exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # --- Data ---
    n_channels = feature_stack.shape[0] if feature_stack is not None else 32
    tiles_dir = os.path.join(paths["tiles"], "satellite")
    masks_dir = paths["masks"]

    train_loader, val_loader, test_loader, class_weights = build_dataloaders(
        tile_metadata,
        tiles_dir=tiles_dir,
        masks_dir=masks_dir,
        batch_size=cfg["batch_size"],
        n_channels=n_channels,
        val_fraction=cfg.get("val_split", 0.15),
        test_fraction=cfg.get("test_split", 0.15),
        seed=seed,
    )

    # --- Model ---
    model = build_unet(
        architecture=cfg.get("architecture", "unet"),
        in_channels=n_channels,
        pretrained=cfg.get("pretrained", True),
    ).to(device)

    # Phase 1: freeze encoder for first 10 epochs
    freeze_encoder(model)

    # --- Loss ---
    criterion = CombinedLoss(
        alpha=cfg.get("focal_loss_alpha", 0.25),
        gamma=cfg.get("focal_loss_gamma", 2.0),
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=1e-6
    )

    # --- AMP scaler ---
    scaler = GradScaler() if (cfg.get("amp", True) and device.type == "cuda") else None

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir="runs/unet_training")

    # --- Resume from checkpoint ---
    best_model_path = os.path.join(paths["models"], "unet_best.pth")
    last_model_path = os.path.join(paths["models"], "unet_last.pth")
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float("inf")

    if os.path.exists(last_model_path):
        logger.info(f"Resuming from checkpoint: {last_model_path}")
        ckpt = load_checkpoint(last_model_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # --- History ---
    history = {
        "train_loss": [], "val_loss": [],
        "val_precision": [], "val_recall": [],
        "val_f1": [], "val_iou": [],
        "lr": [],
    }
    patience_counter = 0

    logger.info(
        f"Training U-Net: {cfg['epochs']} epochs, "
        f"batch={cfg['batch_size']}, lr={cfg['learning_rate']}, device={device}"
    )

    for epoch in range(start_epoch, cfg["epochs"]):
        # --- Unfreeze encoder after epoch 10 ---
        if epoch == 10:
            unfreeze_layers(model, n_layers=2)
            optimizer.add_param_group({
                "params": [p for p in model.parameters() if not p.requires_grad],
                "lr": cfg["learning_rate"] * 0.1,
            })
            for p in model.parameters():
                p.requires_grad = True
            logger.info("Encoder unfrozen at epoch 10.")

        # --- Train ---
        train_loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_clip=cfg.get("grad_clip", 1.0)
        )

        # --- Validate ---
        val_loss, val_metrics = _validate_epoch(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Logging ---
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["lr"].append(current_lr)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Metrics", {
            "F1": val_metrics["f1"],
            "IoU": val_metrics["iou"],
            "Precision": val_metrics["precision"],
            "Recall": val_metrics["recall"],
        }, epoch)
        writer.add_scalar("LR", current_lr, epoch)

        logger.info(
            f"Epoch {epoch+1:3d}/{cfg['epochs']} | "
            f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # --- Checkpoint ---
        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
            "val_metrics": val_metrics,
            "config": cfg,
        }
        save_checkpoint(state, last_model_path, is_best, best_model_path)

        # --- Confusion matrix every 5 epochs ---
        if (epoch + 1) % 5 == 0:
            cm_fig = _plot_confusion_matrix(val_metrics)
            writer.add_figure("ConfusionMatrix", cm_fig, epoch)

        # --- Early stopping ---
        if patience_counter >= cfg.get("early_stopping_patience", 15):
            logger.info(
                f"Early stopping at epoch {epoch+1} "
                f"(no improvement for {patience_counter} epochs)."
            )
            break

    writer.close()
    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f} → {best_model_path}")

    # Save training history
    import json
    with open(os.path.join(paths["models"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    _save_training_plots(history, paths["models"])

    return {
        "best_model_path": best_model_path,
        "final_metrics": {
            "val_f1": best_val_f1,
            "val_loss": best_val_loss,
        },
        "history": history,
    }


def _train_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[Any],
    grad_clip: float = 1.0,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _validate_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Run one validation epoch and return (loss, metrics dict)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    n_batches = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).squeeze(1)  # [B, H, W]
            all_preds.append(probs.cpu().numpy().ravel())
            all_targets.append(masks.cpu().numpy().ravel())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_binary_metrics(all_targets, all_preds, threshold=0.5)

    return total_loss / max(n_batches, 1), metrics


def _plot_confusion_matrix(metrics: Dict[str, Any]):
    """Create a matplotlib figure with the confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
        cm = np.array([[tn, fp], [fn, tp]])

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Background", "Mining"])
        ax.set_yticklabels(["Background", "Mining"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        plt.tight_layout()
        return fig
    except Exception:
        import matplotlib.pyplot as plt
        return plt.figure()


def _save_training_plots(history: Dict[str, list], output_dir: str) -> None:
    """Save loss and metric plots to the models directory."""
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, history["train_loss"], label="Train")
        axes[0].plot(epochs, history["val_loss"], label="Val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, history["val_f1"], label="F1")
        axes[1].plot(epochs, history["val_iou"], label="IoU")
        axes[1].set_title("Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        axes[2].plot(epochs, history["lr"], color="green")
        axes[2].set_title("Learning Rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_yscale("log")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
        plt.close()
        logger.info(f"Training plots saved → {output_dir}/training_curves.png")
    except Exception as exc:
        logger.warning(f"Could not save training plots: {exc}")
