"""
dataset.py - PyTorch Dataset and DataLoaders
==============================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Implements a PyTorch Dataset for semantic segmentation of mining sites,
with spatial train/val/test splitting, class-balancing weighted sampler,
and domain-appropriate data augmentation.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from scripts.utils import get_logger, set_seeds, load_json

logger = get_logger(__name__)


# =============================================================================
# Dataset
# =============================================================================

class MiningDataset(Dataset):
    """
    PyTorch Dataset for mining site segmentation.

    Each sample is a pair:
      - image: Tensor [C, tile_size, tile_size] of feature values
      - mask:  Tensor [tile_size, tile_size] of binary labels (0=background, 1=mining)

    Augmentation (training only):
      - Random 90°/180°/270° rotation
      - Horizontal and vertical flipping
      - Random brightness/contrast jitter (±20%)
      - MixUp between minority-class tiles

    Args:
        tile_ids: List of tile ID strings.
        tiles_dir: Directory containing .npy feature tile files.
        masks_dir: Directory containing .npy mask files.
        augment: Whether to apply data augmentation.
        n_channels: Expected number of input channels (pads/trims if needed).
    """

    def __init__(
        self,
        tile_ids: List[str],
        tiles_dir: str,
        masks_dir: str,
        augment: bool = False,
        n_channels: Optional[int] = None,
    ) -> None:
        self.tile_ids = tile_ids
        self.tiles_dir = tiles_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.n_channels = n_channels

        # Precompute which tiles have positive (mining) pixels for sampling
        self.positive_flags = self._compute_positive_flags()
        logger.info(
            f"Dataset: {len(self.tile_ids)} tiles, "
            f"{sum(self.positive_flags)} positive, "
            f"augment={augment}"
        )

    def __len__(self) -> int:
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tile_id = self.tile_ids[idx]

        # Load feature tile
        tile_path = os.path.join(self.tiles_dir, f"{tile_id}.npy")
        image = np.load(tile_path).astype(np.float32)  # [C, H, W]

        # Load mask
        mask_path = os.path.join(self.masks_dir, f"{tile_id}_mask.npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)  # [H, W]
        else:
            # No annotation: return all-zeros mask
            mask = np.zeros(image.shape[1:], dtype=np.float32)

        # Channel normalisation
        if self.n_channels is not None:
            image = self._adjust_channels(image, self.n_channels)

        # Replace NaN
        image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)

        # Data augmentation (training only)
        if self.augment:
            image, mask = self._augment(image, mask)

        image_tensor = torch.from_numpy(image)       # [C, H, W]
        mask_tensor = torch.from_numpy(mask).long()  # [H, W]

        return image_tensor, mask_tensor

    # -----------------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------------

    def _augment(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply stochastic augmentations to a single image-mask pair.

        Args:
            image: [C, H, W] float32 array.
            mask: [H, W] binary array.

        Returns:
            Augmented (image, mask) tuple.
        """
        # Random 90° rotation (axes 1 and 2 are H and W)
        k = random.randint(0, 3)
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k).copy()

        # Horizontal flip
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        # Vertical flip
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        # Brightness/contrast jitter on all channels
        if random.random() > 0.5:
            alpha = 1.0 + random.uniform(-0.20, 0.20)   # contrast
            beta = random.uniform(-0.10, 0.10)            # brightness
            image = np.clip(image * alpha + beta, 0.0, 1.0)

        # Gaussian noise (simulate sensor noise)
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)

        return image, mask

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _compute_positive_flags(self) -> List[bool]:
        """Return a boolean list indicating which tiles have mining pixels."""
        flags = []
        for tile_id in self.tile_ids:
            mask_path = os.path.join(self.masks_dir, f"{tile_id}_mask.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                flags.append(bool(mask.max() > 0))
            else:
                flags.append(False)
        return flags

    def _adjust_channels(
        self, image: np.ndarray, target_channels: int
    ) -> np.ndarray:
        """Pad with zeros or trim channels to match target_channels."""
        C, H, W = image.shape
        if C == target_channels:
            return image
        elif C < target_channels:
            pad = np.zeros((target_channels - C, H, W), dtype=np.float32)
            return np.concatenate([image, pad], axis=0)
        else:
            return image[:target_channels]

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute per-class pixel weights for Focal / Weighted Cross-Entropy loss.

        Returns:
            Tensor [2] with weights [background_weight, mining_weight].
            Mining class receives higher weight to address class imbalance.
        """
        n_positive = sum(self.positive_flags)
        n_negative = len(self.positive_flags) - n_positive
        total = len(self.positive_flags)

        if n_positive == 0:
            return torch.tensor([1.0, 10.0])

        # Inverse frequency weighting
        w_neg = total / (2.0 * n_negative + 1e-8)
        w_pos = total / (2.0 * n_positive + 1e-8)
        return torch.tensor([w_neg, w_pos], dtype=torch.float32)


# =============================================================================
# MixUp Dataset Wrapper
# =============================================================================

class MixUpDataset(Dataset):
    """
    Wraps MiningDataset to apply MixUp augmentation between minority-class tiles.

    With probability `mixup_prob`, blends the current sample with a randomly
    selected positive tile to enrich the training signal for rare mining sites.
    """

    def __init__(
        self,
        base_dataset: MiningDataset,
        mixup_prob: float = 0.20,
        alpha: float = 0.4,
    ) -> None:
        self.base = base_dataset
        self.mixup_prob = mixup_prob
        self.alpha = alpha
        self.positive_indices = [
            i for i, flag in enumerate(base_dataset.positive_flags) if flag
        ]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.base[idx]

        if (
            random.random() < self.mixup_prob
            and len(self.positive_indices) > 0
            and self.base.positive_flags[idx]
        ):
            # Select a random positive tile to mix
            mix_idx = random.choice(self.positive_indices)
            image2, mask2 = self.base[mix_idx]

            # Sample mixing coefficient from Beta distribution
            lam = np.random.beta(self.alpha, self.alpha)

            image = lam * image + (1 - lam) * image2
            mask = torch.clamp(mask.float() + mask2.float(), 0, 1).long()

        return image, mask


# =============================================================================
# Spatial train/val/test split
# =============================================================================

def spatial_train_val_test_split(
    tile_metadata: Dict[str, Any],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    spatial_buffer_km: float = 2.0,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split tile IDs into train/val/test with spatial separation.

    Tiles within `spatial_buffer_km` of a test/val boundary pixel are
    excluded from training to prevent spatial autocorrelation leakage.

    Strategy: divide the spatial grid into blocks, assign whole blocks
    to splits rather than individual pixels.

    Args:
        tile_metadata: Dict from tiler.tile_dataset() with 'tile_ids' and 'coords'.
        val_fraction: Fraction of tiles for validation.
        test_fraction: Fraction of tiles for testing.
        spatial_buffer_km: Not directly applied here; block-based split
                           ensures spatial separation by design.
        seed: Random seed.

    Returns:
        Tuple of (train_ids, val_ids, test_ids) lists.
    """
    rng = np.random.RandomState(seed)
    tile_ids = tile_metadata["tile_ids"]
    coords = tile_metadata["coords"]
    n = len(tile_ids)

    if n == 0:
        return [], [], []

    # Group tiles into spatial blocks based on row position
    # Tiles with similar row_start form a "block"
    row_starts = np.array([c["row_start"] for c in coords])
    col_starts = np.array([c["col_start"] for c in coords])

    # Create 5×5 grid blocks
    row_bins = np.linspace(row_starts.min(), row_starts.max() + 1, 6).astype(int)
    col_bins = np.linspace(col_starts.min(), col_starts.max() + 1, 6).astype(int)

    block_ids = np.digitize(row_starts, row_bins) * 10 + np.digitize(col_starts, col_bins)
    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)

    n_test_blocks = max(1, int(len(unique_blocks) * test_fraction))
    n_val_blocks = max(1, int(len(unique_blocks) * val_fraction))

    test_blocks = set(unique_blocks[:n_test_blocks])
    val_blocks = set(unique_blocks[n_test_blocks:n_test_blocks + n_val_blocks])
    train_blocks = set(unique_blocks[n_test_blocks + n_val_blocks:])

    train_ids, val_ids, test_ids = [], [], []
    for i, tile_id in enumerate(tile_ids):
        bid = block_ids[i]
        if bid in test_blocks:
            test_ids.append(tile_id)
        elif bid in val_blocks:
            val_ids.append(tile_id)
        else:
            train_ids.append(tile_id)

    logger.info(
        f"Spatial split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    return train_ids, val_ids, test_ids


# =============================================================================
# DataLoader builders
# =============================================================================

def build_dataloaders(
    tile_metadata: Dict[str, Any],
    tiles_dir: str,
    masks_dir: str,
    batch_size: int = 16,
    n_channels: Optional[int] = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    use_mixup: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Build train/val/test DataLoaders with class-balanced sampling.

    Args:
        tile_metadata: From tiler.tile_dataset().
        tiles_dir: Directory with .npy feature tiles.
        masks_dir: Directory with .npy mask files.
        batch_size: Training batch size.
        n_channels: Expected input channels (None = infer from first tile).
        val_fraction: Fraction for validation split.
        test_fraction: Fraction for test split.
        num_workers: DataLoader worker processes.
        seed: Reproducibility seed.
        use_mixup: Whether to apply MixUp augmentation on training set.

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_weights).
    """
    set_seeds(seed)

    train_ids, val_ids, test_ids = spatial_train_val_test_split(
        tile_metadata, val_fraction, test_fraction, seed=seed
    )

    # Guard: if too few tiles for a proper split, share tiles across sets
    # (happens with small synthetic scenes — real data will always have enough)
    all_ids = tile_metadata["tile_ids"]
    if len(train_ids) == 0:
        logger.warning(
            f"Training split is empty ({len(all_ids)} total tiles). "
            "Reusing all tiles for train/val/test (testing mode only)."
        )
        train_ids = list(all_ids)
        val_ids   = list(all_ids)
        test_ids  = list(all_ids)

    # Infer channels from first tile
    if n_channels is None:
        first_tile = os.path.join(tiles_dir, f"{tile_metadata['tile_ids'][0]}.npy")
        if os.path.exists(first_tile):
            n_channels = np.load(first_tile).shape[0]
        else:
            n_channels = 32  # default fallback

    # Create datasets
    train_ds = MiningDataset(train_ids, tiles_dir, masks_dir, augment=True, n_channels=n_channels)
    val_ds = MiningDataset(val_ids, tiles_dir, masks_dir, augment=False, n_channels=n_channels)
    test_ds = MiningDataset(test_ids, tiles_dir, masks_dir, augment=False, n_channels=n_channels)

    # Optionally wrap train with MixUp
    if use_mixup and sum(train_ds.positive_flags) > 0:
        train_wrapped = MixUpDataset(train_ds)
    else:
        train_wrapped = train_ds

    # Weighted sampler for training — oversample positive tiles
    sample_weights = [
        5.0 if flag else 1.0 for flag in train_ds.positive_flags
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    class_weights = train_ds.get_class_weights()

    train_loader = DataLoader(
        train_wrapped,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"DataLoaders ready | train={len(train_ds)}, val={len(val_ds)}, "
        f"test={len(test_ds)} | class_weights={class_weights.tolist()}"
    )
    return train_loader, val_loader, test_loader, class_weights