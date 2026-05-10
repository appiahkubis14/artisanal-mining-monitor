"""
models.py - Deep Learning Architecture Definitions
====================================================
Atewa Forest Reserve Illegal Mining Detection System
Master's Thesis, KNUST Ghana

Defines:
  - U-Net with ResNet-50 encoder (satellite mining segmentation)
  - Attention U-Net (optional architecture)
  - ResU-Net (optional architecture)
  - YOLOv8 wrapper for UAV equipment detection
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# Building blocks
# =============================================================================

class ConvBnRelu(nn.Module):
    """3×3 Conv → BatchNorm → ReLU block."""
    def __init__(self, in_ch: int, out_ch: int, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two successive ConvBnRelu blocks — standard U-Net building block."""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate as in Attention U-Net (Oktay et al. 2018).

    Weights skip-connection features by a gating signal from the decoder.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g if spatial sizes differ
        if g1.shape[-2:] != x1.shape[-2:]:
            g1 = F.interpolate(g1, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """
    U-Net decoder block: upsample + concatenate skip + DoubleConv.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_attention: bool = False) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        concat_ch = in_ch // 2 + skip_ch
        self.conv = DoubleConv(concat_ch, out_ch)
        if use_attention:
            self.attention = AttentionGate(in_ch // 2, skip_ch, skip_ch // 2)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if self.use_attention:
            skip = self.attention(x, skip)
        # Align sizes (handles odd dimensions)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# =============================================================================
# U-Net with ResNet-50 encoder
# =============================================================================

class UNetResNet50(nn.Module):
    """
    U-Net with a ResNet-50 encoder pretrained on ImageNet.

    The encoder produces 5 feature maps at strides 1, 2, 4, 8, 16.
    The decoder uses transposed convolutions with skip connections.

    Input channels are adapted via a 1×1 projection layer so the model
    can accept any number of spectral/feature channels.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output classes (1 for binary segmentation).
        pretrained: Whether to use ImageNet pretrained weights.
        use_attention: Whether to use attention gates in decoder.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
        pretrained: bool = True,
        use_attention: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        try:
            import torchvision.models as models
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
        except Exception:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=pretrained)

        # Input projection: any channels → 64 (ResNet expects 3/64)
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder layers
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)   # [64, H/2, W/2]
        self.pool = resnet.maxpool                                            # [64, H/4, W/4]
        self.enc1 = resnet.layer1   # [256, H/4, W/4]
        self.enc2 = resnet.layer2   # [512, H/8, W/8]
        self.enc3 = resnet.layer3   # [1024, H/16, W/16]
        self.enc4 = resnet.layer4   # [2048, H/32, W/32]

        # Bottleneck
        self.bottleneck = DoubleConv(2048, 1024)

        # Decoder
        self.dec4 = DecoderBlock(1024, 1024, 512, use_attention)
        self.dec3 = DecoderBlock(512, 512, 256, use_attention)
        self.dec2 = DecoderBlock(256, 256, 128, use_attention)
        self.dec1 = DecoderBlock(128, 64, 64, use_attention)
        self.dec0 = DecoderBlock(64, 64, 32, use_attention)

        # Final 1×1 conv → logits
        self.head = nn.Conv2d(32, out_channels, kernel_size=1)

        logger.info(
            f"UNetResNet50: in_channels={in_channels}, out_channels={out_channels}, "
            f"attention={use_attention}, pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, C, H, W] input feature tensor.

        Returns:
            [batch, 1, H, W] logits (apply sigmoid for probabilities).
        """
        # Input projection
        x0 = self.input_proj(x)    # [B, 64, H, W]

        # Encoder
        e0 = self.enc0(x0)         # [B, 64, H/2, W/2]
        e0p = self.pool(e0)        # [B, 64, H/4, W/4]
        e1 = self.enc1(e0p)        # [B, 256, H/4, W/4]
        e2 = self.enc2(e1)         # [B, 512, H/8, W/8]
        e3 = self.enc3(e2)         # [B, 1024, H/16, W/16]
        e4 = self.enc4(e3)         # [B, 2048, H/32, W/32]

        # Bottleneck
        b = self.bottleneck(e4)    # [B, 1024, H/32, W/32]

        # Decoder with skip connections
        d4 = self.dec4(b, e3)      # [B, 512, H/16, W/16]
        d3 = self.dec3(d4, e2)     # [B, 256, H/8, W/8]
        d2 = self.dec2(d3, e1)     # [B, 128, H/4, W/4]
        d1 = self.dec1(d2, e0p)    # [B, 64, H/4, W/4]
        d0 = self.dec0(d1, e0)     # [B, 32, H/2, W/2]

        # Final upsampling to input resolution
        d0_up = F.interpolate(d0, size=x.shape[-2:], mode="bilinear", align_corners=False)

        logits = self.head(d0_up)  # [B, 1, H, W]
        return logits


# =============================================================================
# Lightweight U-Net (fallback, no pretrained weights needed)
# =============================================================================

class LightweightUNet(nn.Module):
    """
    Simple U-Net without pretrained encoder.

    Used as fallback if torchvision is unavailable or when fine-tuning
    on limited data (fewer parameters to overfit).

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output classes.
        features: List of feature map sizes for each encoder level.
    """

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()

        ch = in_channels
        for feat in features:
            self.encoders.append(DoubleConv(ch, feat))
            self.pools.append(nn.MaxPool2d(2, 2))
            ch = feat

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feat in reversed(features):
            self.decoders.append(nn.ConvTranspose2d(feat * 2, feat, 2, 2))
            self.decoders.append(DoubleConv(feat * 2, feat))

        self.head = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skips = list(reversed(skips))

        for i in range(0, len(self.decoders), 2):
            x = self.decoders[i](x)
            skip = skips[i // 2]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i + 1](x)

        return self.head(x)


# =============================================================================
# Model factory
# =============================================================================

def build_unet(
    architecture: str = "unet",
    in_channels: int = 32,
    pretrained: bool = True,
) -> nn.Module:
    """
    Build a segmentation model by name.

    Args:
        architecture: One of 'unet', 'attention_unet', 'resunet', 'lightweight'.
        in_channels: Number of input channels.
        pretrained: Whether to use pretrained encoder weights.

    Returns:
        Instantiated PyTorch model.
    """
    arch = architecture.lower()

    if arch in ("unet", "resunet"):
        try:
            model = UNetResNet50(in_channels=in_channels, pretrained=pretrained)
        except ImportError:
            logger.warning("torchvision not available; using LightweightUNet.")
            model = LightweightUNet(in_channels=in_channels)

    elif arch == "attention_unet":
        try:
            model = UNetResNet50(in_channels=in_channels, pretrained=pretrained, use_attention=True)
        except ImportError:
            model = LightweightUNet(in_channels=in_channels)

    elif arch == "lightweight":
        model = LightweightUNet(in_channels=in_channels)

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Choose from: unet, attention_unet, resunet, lightweight."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {arch}, in_channels={in_channels}, params={n_params:,}")
    return model


# =============================================================================
# Transfer learning utilities
# =============================================================================

def freeze_encoder(model: nn.Module) -> None:
    """
    Freeze all ResNet encoder layers (leave input_proj and decoder trainable).

    Useful for initial training on limited labelled data.
    """
    if hasattr(model, "enc0"):
        for module in [model.enc0, model.enc1, model.enc2, model.enc3, model.enc4]:
            for param in module.parameters():
                param.requires_grad = False
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        logger.info(f"Encoder frozen: {frozen} parameters locked.")


def unfreeze_layers(model: nn.Module, n_layers: int = 2) -> None:
    """
    Gradually unfreeze the last `n_layers` encoder stages.

    Args:
        model: UNetResNet50 model.
        n_layers: Number of encoder stages to unfreeze (1–4, from deepest).
    """
    encoder_stages = []
    for name in ["enc4", "enc3", "enc2", "enc1", "enc0"]:
        if hasattr(model, name):
            encoder_stages.append(getattr(model, name))

    for stage in encoder_stages[:n_layers]:
        for param in stage.parameters():
            param.requires_grad = True

    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Unfroze {n_layers} encoder stage(s): {trainable} trainable params.")


# =============================================================================
# YOLOv8 wrapper
# =============================================================================

class YOLOv8Detector:
    """
    Wrapper around Ultralytics YOLOv8 for UAV equipment detection.

    Handles training, inference, and export via the ultralytics API.

    Equipment classes:
        0: excavator
        1: truck
        2: water_pump
        3: settling_pond
        4: pit

    Args:
        model_path: Path to pretrained .pt file ('yolov8m.pt') or trained model.
        device: 'cuda' or 'cpu'.
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self._model = None

    def load(self) -> None:
        """Load the YOLO model from disk or download pretrained weights."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"YOLOv8 model loaded: {self.model_path}")
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLO. "
                "Install with: pip install ultralytics"
            )

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        patience: int = 20,
        project: str = "data/models",
        name: str = "yolo_mining",
    ) -> str:
        """
        Train YOLOv8 on UAV tile dataset.

        Args:
            data_yaml: Path to YOLO dataset.yaml file.
            epochs: Training epochs.
            imgsz: Image size.
            batch: Batch size.
            patience: Early stopping patience.
            project: Output project directory.
            name: Run name.

        Returns:
            Path to best model weights.
        """
        if self._model is None:
            self.load()

        results = self._model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            project=project,
            name=name,
            device=self.device,
            exist_ok=True,
            save=True,
            val=True,
        )

        best_path = str(results.save_dir / "weights" / "best.pt")
        logger.info(f"YOLO training complete. Best model: {best_path}")
        return best_path

    def predict(
        self,
        image_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to image file.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            imgsz: Inference image size.

        Returns:
            List of detection dicts:
              {class_name, confidence, bbox [x1, y1, x2, y2]}
        """
        if self._model is None:
            self.load()

        results = self._model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            save=False,
            verbose=False,
        )

        detections = []
        class_names = ["excavator", "truck", "water_pump", "settling_pond", "pit"]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls.item())
                detections.append({
                    "class_id": cls_id,
                    "class_name": class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}",
                    "confidence": float(box.conf.item()),
                    "bbox": box.xyxy.cpu().numpy()[0].tolist(),  # [x1, y1, x2, y2]
                })

        return detections
