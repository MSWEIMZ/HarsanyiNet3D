"""
HybridNet: R(2+1)D feature extractor + HarsanyiNet3D top layers.

Architecture:
  Input (1, D, H, W)
    → R(2+1)D-18 backbone (frozen, Kinetics-400 pretrained) → (256, 8, 14, 14)
    → Adapter (AdaptiveAvgPool3d + Conv1×1×1) → z0 (channels, K, K, K)
    → HarsanyiBlock3D × num_layers (exact same code)
    → Per-block FC → sum → FC_final → num_classes

Shapley values computed over the z0 feature map positions (K³ variables).
These correspond to semantic features extracted by the pre-trained backbone.
"""
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.HarsanyiNet3D import HarsanyiBlock3D, HarsanyiNet3D


class HybridR2Plus1D(nn.Module):
    """
    Hybrid model: R(2+1)D-18 feature extractor + HarsanyiBlock3D top.
    """

    def __init__(
        self,
        num_classes: int = 16,
        num_layers: int = 6,
        channels: int = 128,
        beta: int = 1000,
        gamma: float = 1.0,
        conv_size: int = 8,
        fc_size: int = 32,
        threshold_t: float = 0.0,
        device: str = 'cuda:0',
        comparable_DNN: bool = False,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.channels = channels
        self.device = device
        self.conv_size = conv_size

        # ===== R(2+1)D-18 backbone (frozen) =====
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)

        # Adapt first conv from 3→45 to 1→45 (average over input channels)
        old_stem = self.backbone.stem[0]
        new_stem = nn.Conv3d(
            1, old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=False,
        )
        new_stem.weight.data = old_stem.weight.data.mean(dim=1, keepdim=True)
        self.backbone.stem[0] = new_stem

        # Remove classification head
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ===== Feature adapter: (256, 8, 14, 14) → (channels, K, K, K) =====
        # After layer3: (B, 256, 8, 14, 14)
        self.adapter = nn.Sequential(
            nn.AdaptiveAvgPool3d((conv_size, conv_size, conv_size)),
            nn.Conv3d(256, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        # ===== HarsanyiBlock3D layers (exact same as HarsanyiNet3D) =====
        self.HarsanyiBlocks = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(num_layers):
            self.HarsanyiBlocks.append(HarsanyiBlock3D(
                conv_size=conv_size,
                in_channels=channels,
                out_channels=channels,
                beta=beta,
                gamma=gamma,
                threshold_t=threshold_t,
                device=device,
                comparable_DNN=comparable_DNN,
            ))
            self.fc.append(nn.Linear(
                channels * conv_size * conv_size * conv_size, fc_size, bias=False
            ))

        self.fc_final = nn.Linear(fc_size, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. R(2+1)D backbone → intermediate features
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)       # (B, 256, 8, 14, 14)

        # 2. Adapter → z0
        x = self.adapter(x)                # (B, channels, K, K, K)

        # 3. HarsanyiBlocks (same as HarsanyiNet3D)
        hidden_y = None
        for layer in range(self.num_layers):
            x, _ = self.HarsanyiBlocks[layer](x)
            y = self.fc[layer](torch.flatten(x, 1))
            if hidden_y is None:
                hidden_y = y
            else:
                hidden_y = hidden_y + y

        output = self.fc_final(hidden_y)
        return output

    def _get_z0(self, x: torch.Tensor) -> torch.Tensor:
        """Extract z0 features for Shapley computation."""
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.adapter(x)
        return x

    def _get_value(self, x: torch.Tensor):
        """Forward returning intermediate values for Shapley."""
        z0 = self._get_z0(x)
        hidden_y = None
        ys, zs, deltas = [], [], []
        out = z0
        for layer in range(self.num_layers):
            out, delta = self.HarsanyiBlocks[layer](out)
            zs.append(out)
            y = self.fc[layer](torch.flatten(out, 1))
            ys.append(y)
            deltas.append(delta)
            if hidden_y is None:
                hidden_y = y
            else:
                hidden_y = hidden_y + y
        output = self.fc_final(hidden_y)
        return output, ys, zs, deltas
