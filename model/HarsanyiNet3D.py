"""
HarsanyiNet3D: Computing Accurate Shapley Values in a Single Forward Propagation
for 3D volumetric data (e.g., microscopy Z-stacks).

Architecture:
  Input (1, D, H, W)
    → Stem: aggessive spatiotemporal downsampling → z0 (C, K, K, K)
    → HarsanyiBlock3D × L (extend → STE gate → Conv3d → AND)
    → Per-block FC → sum → FC_final → num_classes

Based on: "HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation"
          ICML 2023, https://arxiv.org/abs/2304.01811
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

EPS = 1e-30
THRESHOLD_EPS = 1e-3


# =============================================================================
# Straight-Through Estimator (STE) — unchanged from 2D version
# =============================================================================
class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator.
      forward:  1(input > 0)
      backward: beta * sigmoid'(input)  = beta * e^{-x} / (1 + e^{-x})^2
    """
    @staticmethod
    def forward(ctx, input_, beta=1, slope=1):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        ctx.beta = beta
        out = (input_ > 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = (ctx.beta * grad_input * ctx.slope *
                torch.exp(-ctx.slope * input_) /
                ((torch.exp(-ctx.slope * input_) + 1) ** 2 + EPS))
        return grad, None, None


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, beta=1):
        return STEFunction.apply(x, beta)


# =============================================================================
# 3D Convolution helper
# =============================================================================
def conv3x3x3(in_channels: int, out_channels: int,
              stride: int = 1, padding: int = 1) -> nn.Conv3d:
    """3×3×3 Conv3d with bias=False."""
    return nn.Conv3d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=padding, bias=False
    )


# =============================================================================
# HarsanyiBlock3D
# =============================================================================
class HarsanyiBlock3D(nn.Module):
    """
    A single Harsanyi block for 3D data.

    Given input (B, C, K, K, K):
      1. _extend_layer_3d:  pad → gather along D/H/W → (B, C, 3K, 3K, 3K)
      2. Apply STE on v_weight (3D mask) to select children nodes
      3. Conv3d(3×3×3, stride=3) → (B, C_out, K, K, K)
      4. _get_trigger_value_3d: AND gate — geometric mean of children activations
      5. output = ReLU(conv(x) × delta)
    """

    def __init__(
        self,
        conv_size: int,           # K — cubic spatial dim of z0
        in_channels: int,
        out_channels: int,
        beta: int = 1000,
        gamma: float = 1.0,
        threshold_t: float = 0.0,
        device: str = 'cuda:0',
        comparable_DNN: bool = False,
    ) -> None:
        super().__init__()
        self.conv = conv3x3x3(in_channels, out_channels, stride=3, padding=0)
        # v_weight: 3D mask over the (3K)³ extended space
        # each element selects whether a child node at (i,j,k) connects
        K3 = conv_size * 3
        self.v_weight = nn.Parameter(torch.randn(K3, K3, K3))
        self.conv_size = conv_size
        self.beta = beta
        self.gamma = gamma
        self.device = device
        self.threshold_t = threshold_t
        self.comparable_DNN = comparable_DNN

        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = torch.sigmoid
        self.ste = StraightThroughEstimator()

        self._init_weights()

        if self.comparable_DNN:
            self.conv_comparable_DNN = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=3, stride=1, padding=1, bias=False
            )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # Initialize v_weight similar to Linear in 2D: N(0, 0.01)
        nn.init.normal_(self.v_weight, 0, 0.01)

    def forward(self, x: Tensor) -> Tensor:
        if self.comparable_DNN:
            x = self.conv_comparable_DNN(x)
            delta = torch.zeros(x.shape[0], self.conv_size, self.conv_size, self.conv_size,
                                device=self.device)
        else:
            x, delta = self._layer(x)
        output = self.relu(x)
        return output, delta

    def _layer(self, x: Tensor) -> Tensor:
        # 1. Extend: (B, C, K, K, K) → (B, C, 3K, 3K, 3K)
        x_enlarge = self._extend_layer_3d(x)

        # 2. Apply children-node selection mask via STE
        v_mask = self.ste(self.v_weight, beta=self.beta)          # (3K, 3K, 3K)
        x_enlarge = x_enlarge * v_mask                            # broadcast over (B, C)

        # 3. Linear combination: Conv3d(3×3×3, stride=3) → (B, C_out, K, K, K)
        x = self.conv(x_enlarge)

        # 4. AND operation: compute trigger value δ
        delta = self._get_trigger_value_3d(x_enlarge)             # (B, K, K, K)
        delta = delta.unsqueeze(dim=1)                            # (B, 1, K, K, K)
        x = x * delta

        return x, delta

    def _extend_layer_3d(self, x: Tensor) -> Tensor:
        """
        Extend (B, C, K, K, K) → (B, C, 3K, 3K, 3K) by duplicating
        positions in an overlapping sliding-window pattern.

        Index pattern: [0,1,2,1,2,3,2,3,4,...] for each spatial dimension.
        Each input position is replicated 3 times, creating a 3× expansion.
        """
        B, C, K = x.shape[0], x.shape[1], x.shape[2]
        K3 = K * 3

        # Pad 1 on all 6 sides
        x = F.pad(x, (1, 1, 1, 1, 1, 1))  # (B, C, K+2, K+2, K+2)

        # Generate index pattern: [0,1,2,1,2,3,2,3,4,...] of length 3K
        indice = torch.tensor(
            [int(i / 3) + i % 3 for i in range(K3)],
            device=self.device
        )  # shape: (K3,)

        # Gather along W (dim=4), then H (dim=3), then D (dim=2)
        # Each gather expands that dimension by 3×
        idx_w = indice[None, None, None, None, :]     # (1,1,1,1,K3)
        idx_w = idx_w.expand(B, C, K + 2, K + 2, K3)  # (B,C,K+2,K+2,K3)
        x = torch.gather(x, 4, idx_w)                  # gather along W

        idx_h = indice[None, None, None, :, None]     # (1,1,1,K3,1)
        idx_h = idx_h.expand(B, C, K + 2, K3, K3)     # (B,C,K+2,K3,K3)
        x = torch.gather(x, 3, idx_h)                  # gather along H

        idx_d = indice[None, None, :, None, None]     # (1,1,K3,1,1)
        idx_d = idx_d.expand(B, C, K3, K3, K3)         # (B,C,K3,K3,K3)
        x = torch.gather(x, 2, idx_d)                  # gather along D

        return x  # (B, C, 3K, 3K, 3K)

    def _get_trigger_value_3d(self, input_en: Tensor) -> Tensor:
        """
        Compute the AND-gate trigger value δ for each output unit.

        δ ∈ [0, 1):  > 0  if ALL selected children nodes are activated
                      = 0  if ANY child node is not activated

        Steps:
          1. L1 norm across channels → (B, 3K, 3K, 3K)
          2. tanh(gamma * (norm - threshold)) → δ_raw ∈ [0, 1)
          3. 3D unfold (3×3×3, stride=3) to extract patches
          4. Geometric mean over selected children per patch
          5. Fold back to (B, K, K, K)
        """
        K = self.conv_size
        # 1. L1 norm over input channels
        input_norm = torch.norm(input_en, p=1, dim=1)  # (B, 3K, 3K, 3K)

        # 2. Tanh gating
        delta_en = torch.tanh(self.gamma * (input_norm - self.threshold_t))  # (B, 3K, 3K, 3K)
        delta_en = delta_en.unsqueeze(dim=1)  # (B, 1, 3K, 3K, 3K)

        # 3. 3D "unfold": extract non-overlapping 3×3×3 patches, stride=3
        #    (B, 1, 3K, 3K, 3K) → (B, 27, K³)
        delta_patches = self._unfold_3d(delta_en)         # (B, 27, K³)

        # 4. Likewise for v mask
        v_is_child = self.ste(self.v_weight, beta=1)       # (3K, 3K, 3K)
        v_patches = self._unfold_3d(
            v_is_child.unsqueeze(0).unsqueeze(0)           # (1, 1, 3K, 3K, 3K)
        )  # (1, 27, K³)

        # 5. Geometric mean of children activation
        #    δ_prod = exp( sum(log(δ) * v) / sum(v) )
        log_delta = torch.log(delta_patches + EPS)          # (B, 27, K³)
        weighted_log = log_delta * v_patches                 # (B, 27, K³)
        sum_log = torch.sum(weighted_log, dim=1, keepdim=True)  # (B, 1, K³)
        v_count = torch.sum(v_patches, dim=1, keepdim=True)     # (B, 1, K³)
        delta_prod = torch.exp(sum_log / (v_count + EPS))       # (B, 1, K³)

        # Force to 0 if δ < THRESHOLD_EPS (any child not activated)
        delta_prod = delta_prod.squeeze(dim=1)                # (B, K³)
        ZEROS = torch.zeros_like(delta_prod)
        delta_prod = torch.where(delta_prod > THRESHOLD_EPS, delta_prod, ZEROS)

        # Zero out patches where v has no selected children
        zero_position = v_count.squeeze(dim=1) / (v_count.squeeze(dim=1) + EPS)  # (B, K³)
        delta_prod = delta_prod * zero_position

        # 6. Fold back to (B, K, K, K)
        delta = delta_prod.reshape(-1, K, K, K)
        return delta

    def _unfold_3d(self, x: Tensor, kernel: int = 3, stride: int = 3) -> Tensor:
        """
        3D analog of torch.nn.Unfold: extract non-overlapping patches.

        Args:
            x: (B, C, D, H, W)
            kernel: patch size per dim (3)
            stride: stride per dim (3)
        Returns:
            patches: (B, C*k³, D_out*H_out*W_out)
        """
        B, C, D, H, W = x.shape
        k = kernel
        s = stride
        D_out, H_out, W_out = D // s, H // s, W // s

        # Reshape to expose patches as block dimensions
        # (B, C, D_out, k, H_out, k, W_out, k)
        x = x.reshape(B, C, D_out, k, H_out, k, W_out, k)
        # Permute to bring patch elements together
        # (B, C, D_out, H_out, W_out, k, k, k)
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
        # Flatten: (B, C*k³, D_out*H_out*W_out)
        x = x.reshape(B, C * k**3, D_out * H_out * W_out)
        return x


# =============================================================================
# HarsanyiNet3D
# =============================================================================
class HarsanyiNet3D(nn.Module):
    """
    Full HarsanyiNet for 3D volumetric data.

    Architecture:
      Input (B, 1, D_in, H_in, W_in)
        → Stem (3× 3D Conv+BN+ReLU + MaxPool) → z0 (B, channels, K, K, K)
        → HarsanyiBlock3D × num_layers
        → Per-block FC (flatten → fc_size) → sum
        → FC_final (fc_size → num_classes)

    The Shapley values are computed over the K³ variables of z0.
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
        device: str = 'cuda:1',
        in_channels: int = 1,
        comparable_DNN: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.channels = channels
        self.device = device
        self.conv_size = conv_size

        # Stem: (B, 1, D_in, H_in, W_in) → (B, channels, K, K, K)
        self.stem = self._build_stem(in_channels, channels, conv_size)

        # HarsanyiBlocks
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
        self._init_weights()

    def _build_stem(self, in_channels: int, channels: int, K: int) -> nn.Sequential:
        """
        Aggressive spatiotemporal downsampling stem.

        Input:  (B, 1, 32, 224, 224)   (example)
        Output: (B, channels, K, K, K)   cubic
        """
        layers = [
            # ---- Block 1: heavy spatial down, temporal keep ----
            nn.Conv3d(in_channels, 48,
                      kernel_size=(1, 7, 7), stride=(1, 4, 4),
                      padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            # (48, D, 56, 56) with D = input_D

            # ---- Block 2: moderate spatiotemporal down ----
            nn.Conv3d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            # (96, D//2, 28, 28)  (with floor)

            # ---- Block 3: more aggressive spatiotemporal down ----
            nn.Conv3d(96, channels,
                      kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            # (channels, D//4, 14, 14)

            # ---- Block 4: spatial-only down ----
            nn.Conv3d(channels, channels,
                      kernel_size=(1, 3, 3), stride=(1, 2, 2),
                      padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            # (channels, D//4, 7, 7) — spatial ~7, temporal ~8
        ]

        # After these blocks, temporal dim is D//4 = 8 (for input D=32).
        # Spatial dim is ~7. To get exactly K cubic, we pad spatially.
        #
        # The pad amount is computed at runtime, but we statically set it
        # based on expected input sizes.
        # We'll handle this in forward() via an adaptive crop/pad if needed,
        # OR we just set K to exactly match the spatial dim after stem.
        #
        # For D_in=32, the stem output is (channels, 8, 7, 7).
        # We pad H and W by (0, 1) each to get (channels, 8, 8, 8).
        #
        # To keep the stem flexible, we use a lightweight adaptive layer:
        layers.append(nn.AdaptiveAvgPool3d((K, K, K)))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self._get_z0(x)

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

    def load_pretrained_stem(self, stem_path: str) -> None:
        """
        加载预训练 stem 权重。

        Args:
            stem_path: .pth 文件路径 (由 pretrain_stem.py 生成)
        """
        state_dict = torch.load(stem_path, map_location=self.device)
        missing, unexpected = self.stem.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[Stem loading] Missing keys: {len(missing)}")
            for k in missing:
                print(f"  - {k}")
        if unexpected:
            print(f"[Stem loading] Unexpected keys: {len(unexpected)}")
            for k in unexpected:
                print(f"  - {k}")
        if not missing and not unexpected:
            print(f"[Stem loading] Loaded {len(state_dict)} keys successfully ✓")

    def _get_z0(self, x: Tensor) -> Tensor:
        """
        Input:  raw 3D volume (B, in_channels, D, H, W)
        Output: z0 feature map (B, channels, K, K, K)

        If x already has 'channels' channels, assume it's already z0.
        """
        if x.shape[1] != self.channels:
            x = self.stem(x)
        return x

    def _get_value(self, x: Tensor):
        """
        Forward pass returning intermediate values for Shapley computation.

        Returns:
            output: (B, num_classes)  — final logits
            ys: list of per-block FC outputs [ (B, fc_size), ... ]
            zs: list of per-block feature maps [ (B, C, K, K, K), ... ]
            deltas: list of per-block AND-gate values [ (B, K, K, K), ... ]
        """
        x = self._get_z0(x)

        hidden_y = None
        ys, zs, deltas = [], [], []
        for layer in range(self.num_layers):
            x, delta = self.HarsanyiBlocks[layer](x)
            zs.append(x)
            y = self.fc[layer](torch.flatten(x, 1))
            ys.append(y)
            deltas.append(delta)
            if hidden_y is None:
                hidden_y = y
            else:
                hidden_y = hidden_y + y

        output = self.fc_final(hidden_y)
        return output, ys, zs, deltas
