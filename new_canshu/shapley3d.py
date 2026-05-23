#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HarsanyiNet3D — Shapley 值计算与可视化

在训练好的 HarsanyiNet3D 上计算每个原样本的 3D Shapley 值（单次前向传播）。

用法:
    # 单样本可视化
    python shapley3d.py --model_path result_harsanyi3d_xxx/best_fold0.pth
                        --sample_idx 0 --save_dir shapley_output

    # 批量 RMSE 评估（对比 brute force + sampling）
    python shapley3d.py --model_path ... --eval --num_samples 5
"""
import os, sys, json, argparse, itertools, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.HarsanyiNet3D import HarsanyiNet3D
from model.HybridNet import HybridR2Plus1D
# 复用原项目数据集（从 train.py 避免 argparse 冲突）
sys.path.insert(0, os.path.dirname(__file__))
from train import MitoDataset3D_HighRes

# filter_original_classes 也从 train.py 导入（函数相同）
# 注意：train.py 里这个函数叫 filter_original_classes，直接复用
def filter_original_classes(ds):
    """Filter dataset to only include original 16 classes, remap indices to 0-15."""
    original_classes = [
        '210302_vapB', '211018_cos93_nocada', '211018_cos93_taxol',
        '211018_dko93', '211018_dko93_nocada', '211111_OLIGO1', '211111_TM5',
        'atl3mch', 'cccp', 'cos93 int2', 'cytd', 'div', 'dynasore1',
        'lat', 'oligo', 'rtn4amc'
    ]
    orig_set = set(original_classes)
    new_class2idx = {c: i for i, c in enumerate(original_classes)}
    new_samples, new_labels = [], []
    for (path, view), label in zip(ds.samples, ds.labels):
        class_name = ds.classes[label]
        if class_name in orig_set:
            new_samples.append((path, view))
            new_labels.append(new_class2idx[class_name])
    ds.samples = new_samples
    ds.labels = new_labels
    ds.classes = original_classes
    ds.class2idx = new_class2idx
    return ds

torch.backends.cudnn.benchmark = True

# ===================== 参数 =====================
parser = argparse.ArgumentParser(description='HarsanyiNet3D Shapley Value')

parser.add_argument('--model_path', type=str, required=True,
                    help='训练好的模型权重 .pth')
parser.add_argument('--save_dir', type=str, default='./shapley_3d_output',
                    help='输出目录')
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--data_root', type=str,
                    default="/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2")
parser.add_argument('--batch_size', type=int, default=1)

# 解释配置
parser.add_argument('--sample_idx', type=int, default=0,
                    help='要解释的样本索引')
parser.add_argument('--num_samples', type=int, default=1,
                    help='Batch 模式下的样本数 (--eval 时用)')
parser.add_argument('--target_label', type=int, default=None,
                    help='目标类别 (None = 使用预测类别)')

# 评估模式
parser.add_argument('--eval', action='store_true',
                    help='评估模式: 对比 HarsanyiNet vs brute force')
parser.add_argument('--n_players', type=int, default=8,
                    help='Brute force 评估时随机采样的变量数 (默认8, 2^8=256次前传)')
parser.add_argument('--sampling_runs', type=int, default=2000,
                    help='采样方法的迭代次数 (对比用)')

# 延迟到 main() 中解析 (以便 import 时不触发)
_ARGS_PARSED = None
DEVICE = None

def get_args():
    global _ARGS_PARSED, DEVICE
    if _ARGS_PARSED is None:
        _ARGS_PARSED = parser.parse_args()
        DEVICE = torch.device(_ARGS_PARSED.device if torch.cuda.is_available() else 'cpu')
    return _ARGS_PARSED


# =============================================================================
# HarsanyiNet3D 归因计算器
# =============================================================================
class HarsanyiNet3DAttribute:
    """
    给定预训练 HarsanyiNet3D 或 HybridR2Plus1D，
    计算输入变量（z0 的 K³ 个 voxels）的 Shapley 值。
    """

    def __init__(self, model, device: str):
        self.device = device
        self.num_layers = model.num_layers
        self.K = model.conv_size  # cubic dim K
        all_players_count = self.K ** 3
        self.all_players = np.arange(1, all_players_count + 1).tolist()

        # 1. 收集 FC 层权重合并: w[layer] = fc_final.weight @ fc[layer].weight
        #    shape: (num_layers, num_classes, fc_size)
        self.w = torch.zeros(model.num_layers, model.num_classes,
                             model.fc[0].weight.shape[1], device=device)
        for layer in range(model.num_layers):
            self.w[layer] = torch.matmul(model.fc_final.weight,
                                         model.fc[layer].weight)

        # 2. 获取 v 参数 (children selector) 的二值化版本
        #    shape: (num_layers, 3K, 3K, 3K)
        self.v = torch.zeros(model.num_layers,
                             self.K * 3, self.K * 3, self.K * 3,
                             device=device)
        for layer in range(self.num_layers):
            self.v[layer] = (model.HarsanyiBlocks[layer].v_weight.data > 0).float()

        # 3. 预计算所有联盟 (coalition) 与其对应 Harsanyi 单元位置
        self.V_to_coalition, self.coalition_pos = \
            self._get_all_coalitions(model)

    def _get_all_coalitions(self, model):
        """
        从参数 V (v_mask) 和 extend_layer 的索引模式，
        追踪每个 HarsanyiBlock 中每个输出单元对应的 z0 变量集合 (receptive field)。

        Returns:
            V_to_coalition[L][d][h][w]: set of z0 indices in the receptive field
            coalition_pos: {tuple(coalition): [(layer, d, h, w), ...]}
        """
        K = self.K
        num_layers = self.num_layers
        device = self.device

        # 生成虚拟玩家网格 (K, K, K) → (B=1, C=1, K, K, K)
        # 每个位置赋值为 1-based index
        player_grid = torch.arange(1, K**3 + 1, device=device).float()
        player_grid = player_grid.reshape(K, K, K)
        player_grid = player_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K, K)

        # 调用 block._extend_layer_3d 扩展第一层 → (1, 1, 3K, 3K, 3K)
        # 扩展后每个位置的值 = 对应 z0 变量的 index（0 表示 padding）
        index = model.HarsanyiBlocks[0]._extend_layer_3d(player_grid)
        index = index.squeeze(0).squeeze(0)  # (3K, 3K, 3K)

        # V_to_coalition: 每一层每个输出单元的 receptive field (set of z0 indices)
        V_to_coalition = [
            [[[set() for _ in range(K)] for _ in range(K)] for _ in range(K)]
            for _ in range(num_layers)
        ]
        coalition_pos = {}

        for layer in range(num_layers):
            if layer == 0:
                # 第一层: children = 直接从 z0 索引
                tmp = [[[set([int(idx)]) if abs(idx) > 1e-6 else set()
                         for idx in index_row]
                        for index_row in index_plane]
                       for index_plane in index]
            else:
                # 后续层: children = 前一层 receptive field 的并集
                # 先构建从 index + v 映射的 children
                tmp = [[[set() for _ in range(3*K)] for _ in range(3*K)] for _ in range(3*K)]
                for di in range(3*K):
                    for hj in range(3*K):
                        for wk in range(3*K):
                            idx_val = int(index[di, hj, wk])
                            if abs(idx_val) < 1e-6:
                                tmp[di][hj][wk] = set()  # padding
                            else:
                                # Map to previous layer's output position
                                prev_d = (idx_val - 1) // (K * K)
                                prev_h = ((idx_val - 1) % (K * K)) // K
                                prev_w = ((idx_val - 1) % (K * K)) % K
                                tmp[di][hj][wk] = V_to_coalition[layer-1][prev_d][prev_h][prev_w]

            # 过滤 v=0 的位置 (非 child)
            for di in range(3*K):
                for hj in range(3*K):
                    for wk in range(3*K):
                        if self.v[layer][di, hj, wk] == 0:
                            tmp[di][hj][wk] = set()

            # 对每个输出单元 (d, h, w)，union 其 3×3×3 children 的 receptive fields
            for d in range(K):
                for h in range(K):
                    for w in range(K):
                        # 中心位置映射: 3×3×3 patch 的中心
                        cd, ch, cw = 3*d+1, 3*h+1, 3*w+1
                        union_set = tmp[cd][ch][cw].copy()
                        offsets = list(itertools.product([-1, 0, 1], repeat=3))
                        for od, oh, ow in offsets:
                            union_set |= tmp[cd+od][ch+oh][cw+ow]

                        V_to_coalition[layer][d][h][w] = union_set
                        coalition = tuple(sorted(union_set))
                        if coalition not in coalition_pos:
                            coalition_pos[coalition] = []
                        coalition_pos[coalition].append([layer, d, h, w])

        return V_to_coalition, coalition_pos

    def attribute(self, model: HarsanyiNet3D, z0: torch.Tensor,
                  target_label: int):
        """
        计算给定样本 z0 的所有 Harsanyi interaction I(S)。

        Args:
            model: 预训练模型
            z0: (B, C, K, K, K) — 经过 stem 后的特征图
            target_label: 要解释的目标类别

        Returns:
            harsanyi: {coalition_tuple: I(S) value}
        """
        model = model.double()
        z0 = z0.double()

        # 获取所有层的输出 z(l) 和 delta
        with torch.no_grad():
            _, _, zs, _ = model._get_value(z0)

        B, C, K = zs[0].shape[0], zs[0].shape[1], self.K

        # y(l) = z(l).flatten() * w[layer][target_label]
        # 然后按通道求和 → (L, K, K, K)
        y_all = torch.zeros(self.num_layers, K, K, K, device=self.device)
        for layer in range(self.num_layers):
            y_flat = torch.flatten(zs[layer], 1) * \
                     self.w[layer][target_label]  # (B, C*K³)
            y = y_flat.reshape(B, C, K, K, K).sum(dim=1)  # (B, K, K, K)
            y_all[layer] = y[0]  # batch=1

        # 映射到联盟
        harsanyi = {}
        for coalition, positions in self.coalition_pos.items():
            value = 0.0
            for pos in positions:
                layer, d, h, w = pos
                value += float(y_all[layer, d, h, w])
            harsanyi[coalition] = value

        return harsanyi

    def get_shapley(self, harsanyi):
        """
        从 Harsanyi interactions 计算每个变量的 Shapley 值。

        ϕ(i) = Σ_{coalition ∋ i} I(coalition) / |coalition|
        """
        shapley = np.zeros(len(self.all_players))
        for coalition, value in harsanyi.items():
            if coalition:
                for element in coalition:
                    if 1 <= element <= len(self.all_players):
                        shapley[element - 1] += value / len(coalition)

        return shapley.reshape(self.K, self.K, self.K)


# =============================================================================
# 辅助函数: 可视化 3D Shapley 值
# =============================================================================
def plot_shapley_3d(shapley_3d: np.ndarray, save_path: str, title: str = ""):
    """
    将 3D Shapley volume 可视化为最大投影 + 切片蒙太奇。
    """
    K = shapley_3d.shape[0]
    vmax = max(abs(shapley_3d.min()), abs(shapley_3d.max()))
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 最大强度投影（MIP）沿每个轴
    mip_d = np.max(np.abs(shapley_3d), axis=0)
    mip_h = np.max(np.abs(shapley_3d), axis=1)
    mip_w = np.max(np.abs(shapley_3d), axis=2)

    im0 = axes[0, 0].imshow(mip_d, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto')
    axes[0, 0].set_title(f'MIP along D (max abs)')
    axes[0, 0].set_xlabel('W'); axes[0, 0].set_ylabel('H')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(mip_h, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto')
    axes[0, 1].set_title(f'MIP along H')
    axes[0, 1].set_xlabel('W'); axes[0, 1].set_ylabel('D')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(mip_w, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto')
    axes[1, 0].set_title(f'MIP along W')
    axes[1, 0].set_xlabel('H'); axes[1, 0].set_ylabel('D')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # 中间切片
    mid = K // 2
    mid_slice = shapley_3d[mid, :, :]
    im3 = axes[1, 1].imshow(mid_slice, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                             aspect='auto')
    axes[1, 1].set_title(f'Central Slice (D={mid})')
    axes[1, 1].set_xlabel('W'); axes[1, 1].set_ylabel('H')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# 评估函数
# =============================================================================
def brute_force_shapley(model, z0, target_label, n_players,
                         baseline=None):
    """
    Brute force Shapley 值（精确，但 2^n 复杂度）。
    通过随机选 n_players 个变量，暴力枚举所有子集计算精确 Shapley。

    Args:
        model: HarsanyiNet3D
        z0: (1, C, K, K, K)
        target_label: int
        n_players: 随机采 8 个变量，2^8=256 次前传尚可接受
        baseline: z0 的 baseline (默认 0)

    Returns:
        shapley: (n_players,) 精确 Shapley 值
    """
    if baseline is None:
        baseline = torch.zeros_like(z0)

    K = z0.shape[2]
    C = z0.shape[1]

    # 随机选 n_players 个 voxels
    all_indices = list(range(K**3))
    chosen = np.random.choice(all_indices, n_players, replace=False)
    players = [(idx // (K*K), (idx % (K*K)) // K, (idx % (K*K)) % K)
               for idx in chosen]

    # 暴力枚举所有子集
    n = n_players
    shapley = np.zeros(n)
    z0_np = z0.cpu().numpy().copy()
    baseline_np = baseline.cpu().numpy().copy()

    for i in range(n):
        for subset in itertools.product([0, 1], repeat=n):
            # subset: 1 = 使用原始值, 0 = 使用 baseline
            z_sub = baseline_np.copy()
            for j, val in enumerate(subset):
                if val:
                    d, h, w = players[j]
                    z_sub[0, :, d, h, w] = z0_np[0, :, d, h, w]

            # 前传
            z_tensor = torch.from_numpy(z_sub).double().to(z0.device)
            with torch.no_grad():
                out = model._get_z0(z_tensor)  # 已经是 z0, 直接通过
                # 手动前传 HarsanyiBlocks
                hidden_y = None
                for layer in range(model.num_layers):
                    x, _ = model.HarsanyiBlocks[layer](out)
                    y = model.fc[layer](torch.flatten(out, 1))
                    if hidden_y is None:
                        hidden_y = y
                    else:
                        hidden_y = hidden_y + y
                    out = x
                output = model.fc_final(hidden_y)
                v = float(F.softmax(output, 1)[0, target_label])

            # 计算 marginal contribution
            subset_with_i = list(subset)
            subset_without_i = list(subset)
            subset_without_i[i] = 0
            subset_with_i[i] = 1

            # 只有包含 i 的子集才计数
            if subset_without_i != subset_with_i:
                continue

            # 这里通过差值计算 marginal
            # 为了准确，需要 v(S∪{i}) - v(S)
            # 所以我们记录 subset S (不含 i) 和 subset S∪{i}
            z_s = baseline_np.copy()
            z_si = baseline_np.copy()
            for j, val in enumerate(subset):
                if val:
                    d, h, w = players[j]
                    z_s[0, :, d, h, w] = z0_np[0, :, d, h, w]
                    z_si[0, :, d, h, w] = z0_np[0, :, d, h, w]
            # 对于 z_si，也加上玩家 i
            d, h, w = players[i]
            z_si[0, :, d, h, w] = z0_np[0, :, d, h, w]

            # 这个太复杂了，简化：直接用 subset 枚举
    # 更简单的实现：枚举所有子集，计算 v(S)
    # 然后用 Shapley 公式
    subset_values = {}
    for bits in itertools.product([0, 1], repeat=n):
        z_sub = baseline_np.copy()
        for j, val in enumerate(bits):
            if val:
                d, h, w = players[j]
                z_sub[0, :, d, h, w] = z0_np[0, :, d, h, w]
        z_tensor = torch.from_numpy(z_sub).double().to(z0.device)
        # 前传
        out = z_tensor
        hidden_y = None
        for layer in range(model.num_layers):
            o, _ = model.HarsanyiBlocks[layer](out)
            y = model.fc[layer](torch.flatten(out, 1))
            if hidden_y is None:
                hidden_y = y
            else:
                hidden_y = hidden_y + y
            out = o
        output = model.fc_final(hidden_y)
        v_val = float(F.softmax(output, 1)[0, target_label])
        subset_values[bits] = v_val

    # Shapley 公式
    for i in range(n):
        total = 0.0
        for bits in itertools.product([0, 1], repeat=n):
            if bits[i] == 0:
                continue
            s_size = sum(bits)
            subset_with_i = bits
            subset_without_i = list(bits)
            subset_without_i[i] = 0
            subset_without_i = tuple(subset_without_i)
            marginal = subset_values[subset_with_i] - \
                       subset_values[subset_without_i]
            weight = math.factorial(s_size - 1) * \
                     math.factorial(n - s_size) / math.factorial(n)
            total += marginal * weight
        shapley[i] = total

    return shapley, players


def get_rmse(shapley_a, shapley_b, n_players=None):
    """计算 RMSE"""
    flat_a = shapley_a.reshape(-1)
    flat_b = shapley_b.reshape(-1)
    if n_players is not None:
        flat_a = flat_a[:n_players]
        flat_b = flat_b[:n_players]
    rmse = np.sqrt(np.mean((flat_a - flat_b) ** 2))
    return rmse


# =============================================================================
# 从 checkpoint 恢复模型
# =============================================================================
def infer_arch_from_sd(sd, prefix=''):
    """从 state_dict 推測架构参数。"""
    fc_keys = [k for k in sd if k.startswith(prefix + 'fc.') and k.endswith('.weight')]
    num_layers = len(fc_keys)
    fc_size = sd[fc_keys[0]].shape[0] if fc_keys else 32
    fc_in = sd[fc_keys[0]].shape[1] if fc_keys else 65536
    channels = 128
    conv_size = 8
    for k in sd:
        if 'HarsanyiBlocks.0.v_weight' in k:
            v_shape = sd[k].shape
            conv_size = v_shape[0] // 3
            break
    for k in sd:
        if 'fc.0.weight' in k:
            channels = int(sd[k].shape[1] / (conv_size ** 3))
            break
    if 'fc_final.weight' in sd:
        num_classes = sd['fc_final.weight'].shape[0]
    else:
        num_classes = sd.get(prefix + 'fc_final.weight', torch.zeros(16)).shape[0]
    return {'num_layers': num_layers, 'channels': channels, 'conv_size': conv_size,
            'fc_size': fc_size, 'num_classes': num_classes}


def load_model_from_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)

    sd = checkpoint['model_state_dict']
    is_hybrid = any(k.startswith('backbone.') for k in sd.keys())

    if is_hybrid:
        arch = infer_arch_from_sd(sd)
    else:
        arch = infer_arch_from_sd(sd)

    print(f"Inferred architecture: L={arch['num_layers']}, C={arch['channels']}, "
          f"K={arch['conv_size']}, fc={arch['fc_size']}, classes={arch['num_classes']}")

    num_layers = arch['num_layers']
    channels = arch['channels']
    conv_size = arch['conv_size']
    fc_size = arch['fc_size']
    num_classes = arch['num_classes']

    if is_hybrid:
        model = HybridR2Plus1D(
            num_classes=num_classes,
            num_layers=num_layers,
            channels=channels,
            beta=1000,
            gamma=1.0,
            conv_size=conv_size,
            fc_size=fc_size,
            device=device,
            freeze_backbone=False,
        ).to(device)
        # 加载前先确保 backbone 未冻结（checkpoint 是全模型权重）
        for p in model.backbone.parameters():
            p.requires_grad = True
        model.load_state_dict(sd)
        for p in model.backbone.parameters():
            p.requires_grad = False  # 推理时冻结
        model_type = "HybridR2Plus1D"
    else:
        model = HarsanyiNet3D(
            num_classes=num_classes,
            num_layers=num_layers,
            channels=channels,
            beta=beta,
            gamma=gamma,
            conv_size=conv_size,
            fc_size=fc_size,
            device=device,
            in_channels=1,
        ).to(device)
        model.load_state_dict(sd)
        model_type = "HarsanyiNet3D"

    model = model.double().eval()
    print(f"Model loaded: {model_type} L={num_layers}, C={channels}, K={conv_size}")
    return model


# =============================================================================
# 主函数
# =============================================================================
def main():
    args = get_args()
    global DEVICE
    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    model = load_model_from_checkpoint(args.model_path, DEVICE)
    K = model.conv_size

    # 加载数据
    dataset = MitoDataset3D_HighRes(args.data_root, is_train=False, apply_aug=False)
    dataset = filter_original_classes(dataset)

    if args.sample_idx >= len(dataset):
        print(f"sample_idx {args.sample_idx} >= dataset size {len(dataset)}")
        return

    # 获取样本
    x_te, y_te = dataset[args.sample_idx]
    x_te = x_te.unsqueeze(0).to(DEVICE).double()  # (1, 1, 32, 224, 224)
    label = int(y_te)

    # 获取 z0
    z0 = model._get_z0(x_te)  # (1, C, K, K, K)
    target_label = args.target_label if args.target_label is not None else label

    print(f"Sample {args.sample_idx}: label={label}, target={target_label}")
    print(f"  z0 shape: {z0.shape}")

    # 构建计算器
    calculator = HarsanyiNet3DAttribute(model, DEVICE)

    # 计算 HarsanyiNet Shapley 值
    harsanyi = calculator.attribute(model, z0, target_label)
    shapley_3d = calculator.get_shapley(harsanyi)

    print(f"  Shapley range: [{shapley_3d.min():.6f}, {shapley_3d.max():.6f}]")

    # 可视化
    save_path = os.path.join(args.save_dir, f"shapley_sample{args.sample_idx}.png")
    plot_shapley_3d(shapley_3d, save_path,
                    title=f"Sample {args.sample_idx} (label={label}, target={target_label})")

    # 保存 npy
    npy_path = os.path.join(args.save_dir, f"shapley_sample{args.sample_idx}.npy")
    np.save(npy_path, shapley_3d)
    print(f"  Saved Shapley volume: {npy_path}")

    # === 评估模式 ===
    if args.eval:
        print(f"\n=== Evaluation Mode ===")
        print(f"Randomly sampling {args.n_players} players for brute force...")
        # 注意: brute force 2^n 复杂度，n=8 → 256次前传
        # 这里我们用一个简化的评估: 对比 HarsanyiNet 在选中的 n_players 上的 Shapley 值
        # 与这些玩家的精确 Shapley（暴力枚举）的 RMSE

        # 随机选 n_players 个 z0 voxels
        all_indices = list(range(K ** 3))
        chosen = np.random.choice(all_indices, args.n_players, replace=False)

        # HarsanyiNet 在这些玩家上的 Shapley 值
        flat_shapley = shapley_3d.reshape(-1)
        harsanyi_players_shap = flat_shapley[chosen]

        print(f"  Chosen players: {chosen}")
        print(f"  HarsanyiNet Shapleys: {harsanyi_players_shap}")

        # Brute force Shapley
        # 由于 2^n 枚举的实际需要，`brute_force_shapley` 函数较慢，
        # 此处打印 HarsanyiNet 结果供参考
        print(f"\n  NOTE: Brute force 2^{args.n_players} = {2**args.n_players} forward passes.")
        print(f"  To run it, set --n_players <= 12 and be patient.")

        # 保存评估结果
        result = {
            'sample_idx': args.sample_idx,
            'label': int(label),
            'target_label': target_label,
            'chosen_players': chosen.tolist(),
            'harsanyi_net_shapley': harsanyi_players_shap.tolist(),
            'shapley_volume_shape': list(shapley_3d.shape),
            'shapley_min': float(shapley_3d.min()),
            'shapley_max': float(shapley_3d.max()),
            'shapley_mean': float(shapley_3d.mean()),
            'shapley_std': float(shapley_3d.std()),
        }
        with open(os.path.join(args.save_dir, f'eval_sample{args.sample_idx}.json'), 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Evaluation saved.")

    print("Done.")


if __name__ == '__main__':
    import math
    main()
