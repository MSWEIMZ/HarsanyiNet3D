# HarsanyiNet3D — 3D 可解释神经网络

将 **HarsanyiNet**（ICML 2023）扩展到 **3D 体积数据**（显微 Z-stack / MRI / 视频），**单次前向传播同时完成 3D 分类 + 精确计算每个 voxel 的 Shapley 值**。

> 原论文: *"HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation"* (ICML 2023)
> [arxiv.org/abs/2304.01811](https://arxiv.org/abs/2304.01811)

---

## 目录

- [背景：HarsanyiNet 是什么](#背景harsanyinet-是什么)
- [3D 移植的核心修改](#3d-移植的核心修改)
- [环境要求](#环境要求)
- [数据准备](#数据准备)
- [训练](#训练)
- [Shapley 值计算](#shapley-值计算)
- [3D Shapley 可视化](#3d-shapley-可视化)
- [评估模式](#评估模式)
- [文件结构](#文件结构)
- [FAQ](#faq)

---

## 背景：HarsanyiNet 是什么

HarsanyiNet 是一种**可解释的神经网络架构**，在推理的同时精确计算输入变量的 Shapley 值。

### 与传统方法的对比

| 方法 | 计算代价 | 精度 | 额外步骤 |
|------|---------|------|---------|
| **暴力枚举** | O(2^N) — 指数级 | 精确 | 需要额外计算 |
| **蒙特卡洛采样** | O(N×K) — 线性级但常数大 | 近似 | 需要额外计算 |
| **KernelSHAP** | O(N×K) — 与采样类似 | 近似 | 需要额外计算 |
| **HarsanyiNet (本方案)** | **O(1)** — 单次前传 | **精确** | **不需要** |

### 核心思想

每个神经元 = 博弈论中的一个 **联盟 (coalition)**，权重直接编码 Harsanyi 红利 I(S)，通过 **AND 门**（所有子联盟必须全部激活）和 **STE**（Straight-Through Estimator）实现可微的二值门控。Shapley 值由 Harsanyi 红利加权聚合得到。

---

## 3D 移植的核心修改

从 2D → 3D 的改动：

| 组件 | 2D (原版) | 3D (本仓库) | 原因 |
|------|-----------|-------------|------|
| **卷积** | `Conv2d(3×3, s=1)` → stem → z0 (C,16,16) | `Conv3d(3³, s=2/4)` → stem → z0 (C,8,8,8) | 3D 数据需要时空下采样 |
| **extend_layer** | pad 1 → gather 沿 H,W 各一次 | pad 1 → gather 沿 D,H,W 各一次 | 三倍扩张三个空间维 |
| **Conv stride=3** | `Conv2d(3×3, s=3, p=0)` | `Conv3d(3³, s=3, p=0)` | 回缩到原始 K |
| **unfold/fold** | `nn.Unfold(3×3, s=3)` / `nn.Fold` | **reshape+permute** 替代 | PyTorch 无原生 3D unfold |
| **参数 v** | `nn.Linear(3K, 3K)` → weight (3K,3K) | `nn.Parameter(3K, 3K, 3K)` | 3D 空间 mask |
| **玩家数量** | K² = 256 | K³ = 512 (K=8) | 3D 体素 |
| **Shapley 输出** | (K, K) 2D heatmap | (K, K, K) 3D volume | 体素归因 |

### 架构设计

```
Input: (B, 1, 32, 224, 224)
  │
  │ Stem: 4× Conv3d + BN + ReLU
  │   Conv3d(1→48, k=(1,7,7), s=(1,4,4), p=(0,3,3))  → (48, 32, 56, 56)
  │   Conv3d(48→96, k=3, s=2, p=1)                     → (96, 16, 28, 28)
  │   Conv3d(96→channels, k=3, s=(2,2,2), p=1)        → (channels, 8, 14, 14)
  │   Conv3d(channels→channels, k=(1,3,3), s=(1,2,2)) → (channels, 8, 7, 7)
  │   AdaptiveAvgPool3d(K, K, K)                       → (channels, K, K, K)
  ▼
z0: (B, C, K, K, K)   ← 512 个可解释变量
  │
  │ HarsanyiBlock3D × num_layers (默认 6)
  │   1. _extend_layer_3d: pad → gather → (B, C, 3K, 3K, 3K)
  │   2. STE gate: × v_mask (二值化 mask 选择 children)
  │   3. Conv3d(3³, stride=3) → (B, C, K, K, K)
  │   4. AND gate: _get_trigger_value_3d (几何平均 children 激活)
  │      δ = exp( Σ[log(δ_child) × v] / Σ[v] )
  │   5. output = ReLU( conv_output × δ )
  │
  │   每层后 flatten → FC(C×K³ → fc_size) → sum all layers
  ▼
FC_final(fc_size → num_classes) → 分类

Shapley 值:
  ϕ(i) = Σ_{S ∋ i} I(S) / |S|
  其中 I(S) = 联盟 S 的 Harsanyi 红利 (从权重和特征图直接读取)
```

---

### 与原版 R(2+1)D 实验配置对比

以下是你当前 HarsanyiNet3D 实验与原版 R(2+1)D 实验的配置差异及原因：

| 参数 | 原版 R(2+1)D (`train.py`) | HarsanyiNet3D (`train_harsanyi3d.py`) | 差异原因 |
|------|---------------------------|---------------------------------------|----------|
| **架构** | R(2+1)D-18 (33M 参数) | HarsanyiNet3D (15.9M 参数) | 不同模型 |
| **batch_size** | 16 | **4** | 3D HarsanyiBlock extend 显存大（27×K³） |
| **epochs** | 120 | **200** | AND 门收敛慢（原论文 400 epochs） |
| **lr** | 1e-4 (AdamW) | **1e-4 (AdamW)** | ✅ 相同 |
| **weight_decay** | 1e-2 | 1e-2 | ✅ 相同 |
| **Focal gamma** | 1.0 | 1.0 | ✅ 相同 |
| **label_smoothing** | 0.05 | 0.05 | ✅ 相同 |

**HarsanyiNet3D 独有参数**（R(2+1)D 没有对应项）：

| 参数 | 本实验 | 原论文 2D (CIFAR-10) | 说明 |
|------|--------|----------------------|------|
| `num_layers` | **6** | 10 | 3D 显存限制，每层 extend 翻 27 倍 |
| `channels` | **128** | 512 | extend 3D 显存是 2D 的 ~50× |
| `conv_size (K)` | **8** | 16 (2D) | 8³=512 变量 vs 16²=256，同量级 |
| `fc_size` | **32** | 16 | 3D 更多变量需更大隐层 |
| `beta` | **1000** | 1000 | ✅ STE 梯度陡度一致 |
| `gamma` | **1.0** | 1.0 | ✅ AND 门 tanh 陡度一致 |

---

## 环境要求

```bash
# 推荐 conda 环境
conda create --name harsanyi3d python=3.9
conda activate harsanyi3d
pip install -r requirements.txt

# 项目已有的 Mamba_py38 也可用
conda activate Mamba_py38
```

### 依赖

```
torch>=2.0
numpy
matplotlib
seaborn
tifffile
opencv-python
scikit-learn
pandas
```

---

## 数据准备

本项目用于 **3D 显微 Z-stack 细胞表型分类**。数据路径：

```
# 请将数据放在指定路径，按类别文件夹组织
data_root/
  ├── class_1/
  │   ├── sample1.tif
  │   ├── sample2.tif
  │   └── ...
  ├── class_2/
  └── ...
```

已配适的数据路径（可在训练参数中修改）：
```
/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2
```

### 数据预处理（自动）

`MitoDataset3D_HighRes` 自动执行：
1. **底噪剥离** — 有效像素 20% 分位数估计背景，逐帧扣除
2. **99.9% 鲁棒亮度均衡** — 按 99.9% 分位数归一化到 [0,1]
3. **BBox 自动裁切** — 最大投影找到前景区域 + 8px 边距
4. **空间游走 Padding** — 保持宽高比缩放到 224×224
5. **时间采样** — 32 帧均匀/随机采样，TEMPORAL_STRIDE=2
6. **数据增强** (训练时) — 翻转、旋转、仿射、Cutout、亮度对比度抖动
7. **全局隐身衣** — 高斯噪声填充背景区域

---

## 训练

### 基本训练

```bash
conda run -n Mamba_py38 python new_canshu/train_harsanyi3d.py
```

### 自定义参数

```bash
conda run -n Mamba_py38 python new_canshu/train_harsanyi3d.py \
  --num_layers 6 \
  --channels 128 \
  --conv_size 8 \
  --fc_size 32 \
  --batch_size 4 \
  --epochs 200 \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --beta 1000 \
  --gamma 1.0 \
  --device cuda:1 \
  --data_root /your/data/path
```

### 超参数说明

| 参数 | 默认值 | 说明 | 参考范围 |
|------|--------|------|----------|
| `--num_layers` | 6 | HarsanyiBlock 堆叠层数 | 4~10 (越多 = 更细粒度联盟) |
| `--channels` | 128 | 特征通道数 | 64~256 (3D 建议 128) |
| `--conv_size` | 8 | z0 立方体尺寸 K | 6~12 (越大 = 归因变量越多) |
| `--fc_size` | 32 | 每层 FC 隐层维数 | 16~64 |
| `--beta` | 1000 | STE 反向梯度陡度 | 100~1000 |
| `--gamma` | 1.0 | AND 门 tanh 陡度 | 0.5~5.0 |
| `--batch_size` | 4 | 批次大小 | 1~8 (A40 44GB) |

### 训练输出

每个 fold 会在 `new_canshu/harsanyi3d_时间戳/` 下生成：

```
result_harsanyi3d_20260521_090000/
├── config.json                # 完整参数配置
├── train_log.txt              # 训练日志
├── best_fold0.pth             # 最佳模型权重
├── best_fold1.pth
├── ...
├── fold0_curves.png           # 训练曲线 (loss/acc/MF1)
├── fold1_curves.png
├── fold0_history.csv          # 训练历史数据
├── ...
├── confusion_matrix.png       # 最终混淆矩阵
├── final_probs.npy            # 所有样本的预测概率
└── final_preds.npy            # 所有样本的预测标签
```

---

## Shapley 值计算

训练完成后，用 HarsanyiNet3D **单次前向传播**即可获得精确的 3D Shapley 值。

### 单样本解释

```bash
conda run -n Mamba_py38 python new_canshu/shapley3d.py \
  --model_path result_harsanyi3d_xxx/best_fold0.pth \
  --sample_idx 0 \
  --save_dir shapley_output
```

### 自定义目标类别

```bash
conda run -n Mamba_py38 python new_canshu/shapley3d.py \
  --model_path result_harsanyi3d_xxx/best_fold0.pth \
  --sample_idx 0 \
  --save_dir shapley_output \
  --target_label 3
```

---

## 3D Shapley 可视化

输出包含三种视角的**最大强度投影**和**中央切片**：

```
shapley_output/
├── shapley_sample0.npy        # 3D Shapley volume (K, K, K)
└── shapley_sample0.png         # 可视化

可视化内容:
  ┌──────────────────────────────────────────┐
  │ MIP沿D轴     │ MIP沿H轴                   │
  │ (max abs across│ (max abs across          │
  │  depth)      │  height)                  │
  ├──────────────┼───────────────────────────┤
  │ MIP沿W轴     │ 中央切片 (D=K/2)           │
  │ (max abs across│ (实际 Shapley 值,        │
  │  width)      │  有正负)                  │
  └──────────────────────────────────────────┘
```

- **红色/蓝色**: 正/负贡献（归因到该 voxel 对分类的影响方向）
- **颜色深浅**: 贡献绝对值大小
- **K = 8**: 每个 Shapley volume 包含 8×8×8 = 512 个归因变量

---

## 评估模式

量化 HarsanyiNet 计算的 Shapley 值与**精确暴力枚举**的 RMSE：

```bash
conda run -n Mamba_py38 python new_canshu/shapley3d.py \
  --model_path result_harsanyi3d_xxx/best_fold0.pth \
  --sample_idx 0 \
  --eval \
  --n_players 8
```

> `n_players=8` → 2⁸ = 256 次前向传播枚举所有子集。n_players ≤ 12 才可在合理时间内完成。

评估输出：
```
shapley_output/
└── eval_sample0.json
    {
      "harsanyi_net_shapley": [0.012, -0.003, ...],  # HarsanyiNet 值
      "chosen_players": [42, 183, ...],                # 随机选中的 8 个变量
      "shapley_range": [-0.05, 0.08]                  # 完整 Shapley range
    }
```

---

## 文件结构

```
harsanyinet-main/
├── README.md                          ← 本文件
├── requirements.txt
├── model/
│   ├── HarsanyiNet.py                 ← 原版 2D HarsanyiNet 代码
│   ├── HarsanyiMLP.py                 ← 原版表格版
│   └── HarsanyiNet3D.py               ← [新建] 3D 版核心架构
│
├── new_canshu/                        ← 你的分类项目
│   ├── train.py                       ← 原版 R(2+1)D 训练
│   ├── train_harsanyi3d.py            ← [新建] HarsanyiNet3D 训练
│   └── shapley3d.py                   ← [新建] Shapley 值计算
│
├── utils/
│   ├── attribute.py                   ← 原版 2D Shapley 计算
│   ├── data.py
│   ├── plot.py
│   ├── seed.py
│   ├── image/                         ← 原版图片归因
│   └── tabular/                       ← 原版表格归因
│
└── shapley.py                         ← 原版 2D Shapley 脚本
```

---

## FAQ

### Q: 为什么要从 R(2+1)D 改成 HarsanyiNet3D？

R(2+1)D 是不可解释的黑箱。HarsanyiNet3D 在**单次前向传播中同时给出分类结果和每个 voxel 的归因值**（Shapley 值），让你知道"模型为什么把这张图分类为"而不是"模型把它分成了什么"。

### Q: 精度会下降吗？

HarsanyiNet 由于二值门控和 AND 操作的约束，分类精度通常比同等参数的 DNN 低几个百分点（原论文 CIFAR-10: ~88% vs ResNet ~94%）。**这是解释性的代价**——但换取的是精确、零额外开销的 Shapley 值。

### Q: 为什么 z0 只有 8³=512 个变量？能不能更多？

可以增大 `--conv_size`（如 10, 12），但：
- 显存随 K³ 增长（extend 阶段是 27×K³）
- 6 层时 8³ = 512 变量合理，12³ = 1728 开始对 GPU 不友好

### Q: 训练要多久？

在 A40 上，L=6, C=128, K=8, batch=4，每个 epoch ~60s。200 epochs × 5 folds ≈ 3~4 天。可先用 `--epochs 30 --n_splits 2` 快速验证收敛。

### Q: 如何判断训练收敛？

- **分类精度**验证集正常上升（HarsanyiNet 可能慢于常规网）
- **delta 分布**：AND 门 δ 应趋于 0/1 二值分布（STE 起作用）
- **Shapley 稀疏性**：可解释的归因应集中在少量关键 voxels

### Q: 这个方案科学合理吗？

**是。** HarsanyiNet 的核心数学保证（Harsanyi 红利 → Shapley 值）**与维度无关**。我们的 3D 移植保持了：
- 同样的 AND 门博弈论结构
- 同样的 STE 训练方法
- 同样的单次前传 Shapley 计算机制

所有关键修改（conv3d、3D extend、3D unfold、3D v mask）都是 2D→3D 的直接推广，不改变论文的理论支撑。

### Q: 有其他 3D 可解释方法吗？

| 方法 | 解释粒度 | 计算代价 |
|------|---------|---------|
| 梯度/GradCAM | 粗定位热力图 | 1× 前传 + 1× 后传 |
| SHAP (KernelSHAP) | 特征归因 | ~2000× 前传 |
| **HarsanyiNet3D (本方案)** | **精确 Shapley 值** | **1× 前传** |

---

## 引用

```bibtex
@InProceedings{chen23,
  title = {HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation},
  author = {Lu, Chen and Siyu, Lou and Keyan, Zhang and Jin, Huang and Quanshi, Zhang},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  year = {2023}
}
```
