#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HarsanyiNet3D 训练脚本
在原有 R(2+1)D 数据管道基础上替换模型为 HarsanyiNet3D，
单次前向传播同时推理 + 精确计算 3D Shapley 值。

用法:
    # 训练（6层，128通道，K=8）
    python train_harsanyi3d.py --num_layers 6 --channels 128 --conv_size 8

    # 小规模快速验证
    python train_harsanyi3d.py --num_layers 2 --channels 64 --conv_size 6 --epochs 20 --batch_size 2
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, atexit, time, zlib, random, json, cv2, tifffile as tiff
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
import argparse

torch.backends.cudnn.benchmark = True

# ===================== 解析参数 =====================
parser = argparse.ArgumentParser(description='HarsanyiNet3D Training')
# 数据
parser.add_argument('--data_root', type=str,
                    default="/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2")
parser.add_argument('--num_classes', type=int, default=16)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--frames', type=int, default=32)
parser.add_argument('--temporal_stride', type=int, default=2)

# HarsanyiNet3D 架构
parser.add_argument('--num_layers', type=int, default=6,
                    help="HarsanyiBlock 数量")
parser.add_argument('--channels', type=int, default=128,
                    help="特征通道数（原版CIFAR用512，3D显存受限建议128）")
parser.add_argument('--conv_size', type=int, default=8,
                    help="z0 空间尺寸 K（cubic），z0 有 K³ 个可解释变量")
parser.add_argument('--fc_size', type=int, default=32,
                    help="每层全连接隐层维度")

# HarsanyiNet 超参数
parser.add_argument('--beta', type=int, default=1000,
                    help="STE backward 梯度近似陡度（越大越接近真梯度）")
parser.add_argument('--gamma', type=float, default=1.0,
                    help="tanh AND 门陡度（越大越接近二值）")

# 训练
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--mixup_prob', type=float, default=0.5)
parser.add_argument('--label_smoothing', type=float, default=0.05)
parser.add_argument('--focal_gamma', type=float, default=1.0)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--min_epochs', type=int, default=40)

# 系统
parser.add_argument('--device', type=str, default='cuda:1',
                    help="默认用最空闲的 GPU 1")
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--dry_run', action='store_true',
                    help="仅跑一个 batch 验证通路")

# 输出
parser.add_argument('--save_dir', type=str, default=None,
                    help="自动生成 result_harsanyi3d_时间戳")
parser.add_argument('--stem_pretrained', type=str, default=None,
                    help="预训练 stem 权重路径 (由 pretrain_stem.py 生成)")

args = parser.parse_args()

# ===================== 全局设置 =====================
DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
FRAMES_PER_CLIP = args.frames
TEMPORAL_STRIDE = args.temporal_stride
IMG_H = IMG_W = args.img_size
NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
EPOCHS_PER_FOLD = args.epochs
N_SPLITS = args.n_splits if not args.dry_run else 2
RANDOM_SEED = args.seed
MIXUP_PROB = args.mixup_prob
PATIENCE = args.patience
MIN_EPOCHS = args.min_epochs
GRAD_CLIP = 1.0

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")


# ===================== 日志 =====================
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ===================== 绘图工具 =====================
def plot_history(history, save_dir, fold):
    epochs = history['epoch']
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title(f'Fold {fold} Loss'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b--', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.plot(epochs, history['val_mf1'], 'g-', label='Val MF1', linewidth=2)
    plt.title(f'Fold {fold} Metrics'); plt.legend(); plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['lr'], 'm-', label='LR')
    plt.title(f'Fold {fold} LR Schedule'); plt.yscale('log'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold{fold}_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    plt.title('Final Confusion Matrix', fontsize=16)
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


# ===================== Focal Loss =====================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        else:
            targets_one_hot = targets
        targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        probs = F.softmax(inputs, dim=1)
        log_probs = F.log_softmax(inputs, dim=1)
        pt = torch.sum(targets_one_hot * probs, dim=1)
        focal_weight = (1 - pt) ** self.gamma
        if self.weight is not None:
            class_weights = torch.sum(targets_one_hot * self.weight, dim=1)
            focal_weight = focal_weight * class_weights
        loss = -focal_weight * torch.sum(targets_one_hot * log_probs, dim=1)
        return loss.mean()


# ===================== 数据增强 =====================
class FastAug:
    def __init__(self):
        self.do_flip_h = random.random() < 0.5
        self.do_flip_v = random.random() < 0.5
        self.do_affine = random.random() < 0.7
        self.angle = random.uniform(-15, 15)
        self.scale = random.uniform(0.9, 1.1)
        self.do_cutout = random.random() < 0.5
        self.do_brightness = random.random() < 0.5
        self.brightness_factor = random.uniform(0.7, 1.3)
        self.do_contrast = random.random() < 0.5
        self.contrast_factor = random.uniform(0.7, 1.3)

    def __call__(self, video_tensor):
        out = np.zeros_like(video_tensor)
        T, H, W = video_tensor.shape
        if self.do_affine:
            center = (W / 2, H / 2)
            M = cv2.getRotationMatrix2D(center, self.angle, self.scale)
        if self.do_cutout:
            ch, cw = 56, 56
            cy = random.randint(0, max(1, H - ch))
            cx = random.randint(0, max(1, W - cw))
        else:
            cy = cx = ch = cw = 0
        for i in range(T):
            img = video_tensor[i].copy()
            if self.do_flip_h: img = cv2.flip(img, 1)
            if self.do_flip_v: img = cv2.flip(img, 0)
            if self.do_affine:
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            if self.do_brightness:
                img = np.clip(img * self.brightness_factor, 0.0, 1.0)
            if self.do_contrast:
                mean = img.mean()
                img = np.clip((img - mean) * self.contrast_factor + mean, 0.0, 1.0)
            if self.do_cutout:
                img[cy:cy+ch, cx:cx+cw] = 0.0
            out[i] = img
        return out


# ===================== 数据集（完全复用原版） =====================
class MitoDataset3D_HighRes(Dataset):
    def __init__(self, root, is_train=True, apply_aug=True, clip_len=FRAMES_PER_CLIP):
        self.root = root
        self.is_train = is_train
        self.apply_aug = apply_aug and is_train
        self.clip_len = clip_len
        self.classes = sorted([c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))]) if os.path.exists(root) else []
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.views = 1 if is_train else 3
        self.samples, self.labels = self._scan()

    def _scan(self):
        samples, labels = [], []
        if not os.path.exists(self.root): return [], []
        for c in self.classes:
            cdir = os.path.join(self.root, c)
            if not os.path.isdir(cdir): continue
            for f in sorted(os.listdir(cdir)):
                if f.lower().endswith(('.tif', '.tiff')):
                    path = os.path.join(cdir, f)
                    for v in range(self.views):
                        samples.append((path, v))
                        labels.append(self.class2idx[c])
        return samples, labels

    def _process_tif(self, path, view_idx):
        try:
            tif = tiff.imread(path)
        except Exception:
            return np.zeros((1, self.clip_len, IMG_H, IMG_W), np.float32)
        if tif.ndim == 4 and tif.shape[1] == 2: tif = tif[:, 1]
        elif tif.ndim == 4 and tif.shape[-1] == 2: tif = tif[..., 1]
        if tif.ndim == 2: tif = tif[None]
        if tif.ndim != 3 or tif.shape[0] == 0:
            return np.zeros((1, self.clip_len, IMG_H, IMG_W), np.float32)
        T, raw_H, raw_W = tif.shape
        tif = tif.astype(np.float32)
        if tif.max() > 1: tif /= 255.0
        L = self.clip_len
        req_T = (L - 1) * TEMPORAL_STRIDE + 1
        if T < L: indices = np.resize(np.arange(T), L); indices.sort()
        elif T < req_T: indices = np.linspace(0, T - 1, L).astype(int)
        else:
            if self.is_train: start = np.random.randint(0, T - req_T + 1)
            else: start = 0 if view_idx == 0 else ((T - req_T) // 2 if view_idx == 1 else T - req_T)
            indices = np.arange(start, start + req_T, TEMPORAL_STRIDE)
        clip_frames = tif[indices]

        # 底噪剥离 (有效像素20%分位数)
        valid_pixels = clip_frames[clip_frames > 1e-4]
        if len(valid_pixels) > 0:
            bg_mean = np.percentile(valid_pixels, 20)
            bg_std = max(np.std(valid_pixels), 0.001)
        else:
            bg_mean = 0.0; bg_std = 0.001
        clip_frames = np.maximum(clip_frames - bg_mean, 0.0)
        # 99.9% 鲁棒亮度均衡
        p99 = np.percentile(clip_frames, 99.9)
        if p99 > 1e-5: clip_frames = clip_frames / p99
        clip_frames = np.clip(clip_frames, 0.0, 1.0)
        # BBox 提取
        max_proj = np.max(clip_frames, axis=0)
        coords = cv2.findNonZero((max_proj > 0.01).astype(np.uint8))
        if coords is not None and len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords)
            margin = 8
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(raw_W, x + w + margin), min(raw_H, y + h + margin)
        else:
            x1, y1, x2, y2 = 0, 0, raw_W, raw_H
        cropped_clip = clip_frames[:, y1:y2, x1:x2]
        if cropped_clip.size == 0:
            return np.zeros((1, self.clip_len, IMG_H, IMG_W), np.float32)
        _, crop_h, crop_w = cropped_clip.shape
        target_h, target_w = IMG_H, IMG_W
        scale = min(target_h / crop_h, target_w / crop_w)
        new_h = min(int(round(crop_h * scale)), target_h)
        new_w = min(int(round(crop_w * scale)), target_w)
        final_clip = np.zeros((L, target_h, target_w), dtype=np.float32)
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        if self.is_train:
            pad_top = random.randint(0, target_h - new_h)
            pad_left = random.randint(0, target_w - new_w)
        else:
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2
        pad_bottom = target_h - new_h - pad_top
        pad_right = target_w - new_w - pad_left
        for i in range(L):
            resized_frame = cv2.resize(cropped_clip[i], (new_w, new_h), interpolation=interpolation)
            final_clip[i] = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right,
                                                cv2.BORDER_CONSTANT, value=0.0)
        if self.apply_aug:
            final_clip = FastAug()(final_clip)
        # 全局隐身衣
        if self.is_train:
            noise = np.random.normal(loc=0.0, scale=bg_std, size=final_clip.shape)
        else:
            noise_seed = (zlib.crc32(path.encode("utf-8")) + view_idx + RANDOM_SEED) & 0xFFFFFFFF
            noise = np.random.default_rng(noise_seed).normal(loc=0.0, scale=bg_std, size=final_clip.shape)
        noise = np.abs(noise).astype(np.float32)
        final_clip = np.where(final_clip < 1e-4, noise, final_clip)
        final_clip = np.clip(final_clip, 0.0, 1.0)
        return final_clip[np.newaxis, ...]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.from_numpy(self._process_tif(*self.samples[idx])).float(), torch.tensor(self.labels[idx], dtype=torch.long)


# ===================== 模型导入 =====================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.HarsanyiNet3D import HarsanyiNet3D


# ===================== 训练循环 =====================
def train_epoch(model, loader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)
        do_mixup = np.random.rand() < MIXUP_PROB
        if do_mixup:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(x.size(0)).to(device)
            target_a = batch_y
            target_b = batch_y[rand_index]
            x_mixed = lam * x + (1 - lam) * x[rand_index]
            targets_a = F.one_hot(target_a, num_classes=NUM_CLASSES).float()
            targets_b = F.one_hot(target_b, num_classes=NUM_CLASSES).float()
            targets_mixed = lam * targets_a + (1 - lam) * targets_b
        else:
            x_mixed = x
            targets_mixed = batch_y
        optimizer.zero_grad()
        with autocast():
            outputs = model(x_mixed)
            loss = criterion(outputs, targets_mixed)
        prev_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= prev_scale:
            scheduler.step()
        total_loss += loss.item() * batch_y.size(0)
        correct += (outputs.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)
    acc = (correct / total) if total > 0 else 0.0
    return total_loss / len(loader.dataset), acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, labels_list = [], []
    val_criterion = nn.CrossEntropyLoss(weight=criterion.weight)
    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)
        with autocast():
            # TTA: normal + flipped
            outputs = (model(x) + model(torch.flip(x, dims=[4]))) / 2.0
            loss = val_criterion(outputs, batch_y)
        total_loss += loss.item() * batch_y.size(0)
        preds.append(outputs.argmax(1).cpu())
        labels_list.append(batch_y.cpu())
    labels_all = torch.cat(labels_list)
    preds_all = torch.cat(preds)
    return (total_loss / len(labels_all),
            accuracy_score(labels_all, preds_all),
            f1_score(labels_all, preds_all, average='macro', zero_division=0))


# ===================== 类别过滤 =====================
def filter_original_classes(ds):
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


# ===================== 主函数 =====================
def main():
    # 输出目录
    if args.save_dir is None:
        timestamp = time.strftime("harsanyi3d_%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.path.dirname(__file__), timestamp)
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 加载数据
    print("Loading data...")
    tmp_ds = MitoDataset3D_HighRes(args.data_root, is_train=True,
                                   apply_aug=True, clip_len=FRAMES_PER_CLIP)
    tmp_ds = filter_original_classes(tmp_ds)
    if len(tmp_ds) < 5:
        print(f"ERROR: Only {len(tmp_ds)} samples found. Check DATA_ROOT={args.data_root}")
        return
    if len(tmp_ds.classes) != NUM_CLASSES:
        raise ValueError(f"NUM_CLASSES={NUM_CLASSES}, found {len(tmp_ds.classes)} classes")

    X = np.arange(len(tmp_ds))
    y_global = np.array(tmp_ds.labels)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # 日志
    log_path = os.path.join(save_dir, "train_log.txt")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    def _close_log():
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if not log_file.closed: log_file.close()
    atexit.register(_close_log)

    def log(msg): print(msg)

    log("=" * 80)
    log("HarsanyiNet3D Training")
    log(f"  Architecture: L={args.num_layers}, C={args.channels}, K={args.conv_size}")
    log(f"  Params: beta={args.beta}, gamma={args.gamma}, fc_size={args.fc_size}")
    log(f"  Training: epochs={EPOCHS_PER_FOLD}, batch={BATCH_SIZE}, lr={args.lr}")
    log(f"  Data: {args.data_root}")
    log(f"  Save: {save_dir}")
    log("=" * 80)

    # 全验证集（3 views）
    full_val_ds = MitoDataset3D_HighRes(args.data_root, is_train=False, apply_aug=False,
                                        clip_len=FRAMES_PER_CLIP)
    full_val_ds = filter_original_classes(full_val_ds)
    final_file_probs = np.zeros((len(y_global), NUM_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_global)):
        log(f"\n{'='*40} Fold {fold+1}/{N_SPLITS} {'='*40}")

        train_ds = Subset(tmp_ds, train_idx)
        val_idx_set = set(val_idx)
        val_idx_3x = []
        for i, (path, view) in enumerate(full_val_ds.samples):
            file_idx = i // 3
            if file_idx in val_idx_set:
                val_idx_3x.append(i)
        val_ds = Subset(full_val_ds, val_idx_3x)

        # 类别权重
        uni, cnt = np.unique(y_global[train_idx], return_counts=True)
        cw = [0.0] * NUM_CLASSES
        for u, c in zip(uni, cnt):
            cw[u] = len(train_idx) / (NUM_CLASSES * c)
        weights = torch.tensor(cw).float().to(DEVICE)
        criterion = FocalLoss(weight=weights, gamma=args.focal_gamma,
                              label_smoothing=args.label_smoothing)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=len(train_ds) >= BATCH_SIZE,
                                  persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True,
                                persistent_workers=args.num_workers > 0)

        if len(train_loader) == 0:
            raise ValueError(f"Fold {fold}: train_loader empty. Reduce batch_size.")

        # 构建模型
        model = HarsanyiNet3D(
            num_classes=NUM_CLASSES,
            num_layers=args.num_layers,
            channels=args.channels,
            beta=args.beta,
            gamma=args.gamma,
            conv_size=args.conv_size,
            fc_size=args.fc_size,
            device=DEVICE,
            in_channels=1,
        ).to(DEVICE)

        # 加载预训练 stem 权重
        if args.stem_pretrained:
            log(f"Loading pretrained stem from {args.stem_pretrained}")
            model.load_pretrained_stem(args.stem_pretrained)

        # Dry run: just one batch then exit
        if args.dry_run:
            x, _ = next(iter(train_loader))
            x = x.to(DEVICE)
            with torch.no_grad():
                out = model(x)
            log(f"Dry run: input {x.shape} → output {out.shape}")
            log("Dry run PASSED. Exiting.")
            return

        scaler = GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
            epochs=EPOCHS_PER_FOLD, pct_start=0.1,
            div_factor=10, final_div_factor=100
        )

        best_mf1, best_epoch, patience_counter = 0, 0, 0
        history = {'epoch': [], 'train_loss': [], 'val_loss': [],
                   'train_acc': [], 'val_acc': [], 'val_mf1': [], 'lr': []}

        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion,
                                 DEVICE, scaler, scheduler)
            vl, va, vmf1 = validate(model, val_loader, criterion, DEVICE)
            current_lr = optimizer.param_groups[0]['lr']

            history['epoch'].append(epoch)
            history['train_loss'].append(tl)
            history['val_loss'].append(vl)
            history['train_acc'].append(ta)
            history['val_acc'].append(va)
            history['val_mf1'].append(vmf1)
            history['lr'].append(current_lr)

            if vmf1 > best_mf1:
                best_mf1, best_epoch, patience_counter = vmf1, epoch, 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'mf1': vmf1,
                    'args': vars(args),
                }, os.path.join(save_dir, f"best_fold{fold}.pth"))
                marker = "★ Saved"
            else:
                patience_counter += 1
                marker = f"({patience_counter}/{PATIENCE})"

            if epoch > MIN_EPOCHS and patience_counter >= PATIENCE:
                log(f"  Early stopping at epoch {epoch} (best: {best_mf1:.3f} @ Ep{best_epoch})")
                break

            log(f" Ep {epoch:03d} [LR: {current_lr:.2e}] | "
                f"Train Loss {tl:.3f} Acc {ta:.3f} | "
                f"Val Loss {vl:.3f} Acc {va:.3f} | MF1 {vmf1:.3f} {marker}")

        plot_history(history, save_dir, fold)
        pd.DataFrame(history).to_csv(os.path.join(save_dir, f"fold{fold}_history.csv"), index=False)

        # 用最佳模型推理
        checkpoint = torch.load(os.path.join(save_dir, f"best_fold{fold}.pth"),
                                map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            fold_probs_list = []
            for x, _ in val_loader:
                x = x.to(DEVICE)
                with autocast():
                    out = (torch.softmax(model(x), 1) +
                           torch.softmax(model(torch.flip(x, dims=[4])), 1)) / 2.0
                    fold_probs_list.append(out.cpu().numpy())
            fold_probs = np.concatenate(fold_probs_list)
            if len(fold_probs) == len(val_idx_3x):
                final_file_probs[val_idx] = fold_probs.reshape(len(val_idx), 3, NUM_CLASSES).mean(1)

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    final_preds = final_file_probs.argmax(1)
    log(f"\n{'='*80}")
    log(f"Final Accuracy: {accuracy_score(y_global, final_preds):.4f}")
    log(classification_report(y_global, final_preds, target_names=tmp_ds.classes, zero_division=0))
    plot_confusion_matrix(y_global, final_preds, tmp_ds.classes, save_dir)
    log(f"\n>> 混淆矩阵已保存至: {os.path.join(save_dir, 'confusion_matrix.png')}")

    # 保存最终预测
    np.save(os.path.join(save_dir, 'final_probs.npy'), final_file_probs)
    np.save(os.path.join(save_dir, 'final_preds.npy'), final_preds)
    log("Done.")


if __name__ == "__main__":
    main()
