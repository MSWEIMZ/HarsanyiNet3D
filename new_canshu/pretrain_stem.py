#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HarsanyiNet3D — Stem 预训练 (阶段1)

把 Stem 拿出来单独做分类训练，不引入 AND 门约束。
训练好的 Stem 权重可以加载到 HarsanyiNet3D 中替代随机初始化的版本。

用法:
    python new_canshu/pretrain_stem.py --epochs 150 --batch_size 32 --device cuda:2

输出:
    保存 stem_state_dict.pth 和 pretrain_config.json
    后续 train_harsanyi3d.py 加 --stem_pretrained <path> 加载
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, atexit, time, zlib, random, json, argparse, cv2
import numpy as np, tifffile as tiff, pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.HarsanyiNet3D import HarsanyiNet3D  # 只复用 _build_stem

torch.backends.cudnn.benchmark = True

# ===================== 参数 =====================
parser = argparse.ArgumentParser(description='HarsanyiNet3D Stem Pretraining')
parser.add_argument('--data_root', type=str,
                    default="/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2")
parser.add_argument('--num_classes', type=int, default=16)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--frames', type=int, default=32)
parser.add_argument('--temporal_stride', type=int, default=2)

# 架构（与 HarsanyiNet3D 一致）
parser.add_argument('--channels', type=int, default=256)
parser.add_argument('--conv_size', type=int, default=8)
parser.add_argument('--in_channels', type=int, default=1)

# 训练
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--mixup_prob', type=float, default=0.5)
parser.add_argument('--label_smoothing', type=float, default=0.1)

# 系统
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--save_dir', type=str, default='./stem_pretrained')

args = parser.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

FRAMES_PER_CLIP = args.frames
TEMPORAL_STRIDE = args.temporal_stride
IMG_H = IMG_W = args.img_size
NUM_CLASSES = args.num_classes
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
N_SPLITS = args.n_splits
RANDOM_SEED = args.seed
GRAD_CLIP = 1.0

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, 'pretrain_config.json'), 'w') as f:
    json.dump(vars(args), f, indent=2)

print(f"Device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")


# ===================== 日志 =====================
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

log_path = os.path.join(args.save_dir, 'pretrain_log.txt')
log_file = open(log_path, 'a', encoding='utf-8', buffering=1)
orig_stdout = sys.stdout
sys.stdout = Tee(orig_stdout, log_file)


# ===================== Stem + 分类头模型 =====================
class StemClassifier(nn.Module):
    """
    HarsanyiNet3D 的 Stem + Global Average Pooling + FC 分类头
    没有 HarsanyiBlock，纯卷积特征提取 + 分类
    """
    def __init__(self, in_channels, channels, K, num_classes):
        super().__init__()
        # 复用 HarsanyiNet3D._build_stem 的逻辑
        self.stem = nn.Sequential(
            # Block 1: heavy spatial down
            nn.Conv3d(in_channels, 48, kernel_size=(1,7,7), stride=(1,4,4),
                      padding=(0,3,3), bias=False),
            nn.BatchNorm3d(48), nn.ReLU(inplace=True),
            # (48, D, 56, 56)

            nn.Conv3d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            # (96, D//2, 28, 28)

            nn.Conv3d(96, channels, kernel_size=3, stride=(2,2,2), padding=1, bias=False),
            nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
            # (C, D//4, 14, 14)

            nn.Conv3d(channels, channels, kernel_size=(1,3,3), stride=(1,2,2),
                      padding=(0,1,1), bias=False),
            nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
            # (C, D//4, 7, 7)

            nn.AdaptiveAvgPool3d((K, K, K)),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * K**3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.classifier(x)
        return x

    def save_stem(self, path):
        """只保存 stem 部分权重"""
        torch.save(self.stem.state_dict(), path)
        print(f"Stem weights saved to {path}")


# ===================== 数据管道（与原版相同） =====================
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
            M = cv2.getRotationMatrix2D((W/2, H/2), self.angle, self.scale)
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


class MitoDataset3D_HighRes(Dataset):
    def __init__(self, root, is_train=True, apply_aug=True):
        self.root = root
        self.is_train = is_train
        self.apply_aug = apply_aug and is_train
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
        try: tif = tiff.imread(path)
        except: return np.zeros((1, FRAMES_PER_CLIP, IMG_H, IMG_W), np.float32)
        if tif.ndim == 4 and tif.shape[1] == 2: tif = tif[:, 1]
        elif tif.ndim == 4 and tif.shape[-1] == 2: tif = tif[..., 1]
        if tif.ndim == 2: tif = tif[None]
        if tif.ndim != 3 or tif.shape[0] == 0:
            return np.zeros((1, FRAMES_PER_CLIP, IMG_H, IMG_W), np.float32)
        T, raw_H, raw_W = tif.shape
        tif = tif.astype(np.float32)
        if tif.max() > 1: tif /= 255.0
        L = FRAMES_PER_CLIP
        req_T = (L - 1) * TEMPORAL_STRIDE + 1
        if T < L: indices = np.resize(np.arange(T), L); indices.sort()
        elif T < req_T: indices = np.linspace(0, T - 1, L).astype(int)
        else:
            if self.is_train: start = np.random.randint(0, T - req_T + 1)
            else: start = 0 if view_idx == 0 else ((T - req_T) // 2 if view_idx == 1 else T - req_T)
            indices = np.arange(start, start + req_T, TEMPORAL_STRIDE)
        clip_frames = tif[indices]
        valid_pixels = clip_frames[clip_frames > 1e-4]
        if len(valid_pixels) > 0:
            bg_mean = np.percentile(valid_pixels, 20)
            bg_std = max(np.std(valid_pixels), 0.001)
        else:
            bg_mean = 0.0; bg_std = 0.001
        clip_frames = np.maximum(clip_frames - bg_mean, 0.0)
        p99 = np.percentile(clip_frames, 99.9)
        if p99 > 1e-5: clip_frames = clip_frames / p99
        clip_frames = np.clip(clip_frames, 0.0, 1.0)
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
            return np.zeros((1, L, IMG_H, IMG_W), np.float32)
        _, crop_h, crop_w = cropped_clip.shape
        scale = min(IMG_H / crop_h, IMG_W / crop_w)
        new_h = min(int(round(crop_h * scale)), IMG_H)
        new_w = min(int(round(crop_w * scale)), IMG_W)
        final_clip = np.zeros((L, IMG_H, IMG_W), dtype=np.float32)
        interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        if self.is_train:
            pad_top = random.randint(0, IMG_H - new_h)
            pad_left = random.randint(0, IMG_W - new_w)
        else:
            pad_top = (IMG_H - new_h) // 2
            pad_left = (IMG_W - new_w) // 2
        pad_bottom = IMG_H - new_h - pad_top
        pad_right = IMG_W - new_w - pad_left
        for i in range(L):
            resized = cv2.resize(cropped_clip[i], (new_w, new_h), interpolation=interpolation)
            final_clip[i] = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                                cv2.BORDER_CONSTANT, value=0.0)
        if self.apply_aug:
            final_clip = FastAug()(final_clip)
        if self.is_train:
            noise = np.random.normal(0.0, bg_std, final_clip.shape)
        else:
            noise_seed = (zlib.crc32(path.encode("utf-8")) + view_idx + RANDOM_SEED) & 0xFFFFFFFF
            noise = np.random.default_rng(noise_seed).normal(0.0, bg_std, final_clip.shape)
        final_clip = np.where(final_clip < 1e-4, np.abs(noise).astype(np.float32), final_clip)
        return np.clip(final_clip, 0.0, 1.0)[np.newaxis, ...]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        x = self._process_tif(*self.samples[idx])
        y = self.labels[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


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
    ds.samples = new_samples; ds.labels = new_labels
    ds.classes = original_classes; ds.class2idx = new_class2idx
    return ds


# ===================== 训练 =====================
def train_epoch(model, loader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)
        do_mixup = np.random.rand() < args.mixup_prob
        if do_mixup:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(x.size(0)).to(device)
            target_a = batch_y; target_b = batch_y[rand_index]
            x_mixed = lam * x + (1 - lam) * x[rand_index]
            targets_a = F.one_hot(target_a, num_classes=NUM_CLASSES).float()
            targets_b = F.one_hot(target_b, num_classes=NUM_CLASSES).float()
            targets_mixed = lam * targets_a + (1 - lam) * targets_b
        else:
            x_mixed = x; targets_mixed = batch_y
        optimizer.zero_grad()
        with autocast():
            outputs = model(x_mixed)
            loss = criterion(outputs, targets_mixed)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scaler.get_scale():
            scheduler.step()
        total_loss += loss.item() * batch_y.size(0)
        correct += (outputs.argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)
    acc = correct / total if total > 0 else 0.0
    return total_loss / len(loader.dataset), acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0; preds, labels_list = [], []
    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)
        with autocast():
            outputs = model(x)
            loss = F.cross_entropy(outputs, batch_y)
        total_loss += loss.item() * batch_y.size(0)
        preds.append(outputs.argmax(1).cpu())
        labels_list.append(batch_y.cpu())
    labels_all = torch.cat(labels_list)
    preds_all = torch.cat(preds)
    return total_loss / len(labels_all), accuracy_score(labels_all, preds_all), \
           f1_score(labels_all, preds_all, average='macro', zero_division=0)


def main():
    tmp_ds = MitoDataset3D_HighRes(args.data_root, is_train=True)
    tmp_ds = filter_original_classes(tmp_ds)
    if len(tmp_ds) < 5:
        print(f"ERROR: Only {len(tmp_ds)} samples"); return

    X = np.arange(len(tmp_ds)); y_global = np.array(tmp_ds.labels)
    kf = StratifiedKFold(n_splits=min(N_SPLITS, 2), shuffle=True, random_state=RANDOM_SEED)

    full_val_ds = MitoDataset3D_HighRes(args.data_root, is_train=False, apply_aug=False)
    full_val_ds = filter_original_classes(full_val_ds)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_global)):
        print(f"\n{'='*40} Fold {fold+1} {'='*40}")

        train_ds = Subset(tmp_ds, train_idx)
        val_idx_set = set(val_idx)
        val_idx_3x = [i for i, (path, view) in enumerate(full_val_ds.samples) if i // 3 in val_idx_set]
        val_ds = Subset(full_val_ds, val_idx_3x)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True,
                                  drop_last=len(train_ds) >= BATCH_SIZE)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        model = StemClassifier(in_channels=args.in_channels, channels=args.channels,
                               K=args.conv_size, num_classes=NUM_CLASSES).to(DEVICE)
        scaler = GradScaler()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
            epochs=EPOCHS, pct_start=0.1, div_factor=10, final_div_factor=100
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_mf1, best_epoch = 0, 0
        for epoch in range(1, EPOCHS + 1):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, scheduler)
            vl, va, vmf1 = validate(model, val_loader, criterion, DEVICE)

            if vmf1 > best_mf1:
                best_mf1, best_epoch = vmf1, epoch
                model.save_stem(os.path.join(args.save_dir, f"stem_best_fold{fold}.pth"))
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mf1': vmf1},
                           os.path.join(args.save_dir, f"classifier_best_fold{fold}.pth"))

            print(f" Ep {epoch:03d} | Train Loss {tl:.3f} Acc {ta:.3f} | Val Loss {vl:.3f} Acc {va:.3f} | MF1 {vmf1:.3f}")
            if epoch % 20 == 0 or epoch == 1:
                print(f"    best MF1={best_mf1:.3f} @ ep{best_epoch}")

        print(f"Fold {fold} best MF1={best_mf1:.3f} @ epoch {best_epoch}")
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    print("\nDone. Stem weights saved to", args.save_dir)


if __name__ == '__main__':
    main()
