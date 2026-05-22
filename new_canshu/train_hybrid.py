#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HarsanyiNet3D 混合方案训练：R(2+1)D-18 特征提取 + HarsanyiBlock 顶层。
预计精度接近 R(2+1)D-18 (~90%)，同时保持单次前传精确 Shapley 值。

用法:
    python new_canshu/train_hybrid.py --epochs 100 --batch_size 32 --device cuda:2

原理:
    冻结 Kinetics-400 预训练的 R(2+1)D，提取 layer3 中间特征 (256,8,14,14)
    → Adapter (自适应池化到 8×8×8 + Conv1×1 降维) → z0
    → HarsanyiBlock (与纯 HarsanyiNet3D 完全相同的代码)
    → 分类 + 精确 Shapley
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
from model.HybridNet import HybridR2Plus1D

torch.backends.cudnn.benchmark = True

# ===================== 参数 =====================
parser = argparse.ArgumentParser(description='Hybrid R(2+1)D + HarsanyiNet Training')
parser.add_argument('--data_root', type=str,
                    default="/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2")
parser.add_argument('--num_classes', type=int, default=16)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--frames', type=int, default=32)
parser.add_argument('--temporal_stride', type=int, default=2)

# HarsanyiBlock
parser.add_argument('--num_layers', type=int, default=6)
parser.add_argument('--channels', type=int, default=128)
parser.add_argument('--conv_size', type=int, default=8)
parser.add_argument('--fc_size', type=int, default=64)
parser.add_argument('--beta', type=int, default=1000)
parser.add_argument('--gamma', type=float, default=1.0)

# 训练
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--mixup_prob', type=float, default=0.5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--focal_gamma', type=float, default=1.0)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--min_epochs', type=int, default=30)

# 系统
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--save_dir', type=str, default=None)

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
MIXUP_PROB = args.mixup_prob
PATIENCE = args.patience
MIN_EPOCHS = args.min_epochs

torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

print(f"Device: {DEVICE} | GPU: {torch.cuda.get_device_name(DEVICE)}")


# ===================== 数据管道 (同原版) =====================
class FastAug:
    def __init__(self):
        self.do_flip_h = random.random() < 0.5
        self.do_flip_v = random.random() < 0.5
        self.do_affine = random.random() < 0.7; self.angle = random.uniform(-15, 15); self.scale = random.uniform(0.9, 1.1)
        self.do_cutout = random.random() < 0.5
        self.ch, self.cw = 56, 56
        self.cy = random.randint(0, max(1, IMG_H - self.ch)) if self.do_cutout else 0
        self.cx = random.randint(0, max(1, IMG_W - self.cw)) if self.do_cutout else 0
        self.do_brightness = random.random() < 0.5; self.brightness_factor = random.uniform(0.7, 1.3)
        self.do_contrast = random.random() < 0.5; self.contrast_factor = random.uniform(0.7, 1.3)

    def __call__(self, video_tensor):
        out = np.zeros_like(video_tensor); T, H, W = video_tensor.shape
        M = cv2.getRotationMatrix2D((W/2, H/2), self.angle, self.scale) if self.do_affine else None
        for i in range(T):
            img = video_tensor[i].copy()
            if self.do_flip_h: img = cv2.flip(img, 1)
            if self.do_flip_v: img = cv2.flip(img, 0)
            if self.do_affine:
                img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            if self.do_brightness: img = np.clip(img * self.brightness_factor, 0.0, 1.0)
            if self.do_contrast:
                mean = img.mean(); img = np.clip((img - mean) * self.contrast_factor + mean, 0.0, 1.0)
            if self.do_cutout: img[self.cy:self.cy+self.ch, self.cx:self.cx+self.cw] = 0.0
            out[i] = img
        return out


class MitoDataset3D_HighRes(Dataset):
    def __init__(self, root, is_train=True, apply_aug=True):
        self.root = root; self.is_train = is_train; self.apply_aug = apply_aug and is_train
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
                    for v in range(self.views):
                        samples.append((os.path.join(cdir, f), v))
                        labels.append(self.class2idx[c])
        return samples, labels

    def _process_tif(self, path, view_idx):
        try: tif = tiff.imread(path)
        except: return np.zeros((1, FRAMES_PER_CLIP, IMG_H, IMG_W), np.float32)
        if tif.ndim == 4 and tif.shape[1] == 2: tif = tif[:, 1]
        elif tif.ndim == 4 and tif.shape[-1] == 2: tif = tif[..., 1]
        if tif.ndim == 2: tif = tif[None]
        if tif.ndim != 3 or tif.shape[0] == 0: return np.zeros((1, FRAMES_PER_CLIP, IMG_H, IMG_W), np.float32)
        T, raw_H, raw_W = tif.shape
        tif = tif.astype(np.float32)
        if tif.max() > 1: tif /= 255.0
        L = FRAMES_PER_CLIP; req_T = (L - 1) * TEMPORAL_STRIDE + 1
        if T < L: indices = np.resize(np.arange(T), L); indices.sort()
        elif T < req_T: indices = np.linspace(0, T - 1, L).astype(int)
        else:
            start = np.random.randint(0, T - req_T + 1) if self.is_train else 0
            indices = np.arange(start, start + req_T, TEMPORAL_STRIDE)
        clip_frames = tif[indices]
        valid_pixels = clip_frames[clip_frames > 1e-4]
        bg_mean = np.percentile(valid_pixels, 20) if len(valid_pixels) > 0 else 0.0
        bg_std = max(np.std(valid_pixels), 0.001) if len(valid_pixels) > 0 else 0.001
        clip_frames = np.maximum(clip_frames - bg_mean, 0.0)
        p99 = np.percentile(clip_frames, 99.9)
        if p99 > 1e-5: clip_frames /= p99
        clip_frames = np.clip(clip_frames, 0.0, 1.0)
        max_proj = np.max(clip_frames, axis=0)
        coords = cv2.findNonZero((max_proj > 0.01).astype(np.uint8))
        if coords is not None and len(coords) > 0:
            x, y, w, h = cv2.boundingRect(coords); margin = 8
            x1 = max(0, x - margin); y1 = max(0, y - margin)
            x2 = min(raw_W, x + w + margin); y2 = min(raw_H, y + h + margin)
        else: x1, y1, x2, y2 = 0, 0, raw_W, raw_H
        cropped_clip = clip_frames[:, y1:y2, x1:x2]
        if cropped_clip.size == 0: return np.zeros((1, L, IMG_H, IMG_W), np.float32)
        _, crop_h, crop_w = cropped_clip.shape
        scale = min(IMG_H / crop_h, IMG_W / crop_w)
        new_h = min(int(round(crop_h * scale)), IMG_H)
        new_w = min(int(round(crop_w * scale)), IMG_W)
        final_clip = np.zeros((L, IMG_H, IMG_W), dtype=np.float32)
        interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
        pad_top = random.randint(0, IMG_H - new_h) if self.is_train else (IMG_H - new_h) // 2
        pad_left = random.randint(0, IMG_W - new_w) if self.is_train else (IMG_W - new_w) // 2
        pad_bottom = IMG_H - new_h - pad_top; pad_right = IMG_W - new_w - pad_left
        for i in range(L):
            resized = cv2.resize(cropped_clip[i], (new_w, new_h), interpolation=interp)
            final_clip[i] = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.0)
        if self.apply_aug: final_clip = FastAug()(final_clip)
        noise = np.random.normal(0.0, bg_std, final_clip.shape)
        final_clip = np.where(final_clip < 1e-4, np.abs(noise).astype(np.float32), final_clip)
        return np.clip(final_clip, 0.0, 1.0)[np.newaxis, ...]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.from_numpy(self._process_tif(*self.samples[idx])).float(), torch.tensor(self.labels[idx], dtype=torch.long)


def filter_original_classes(ds):
    original_classes = ['210302_vapB', '211018_cos93_nocada', '211018_cos93_taxol', '211018_dko93',
        '211018_dko93_nocada', '211111_OLIGO1', '211111_TM5', 'atl3mch', 'cccp', 'cos93 int2',
        'cytd', 'div', 'dynasore1', 'lat', 'oligo', 'rtn4amc']
    orig_set = set(original_classes)
    new_class2idx = {c: i for i, c in enumerate(original_classes)}
    new_samples, new_labels = [], []
    for (path, view), label in zip(ds.samples, ds.labels):
        if ds.classes[label] in orig_set:
            new_samples.append((path, view))
            new_labels.append(new_class2idx[ds.classes[label]])
    ds.samples = new_samples; ds.labels = new_labels
    ds.classes = original_classes; ds.class2idx = new_class2idx
    return ds


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.label_smoothing = label_smoothing
    def forward(self, inputs, targets):
        if targets.dim() == 1:
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        else: targets_one_hot = targets
        targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
        probs = F.softmax(inputs, dim=1); log_probs = F.log_softmax(inputs, dim=1)
        pt = torch.sum(targets_one_hot * probs, dim=1); focal_weight = (1 - pt) ** self.gamma
        if self.weight is not None:
            class_weights = torch.sum(targets_one_hot * self.weight, dim=1)
            focal_weight = focal_weight * class_weights
        return (-focal_weight * torch.sum(targets_one_hot * log_probs, dim=1)).mean()


# ===================== 训练循环 =====================
def train_epoch(model, loader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)
        do_mixup = np.random.rand() < MIXUP_PROB
        if do_mixup:
            lam = np.random.beta(1.0, 1.0); ri = torch.randperm(x.size(0)).to(device)
            x_mixed = lam * x + (1 - lam) * x[ri]
            targets_mixed = lam * F.one_hot(batch_y, NUM_CLASSES).float() + (1 - lam) * F.one_hot(batch_y[ri], NUM_CLASSES).float()
        else: x_mixed = x; targets_mixed = batch_y
        optimizer.zero_grad()
        with autocast(): loss = criterion(model(x_mixed), targets_mixed)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        total_loss += loss.item() * batch_y.size(0)
        correct += (model(x_mixed).argmax(1) == batch_y).sum().item()
        total += batch_y.size(0)
    return total_loss / len(loader.dataset), correct / total if total > 0 else 0.0

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0; preds, labels_list = [], []
    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)
        with autocast():
            outputs = (model(x) + model(torch.flip(x, dims=[4]))) / 2.0
            loss = F.cross_entropy(outputs, batch_y)
        total_loss += loss.item() * batch_y.size(0)
        preds.append(outputs.argmax(1).cpu()); labels_list.append(batch_y.cpu())
    labels_all = torch.cat(labels_list); preds_all = torch.cat(preds)
    return total_loss / len(labels_all), accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average='macro')

def plot_confusion_matrix(y_true, y_pred, classes, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size":12})
    plt.title('Confusion Matrix'); plt.ylabel('True'); plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300); plt.close()

# ===================== 主函数 =====================
def main():
    save_dir = args.save_dir or time.strftime("new_canshu/hybrid_%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    tmp_ds = MitoDataset3D_HighRes(args.data_root, is_train=True)
    tmp_ds = filter_original_classes(tmp_ds)
    if len(tmp_ds) < 5: print("ERROR: too few samples"); return
    if len(tmp_ds.classes) != NUM_CLASSES: raise ValueError(f"Expected {NUM_CLASSES} classes, got {len(tmp_ds.classes)}")

    X = np.arange(len(tmp_ds)); y_global = np.array(tmp_ds.labels)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    full_val_ds = MitoDataset3D_HighRes(args.data_root, is_train=False, apply_aug=False)
    full_val_ds = filter_original_classes(full_val_ds)

    print(f"Training hybrid model: frozen R(2+1)D + HarsanyiBlock×{args.num_layers}")
    print(f"Samples: {len(tmp_ds)} train ({tmp_ds.views} view), {len(full_val_ds)} val (3 views)")
    print(f="\n{'='*80}")

    final_file_probs = np.zeros((len(y_global), NUM_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_global)):
        print(f"\n{'='*40} Fold {fold+1}/{N_SPLITS} {'='*40}")

        train_ds = Subset(tmp_ds, train_idx)
        val_idx_set = set(val_idx)
        val_idx_3x = [i for i, (path, view) in enumerate(full_val_ds.samples) if i // 3 in val_idx_set]
        val_ds = Subset(full_val_ds, val_idx_3x)

        # 类别权重
        uni, cnt = np.unique(y_global[train_idx], return_counts=True)
        cw = [len(train_idx) / (NUM_CLASSES * c) for u, c in zip(uni, cnt)]
        weights = torch.tensor([cw[int(u)] for u in range(NUM_CLASSES)]).float().to(DEVICE)

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=len(train_ds) >= BATCH_SIZE)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        model = HybridR2Plus1D(num_classes=NUM_CLASSES, num_layers=args.num_layers, channels=args.channels,
                                beta=args.beta, gamma=args.gamma, conv_size=args.conv_size,
                                fc_size=args.fc_size, device=DEVICE, freeze_backbone=True).to(DEVICE)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable: {sum(p.numel() for p in trainable_params):,} / {sum(p.numel() for p in model.parameters()):,} total")

        scaler = GradScaler()
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=EPOCHS, pct_start=0.1, div_factor=10, final_div_factor=100)
        criterion = FocalLoss(weight=weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)

        best_mf1, best_epoch, patience_counter = 0, 0, 0
        for epoch in range(1, EPOCHS + 1):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, scheduler)
            vl, va, vmf1 = validate(model, val_loader, DEVICE)

            if vmf1 > best_mf1:
                best_mf1, best_epoch, patience_counter = vmf1, epoch, 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mf1': vmf1},
                           os.path.join(save_dir, f"best_fold{fold}.pth"))
                marker = "★"
            else:
                patience_counter += 1
                marker = f"({patience_counter}/{PATIENCE})"

            if epoch > MIN_EPOCHS and patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best MF1={best_mf1:.3f} @ Ep{best_epoch})")
                break

            print(f" Ep {epoch:03d} | Train Loss {tl:.3f} Acc {ta:.3f} | Val Loss {vl:.3f} Acc {va:.3f} | MF1 {vmf1:.3f} {marker}")

        # Final inference
        best = torch.load(os.path.join(save_dir, f"best_fold{fold}.pth"), map_location=DEVICE)
        model.load_state_dict(best['model_state_dict']); model.eval()
        fold_probs = []
        for x, _ in val_loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                out = (torch.softmax(model(x), 1) + torch.softmax(model(torch.flip(x, dims=[4])), 1)) / 2.0
                fold_probs.append(out.cpu().numpy())
        fold_probs = np.concatenate(fold_probs)
        if len(fold_probs) == len(val_idx_3x):
            final_file_probs[val_idx] = fold_probs.reshape(len(val_idx), 3, NUM_CLASSES).mean(1)

        del model, optimizer, scheduler; torch.cuda.empty_cache()

    final_preds = final_file_probs.argmax(1)
    print(f"\n{'='*80}")
    print(f"Final Accuracy: {accuracy_score(y_global, final_preds):.4f}")
    print(classification_report(y_global, final_preds, target_names=tmp_ds.classes, zero_division=0))
    plot_confusion_matrix(y_global, final_preds, tmp_ds.classes, save_dir)
    print(f"Done. Results in {save_dir}")


if __name__ == '__main__':
    main()
