#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
版本58 破壁者增强版 (The Wall-Breaker V3)
新数据集 新的参数
修复核心: 替换底噪估计方法 - 用有效像素20%分位数代替边缘像素中位数
数据管道: 99.9%亮度均衡 + BBox裁切 + 空间游走 + 全局隐身衣 + 极值修剪
训练引擎: Focal Loss + Video Mixup + OneCycleLR + Heavy Weight Decay
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, atexit, random, time, zlib, cv2, numpy as np, tifffile as tiff
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True

# ===================== 配置 =====================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 请确保此路径正确
DATA_ROOT = "/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2"

NUM_CLASSES = 16
IMG_H = IMG_W = 224
FRAMES_PER_CLIP = 32
TEMPORAL_STRIDE = 2

# 【V56 极难任务超参】
BATCH_SIZE = 16
EPOCHS_PER_FOLD = 120
MAX_LR = 1e-4
NUM_WORKERS = 12
MIXUP_PROB = 0.5

N_SPLITS = 5
RANDOM_SEED = 42
GRAD_CLIP = 1.0
PATIENCE = 30
MIN_EPOCHS = 40

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data); s.flush()
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    plt.title('Final Confusion Matrix', fontsize=16); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

# ===================== Focal Loss =====================
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.0, label_smoothing=0.05):
        super(FocalLoss, self).__init__()
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

# ===================== 数据集 =====================
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

        # ========== V58 修改点: 底噪剥离 (新方法: 从有效像素估计) ==========
        valid_pixels = clip_frames[clip_frames > 1e-4]
        if len(valid_pixels) > 0:
            bg_mean = np.percentile(valid_pixels, 20)
            bg_std = max(np.std(valid_pixels), 0.001)
        else:
            bg_mean = 0.0
            bg_std = 0.001
        clip_frames = np.maximum(clip_frames - bg_mean, 0.0)
        # ========== V58 修改点 结束 ==========

        # 2. 99.9% 鲁棒亮度均衡
        p99 = np.percentile(clip_frames, 99.9)
        if p99 > 1e-5:
            clip_frames = clip_frames / p99
            bg_std = bg_std / p99

        clip_frames = np.clip(clip_frames, 0.0, 1.0)

        # 3. BBox 提取
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

        # 4. 空间游走与 Padding
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
            final_clip[i] = cv2.copyMakeBorder(resized_frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0.0)

        if self.apply_aug:
            final_clip = FastAug()(final_clip)

        # =========================================================
        # 【重大修复】5. 全局隐身衣与极值修剪
        # 不再区分 is_train，训练和验证统一披上背景噪音，消除域偏移！
        # =========================================================
        if self.is_train:
            noise = np.random.normal(loc=0.0, scale=bg_std, size=final_clip.shape)
        else:
            noise_seed = (zlib.crc32(path.encode("utf-8")) + view_idx + RANDOM_SEED) & 0xFFFFFFFF
            noise = np.random.default_rng(noise_seed).normal(loc=0.0, scale=bg_std, size=final_clip.shape)
        noise = np.abs(noise).astype(np.float32)
        final_clip = np.where(final_clip < 1e-4, noise, final_clip)

        # 削平极其罕见的噪音尖峰和插值过冲，保证送入模型的是绝对的 [0, 1] 纯净张量
        final_clip = np.clip(final_clip, 0.0, 1.0)

        return final_clip[np.newaxis, ...]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.from_numpy(self._process_tif(*self.samples[idx])).float(), torch.tensor(self.labels[idx], dtype=torch.long)

# ===================== 模型定义 =====================
class MitoModel3D(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        self.backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        old_stem = self.backbone.stem[0]
        new_stem = nn.Conv3d(1, old_stem.out_channels, kernel_size=old_stem.kernel_size,
                             stride=old_stem.stride, padding=old_stem.padding, bias=False)
        new_stem.weight.data = old_stem.weight.data.mean(dim=1, keepdim=True)
        self.backbone.stem[0] = new_stem
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )
    def forward(self, x): return self.backbone(x)

# ===================== 训练循环 =====================
def train_epoch(model, loader, optimizer, criterion, device, scaler, scheduler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, batch_y in loader:
        x, batch_y = x.to(device), batch_y.to(device)

        do_mixup = np.random.rand() < MIXUP_PROB
        if do_mixup:
            lam = np.random.beta(1.0, 1.0)
            rand_index = torch.randperm(x.size()[0]).to(device)
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
            outputs = (model(x) + model(torch.flip(x, dims=[4]))) / 2.0
            loss = val_criterion(outputs, batch_y)
        total_loss += loss.item() * batch_y.size(0)
        preds.append(outputs.argmax(1).cpu())
        labels_list.append(batch_y.cpu())

    labels_all = torch.cat(labels_list)
    preds_all = torch.cat(preds)
    return total_loss / len(labels_all), accuracy_score(labels_all, preds_all), f1_score(labels_all, preds_all, average='macro', zero_division=0)

# ===================== 主函数 =====================
def filter_original_classes(ds):
    """Filter dataset to only include original 7 classes, remap indices to 0-6."""
    original_classes = [
    '210302_vapB',
    '211018_cos93_nocada',
    '211018_cos93_taxol',
    '211018_dko93',
    '211018_dko93_nocada',
    '211111_OLIGO1',
    '211111_TM5',
    'atl3mch',
    'cccp',
    'cos93 int2',
    'cytd',
    'div',
    'dynasore1',
    'lat',
    'oligo',
    'rtn4amc'
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

def main():
    tmp_ds = MitoDataset3D_HighRes(DATA_ROOT, is_train=True)
    tmp_ds = filter_original_classes(tmp_ds)
    if len(tmp_ds) < 5: return
    if len(tmp_ds.classes) != NUM_CLASSES:
        raise ValueError(f"NUM_CLASSES={NUM_CLASSES}, but found {len(tmp_ds.classes)} classes under {DATA_ROOT}: {tmp_ds.classes}")

    save_dir = time.strftime("result_v58_fixed_%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)

    X = np.arange(len(tmp_ds))
    y_global = np.array(tmp_ds.labels)
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    log_path = os.path.join(save_dir, "train_log.txt")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(orig_stdout, log_file)
    sys.stderr = Tee(orig_stderr, log_file)

    def _close_log_file():
        sys.stdout = orig_stdout; sys.stderr = orig_stderr
        if not log_file.closed: log_file.close()
    atexit.register(_close_log_file)

    def log(msg): print(msg)

    log("=" * 80)
    log("v58 FIXED: R(2+1)D-18 with brightness bias fix")
    log("Fixes applied:")
    log("  1. weight_decay: 5e-2 -> 1e-2")
    log("  2. Focal gamma: 2.0 -> 1.0")
    log("  3. label_smoothing: 0.1 -> 0.05")
    log("  4. Mixup beta: (0.4,0.4) -> (1.0,1.0)")
    log("  5. Dropout 0.5 before fc")
    log("  6. Cutout augmentation (56x56)")
    log("  7. Brightness/Contrast jitter augmentation")
    log("=" * 80)

    full_val_ds = MitoDataset3D_HighRes(DATA_ROOT, is_train=False)
    full_val_ds = filter_original_classes(full_val_ds)
    final_file_probs = np.zeros((len(y_global), NUM_CLASSES))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_global)):
        log(f"\n{'='*40} Fold {fold+1}/{N_SPLITS} {'='*40}")

        train_ds = Subset(tmp_ds, train_idx)
        val_idx_set = set(val_idx)
        val_idx_3x = []
        for i, (path, view) in enumerate(full_val_ds.samples):
            file_idx = i // 3
            if file_idx in val_idx_set: val_idx_3x.append(i)

        val_ds = Subset(full_val_ds, val_idx_3x)

        uni, cnt = np.unique(y_global[train_idx], return_counts=True)
        cw = [0.0] * NUM_CLASSES
        for u, c in zip(uni, cnt): cw[u] = len(train_idx) / (NUM_CLASSES * c)
        weights = torch.tensor(cw).float().to(DEVICE)

        criterion = FocalLoss(weight=weights, gamma=1.0, label_smoothing=0.05)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=len(train_ds) >= BATCH_SIZE,
            persistent_workers=NUM_WORKERS > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            persistent_workers=NUM_WORKERS > 0,
        )
        if len(train_loader) == 0:
            raise ValueError(f"Fold {fold}: train_loader is empty. Reduce BATCH_SIZE={BATCH_SIZE} or use more training samples.")

        model = MitoModel3D().to(DEVICE)
        scaler = GradScaler()

        optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, steps_per_epoch=len(train_loader), epochs=EPOCHS_PER_FOLD,
            pct_start=0.1, div_factor=10, final_div_factor=100
        )

        best_mf1, best_epoch, patience_counter = 0, 0, 0
        history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_mf1': [], 'lr': []}

        for epoch in range(1, EPOCHS_PER_FOLD + 1):
            tl, ta = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler, scheduler)
            vl, va, vmf1 = validate(model, val_loader, criterion, DEVICE)

            current_lr = optimizer.param_groups[0]['lr']
            history['epoch'].append(epoch); history['train_loss'].append(tl); history['val_loss'].append(vl)
            history['train_acc'].append(ta); history['val_acc'].append(va); history['val_mf1'].append(vmf1); history['lr'].append(current_lr)

            if vmf1 > best_mf1:
                best_mf1, best_epoch, patience_counter = vmf1, epoch, 0
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'mf1': vmf1}, os.path.join(save_dir, f"best_fold{fold}.pth"))
                marker = "★ Saved"
            else:
                patience_counter += 1
                marker = f"({patience_counter}/{PATIENCE})"

            if epoch > MIN_EPOCHS and patience_counter >= PATIENCE:
                log(f"  Early stopping at epoch {epoch} (best: {best_mf1:.3f} @ Ep{best_epoch})")
                break
            log(f" Ep {epoch:03d} [LR: {current_lr:.2e}] | Train Loss {tl:.3f} Acc {ta:.3f} | Val Loss {vl:.3f} Acc {va:.3f} | MF1 {vmf1:.3f} {marker}")

        plot_history(history, save_dir, fold)
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(save_dir, f"fold{fold}_history.csv"), index=False)

        checkpoint = torch.load(os.path.join(save_dir, f"best_fold{fold}.pth"))
        model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        with torch.no_grad():
            fold_probs = []
            for x, _ in val_loader:
                with autocast():
                    out = (torch.softmax(model(x.to(DEVICE)), 1) + torch.softmax(model(torch.flip(x.to(DEVICE), dims=[4])), 1)) / 2.0
                    fold_probs.append(out.cpu().numpy())
            fold_probs = np.concatenate(fold_probs)
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

if __name__ == "__main__":
    main()
