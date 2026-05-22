#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量分析 HarsanyiNet3D Shapley 值的跨样本、跨类别一致性。

用法:
    python new_canshu/analyze_shapley.py \
        --model_path new_canshu/harsanyi3d_20260521_162451/best_fold2.pth
"""
import sys, os, argparse, json
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from shapley3d import HarsanyiNet3DAttribute, load_model_from_checkpoint, plot_shapley_3d
from train import MitoDataset3D_HighRes as OrigDataset

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda:2')
parser.add_argument('--data_root', type=str, default='/public/home/jiaqi/home/weimingzhi/projects/er-project-master/tsne_umap/data/ch2')
parser.add_argument('--save_dir', type=str, default='new_canshu/shapley_analysis')
parser.add_argument('--samples_per_class', type=int, default=3, help='每类抽几个样本')
parser.add_argument('--classes', type=str, nargs='+', 
                    default=['210302_vapB', '211018_cos93_taxol', '211018_dko93', 
                             '211111_OLIGO1', 'cccp', 'cytd', 'dynasore1', 'oligo'],
                    help='要分析的类别名（不限数量）')
args = parser.parse_args()

DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')
os.makedirs(args.save_dir, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Loading model from {args.model_path} ...")
model = load_model_from_checkpoint(args.model_path, DEVICE)

# 数据集（test mode, no augmentation）
dataset = OrigDataset(args.data_root, is_train=False, apply_aug=False)

# 构建类别 → 样本索引映射
class_to_indices = defaultdict(list)
for idx, (path, view) in enumerate(dataset.samples):
    class_name = dataset.classes[dataset.labels[idx]]
    if class_name in args.classes:
        class_to_indices[class_name].append(idx)

print(f"\nFound samples per class:")
for c in args.classes:
    print(f"  {c}: {len(class_to_indices.get(c, []))} test samples")

calculator = HarsanyiNet3DAttribute(model, DEVICE)
K = model.conv_size

all_results = {}

for class_name in args.classes:
    indices = class_to_indices.get(class_name, [])
    if len(indices) == 0:
        print(f"\nWARNING: No samples for {class_name}, skipping")
        continue
    
    # 选 samples_per_class 个样本（不重复）
    chosen = np.random.choice(indices, min(args.samples_per_class, len(indices)), replace=False)
    print(f"\n{'='*60}")
    print(f"Class: {class_name} ({len(chosen)} samples)")
    print(f"{'='*60}")
    
    class_shapleys = []
    
    for i, sample_idx in enumerate(chosen):
        # 每个 sample_idx 对应 1 个 TIF 文件，但 view=3（3 views）
        # 取 view=0 即可
        raw_idx = sample_idx * 3  # 因为 val 数据集 views=3
        if raw_idx >= len(dataset):
            # 如果是 train view=1, 直接取
            x_te, y_te = dataset[sample_idx]
        else:
            x_te, y_te = dataset[raw_idx]
        
        x_te = x_te.unsqueeze(0).to(DEVICE).double()
        label = int(y_te)
        
        # 验证预测
        with torch.no_grad():
            z0 = model._get_z0(x_te)
            out = model(x_te)
            pred = int(torch.softmax(out, 1).argmax(1)[0])
            correct = "✓" if pred == label else "✗"
        
        # Shapley
        harsanyi = calculator.attribute(model, z0, target_label=label)
        shapley_3d = calculator.get_shapley(harsanyi)
        class_shapleys.append(shapley_3d)
        
        # 分析
        top5_pos = np.argsort(shapley_3d.ravel())[-5:]
        top5_neg = np.argsort(shapley_3d.ravel())[:5]
        
        # 空间集中度: top10% voxels 占总归因的比例
        flat = np.abs(shapley_3d.ravel())
        sorted_abs = np.sort(flat)
        top10pct = int(len(flat) * 0.1)
        concentration = sorted_abs[-top10pct:].sum() / flat.sum() if flat.sum() > 0 else 0
        
        # 各深度归因分布
        depth_importance = np.abs(shapley_3d).sum(axis=(1,2))
        depth_importance = depth_importance / depth_importance.sum() if depth_importance.sum() > 0 else depth_importance
        
        print(f"  [{i+1}] sample_idx={sample_idx} true={label} pred={pred} {correct}")
        print(f"       Shapley range=[{shapley_3d.min():.5f},{shapley_3d.max():.5f}]")
        print(f"       Top10% concentration={concentration:.1%}")
        print(f"       Dominant depth: argmax={depth_importance.argmax()}, " 
              f"peak_depth_weight={depth_importance.max():.1%}")
        
        # 保存图
        save_path = os.path.join(args.save_dir, 
                                 f"{class_name}_sample{sample_idx}.png")
        plot_shapley_3d(shapley_3d, save_path, 
                        title=f"{class_name} sample#{sample_idx} (pred={pred}{'✓' if correct else '✗'})")
    
    if len(class_shapleys) >= 2:
        # 类内一致性: 两两之间的余弦相似度
        flat_shaps = np.array([s.ravel() for s in class_shapleys])
        norms = np.linalg.norm(flat_shaps, axis=1, keepdims=True)
        flat_shaps_norm = flat_shaps / np.where(norms > 1e-8, norms, 1.0)
        cos_sim = flat_shaps_norm @ flat_shaps_norm.T
        # 只取上三角
        triu_inds = np.triu_indices(len(class_shapleys), k=1)
        within_sim = cos_sim[triu_inds].mean() if len(triu_inds[0]) > 0 else 0
        print(f"       Within-class cosine similarity: {within_sim:.3f}")
        all_results[class_name] = {
            'within_similarity': float(within_sim),
            'num_samples': len(class_shapleys),
            'mean_strength': float(np.mean([np.abs(s).mean() for s in class_shapleys])),
        }
    else:
        all_results[class_name] = {'num_samples': 1}

# 类间对比
print(f"\n{'='*60}")
print("Cross-class comparison (cosine similarity matrix)")
print(f"{'='*60}")

# 每个类的平均 Shapley
class_avg = {}
for class_name in args.classes:
    indices = class_to_indices.get(class_name, [])
    if len(indices) == 0:
        continue
    chosen = np.random.choice(indices, min(10, len(indices)), replace=False)
    shaps = []
    for sample_idx in chosen:
        raw_idx = sample_idx * 3
        if raw_idx < len(dataset):
            x_te, y_te = dataset[raw_idx]
        else:
            x_te, y_te = dataset[sample_idx]
        x_te = x_te.unsqueeze(0).to(DEVICE).double()
        label = int(y_te)
        with torch.no_grad():
            z0 = model._get_z0(x_te)
        harsanyi = calculator.attribute(model, z0, target_label=label)
        shaps.append(calculator.get_shapley(harsanyi))
    class_avg[class_name] = np.mean(shaps, axis=0)

# 余弦相似度矩阵
class_names_list = list(class_avg.keys())
n = len(class_names_list)
sim_matrix = np.eye(n)
for i in range(n):
    for j in range(i+1, n):
        a = class_avg[class_names_list[i]].ravel()
        b = class_avg[class_names_list[j]].ravel()
        cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        sim_matrix[i, j] = sim_matrix[j, i] = cos

print(f"{'':<25}", end="")
for cn in class_names_list:
    print(f"{cn[:10]:>10}", end="")
print()
for i in range(n):
    print(f"{class_names_list[i][:25]:<25}", end="")
    for j in range(n):
        print(f"{sim_matrix[i,j]:>10.3f}", end="")
    print()

# 保存结果
result = {
    'within_class': all_results,
    'cross_class_similarity': {
        'classes': class_names_list,
        'matrix': sim_matrix.tolist(),
    },
    'config': vars(args),
}
with open(os.path.join(args.save_dir, 'analysis_results.json'), 'w') as f:
    json.dump(result, f, indent=2, default=str)

print(f"\nResults saved to {args.save_dir}/analysis_results.json")
print("Done.")
