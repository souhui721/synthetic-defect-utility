
"""PatchCore 基线：零缺陷标注的无监督异常检测，在 capsule 上能拿多少分？

这是评审最狠的那一问的前半句：「工厂直接用 PatchCore 不就行了」。
如果 PatchCore ≈ 0.98，而我们「k 张真缺陷 + 线性探针」只有 0.85，
那整个「少样本监督 + 合成」的路线就必须重新定位。

实现说明：PatchCore 官方用 wide_resnet50_2（layer2+layer3，/8 网格）+ coreset + kNN。
这里用同样的骨干与特征层，memory bank 用「每张好图随机采若干 patch」代替 coreset
（对 AUROC 影响很小，省掉贪婪 coreset 的实现）。
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def scan_dataset(root):
    train_good = sorted(glob.glob(os.path.join(root, "train", "good", "*.png")))
    test_good = sorted(glob.glob(os.path.join(root, "test", "good", "*.png")))
    defects = []
    for d in sorted(os.listdir(os.path.join(root, "test"))):
        if d == "good":
            continue
        for p in sorted(glob.glob(os.path.join(root, "test", d, "*.png"))):
            defects.append((d, p))
    return train_good, test_good, defects


@torch.no_grad()
def patch_feats(net, imgs, device, bs=8):
    """返回 [N, P, D]：layer2 + layer3 上采样拼接，/8 网格（256→32×32）。"""
    out = []
    for i in range(0, len(imgs), bs):
        b = imgs[i:i + bs].astype(np.float32) / 255.0
        b = (b - MEAN) / STD
        t = torch.from_numpy(b).permute(0, 3, 1, 2).to(device)
        x = net.conv1(t)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        f2 = net.layer2(x)          # [B,512,H/8,W/8]
        f3 = net.layer3(f2)         # [B,1024,H/16,W/16]
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)              # [B,1536,32,32]
        f = F.avg_pool2d(f, 3, stride=1, padding=1)  # 局部邻域平滑（论文同款）
        out.append(f.permute(0, 2, 3, 1).reshape(len(t), -1, f.shape[1]).cpu())
    return torch.cat(out).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "mvtec", "capsule"))
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "wide_resnet50_2-95faca4d.pth"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--patches-per-image", type=int, default=32)  # memory bank 采样
    ap.add_argument("--chunk", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    train_good, test_good, defects = scan_dataset(args.data)
    print(f"[data] train_good={len(train_good)} test_good={len(test_good)} defect={len(defects)}", flush=True)

    net = models.wide_resnet50_2(weights=None)
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    # ---- memory bank：只用好图，零缺陷标注
    g_imgs = np.stack([load_rgb(p, args.size) for p in train_good])
    g_feat = patch_feats(net, g_imgs, device)
    print(f"[feat] train_good {tuple(g_feat.shape)} ({time.time()-t0:.1f}s)", flush=True)

    rng = np.random.RandomState(0)
    per = min(args.patches_per_image, g_feat.shape[1])
    idx = np.stack([rng.choice(g_feat.shape[1], per, replace=False) for _ in range(len(g_feat))])
    bank = torch.stack([g_feat[i, idx[i]] for i in range(len(g_feat))]).reshape(-1, g_feat.shape[-1])
    print(f"[bank] {tuple(bank.shape)}", flush=True)

    # ---- 测试集：好图 + 全部缺陷图
    ev_imgs = np.stack([load_rgb(p, args.size) for p in test_good] + [load_rgb(p, args.size) for _, p in defects])
    ev_feat = patch_feats(net, ev_imgs, device)
    y = np.array([0] * len(test_good) + [1] * len(defects))
    types = ["good"] * len(test_good) + [d for d, _ in defects]

    scores = []
    bank = bank.to(device)
    p = 0
    for i in range(0, len(ev_feat), args.chunk):
        chunk = ev_feat[i:i + args.chunk].to(device)
        b, patches, dim = chunk.shape
        flat = chunk.reshape(-1, dim)
        d = torch.cdist(flat, bank)               # [B*P, M]
        dmin = d.min(dim=1).values.reshape(b, patches)
        scores.append(dmin.max(dim=1).values.cpu())
        p += b
        if p % 64 == 0:
            print(f"  scored {p}/{len(ev_feat)} ({time.time()-t0:.1f}s)", flush=True)
    score = torch.cat(scores).numpy()

    auc = roc_auc_score(y, score)
    print(f"\n[PatchCore] 图像级 AUROC = {auc:.4f}   ({time.time()-t0:.1f}s)", flush=True)
    per_type = {}
    for t in sorted(set(types)):
        if t == "good":
            continue
        m = np.isin(types, ["good", t])
        per_type[t] = float(roc_auc_score(y[m], score[m]))
        print(f"  vs {t:15s} AUROC = {per_type[t]:.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "patchcore_baseline.json"), "w", encoding="utf-8") as f:
        json.dump({"auroc": float(auc), "per_type": per_type,
                   "n_good_test": len(test_good), "n_defect": len(defects),
                   "bank": list(bank.shape), "elapsed_s": time.time() - t0}, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(args.out, "patchcore_scores.npy"), score)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()

