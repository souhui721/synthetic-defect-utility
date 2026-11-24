
"""
MVP 技术验证：合成缺陷样本的"边际效用"到底测不测得出？

实验设计（最小可行）：
  · 数据集：MVTec AD 单类（默认 capsule）
  · 真实缺陷样本数 k = 2/4/8/16（从每类缺陷的 train pool 里分层抽）
  · 合成缺陷样本数 N = 0/25/50/100/200（CutPaste 生成，自带精确掩码）
  · 检测器：冻结 ResNet18 的 patch 特征 + 线性探针（patch 级监督，图像分数取 max）
  · 指标：图像级 AUROC（好图 vs 坏图），3 个随机种子平均
  · 产出：AUROC(k, N) 曲线 + 热力图

目的不是复现论文数字，而是回答"这条管线跑不跑得通、信号出不出现、一格要多久"。
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- 数据
def load_rgb(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def load_mask(path, size, grid):
    """把缺陷掩码降到 grid×grid：任一像素落在缺陷内即为正。"""
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    a = (np.asarray(m) > 127).astype(np.float32)
    blk = size // grid
    a = a.reshape(grid, blk, grid, blk).max(axis=(1, 3))
    return a  # grid × grid, 0/1


def scan_dataset(root):
    """返回 train_good / test_good / test_defect[(type, img, mask)]"""
    train_good = sorted(glob.glob(os.path.join(root, "train", "good", "*.png")))
    test_good = sorted(glob.glob(os.path.join(root, "test", "good", "*.png")))
    defects = []
    for d in sorted(os.listdir(os.path.join(root, "test"))):
        if d == "good":
            continue
        ddir = os.path.join(root, "test", d)
        if not os.path.isdir(ddir):
            continue
        for p in sorted(glob.glob(os.path.join(ddir, "*.png"))):
            stem = os.path.splitext(os.path.basename(p))[0]
            cand = [
                os.path.join(root, "ground_truth", d, stem + "_mask.png"),
                os.path.join(root, "ground_truth", d, stem + "_mask.PNG"),
            ]
            mask = next((c for c in cand if os.path.exists(c)), None)
            defects.append((d, p, mask))
    return train_good, test_good, defects


# ---------------------------------------------------------------- 合成
def cutpaste(img, rng, size):
    """返回 (合成图, 掩码 grid 外的全尺寸 0/1 掩码)。两种形态各半：方块 / 细长条。"""
    a = img.copy()
    area = size * size
    if rng.random() < 0.5:  # 方块
        w = int(rng.uniform(0.05, 0.13) * size)
        h = int(rng.uniform(0.05, 0.13) * size)
    else:  # 划痕状细长条
        if rng.random() < 0.5:
            w, h = int(rng.uniform(0.03, 0.07) * size), int(rng.uniform(0.2, 0.45) * size)
        else:
            w, h = int(rng.uniform(0.2, 0.45) * size), int(rng.uniform(0.03, 0.07) * size)
    w, h = max(4, min(w, size - 4)), max(4, min(h, size - 4))
    x0 = rng.randint(0, size - w - 1)
    y0 = rng.randint(0, size - h - 1)
    patch = a[y0:y0 + h, x0:x0 + w].copy()
    # 旋转（用 PIL 保持简单）
    ang = rng.uniform(-30, 30)
    patch = np.asarray(
        Image.fromarray(patch).rotate(ang, resample=Image.BILINEAR), dtype=np.uint8
    )
    ph, pw = patch.shape[:2]
    x1 = rng.randint(0, max(0, size - pw - 1))
    y1 = rng.randint(0, max(0, size - ph - 1))
    a[y1:y1 + ph, x1:x1 + pw] = patch
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y1:y1 + ph, x1:x1 + pw] = 1.0
    _ = area
    return a, mask


def cutpaste_salient(img, rng, size):
    """从高纹理区（物体）取块、贴到低纹理区（背景）：制造真正可见的异物缺陷。

    动机：均匀背景上原地取块再贴回去，视觉上什么也没发生——那批"合成缺陷"是空操作。
    """
    import cv2

    a = img.copy()
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    grad = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, 3))
    if rng.random() < 0.6:
        w = int(rng.uniform(0.07, 0.13) * size)
        h = int(rng.uniform(0.07, 0.13) * size)
    else:
        if rng.random() < 0.5:
            w, h = int(rng.uniform(0.04, 0.07) * size), int(rng.uniform(0.18, 0.3) * size)
        else:
            w, h = int(rng.uniform(0.18, 0.3) * size), int(rng.uniform(0.04, 0.07) * size)
    w, h = max(6, min(w, size - 12)), max(6, min(h, size - 12))

    k = max(w, h)
    energy = cv2.blur(grad, (k, k))
    m = k // 2
    inner = energy[m:size - m, m:size - m]
    sy, sx = np.unravel_index(np.argmax(inner), inner.shape)   # 最像物体
    ty, tx = np.unravel_index(np.argmin(inner), inner.shape)   # 最像背景
    sy, sx, ty, tx = sy + m, sx + m, ty + m, tx + m

    patch = a[sy:sy + h, sx:sx + w].copy()
    if patch.size == 0:
        patch = a[:h, :w].copy()
    patch = np.asarray(Image.fromarray(patch).rotate(rng.uniform(-20, 20)), dtype=np.uint8)
    ph, pw = patch.shape[:2]
    gain = rng.uniform(0.85, 1.18)
    patch = np.clip(patch.astype(np.float32) * gain + rng.uniform(-12, 12), 0, 255).astype(np.uint8)

    # 贴到背景处，且尽量不与取样区重叠
    for _ in range(8):
        px = int(np.clip(tx + rng.randint(-k // 2, k // 2), 0, size - pw))
        py = int(np.clip(ty + rng.randint(-k // 2, k // 2), 0, size - ph))
        if abs(px - sx) > pw // 2 or abs(py - sy) > ph // 2:
            break
    a[py:py + ph, px:px + pw] = patch
    mask = np.zeros((size, size), dtype=np.float32)
    mask[py:py + ph, px:px + pw] = 1.0
    return a, mask


# ---------------------------------------------------------------- 特征
def build_backbone(weights_path, device):
    net = models.resnet18(weights=None)
    sd = torch.load(weights_path, map_location="cpu")
    net.load_state_dict(sd)
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def extract(net, imgs, device, grid, size, bs=32):
    """imgs: uint8 [B,H,W,3] → tokens [B,grid*grid,C], global [B, C*2]"""
    out_tok, out_glob = [], []
    for i in range(0, len(imgs), bs):
        batch = imgs[i:i + bs].astype(np.float32) / 255.0
        batch = (batch - MEAN) / STD
        t = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
        x = net.conv1(t)
        x = net.bn1(x)
        x = net.relu(x)
        x = net.maxpool(x)
        x = net.layer1(x)
        x = net.layer2(x)
        x = net.layer3(x)  # [B,256,H/16,W/16]
        if x.shape[-1] != grid:
            x = F.adaptive_avg_pool2d(x, grid)
        tok = x.permute(0, 2, 3, 1).reshape(len(t), -1, x.shape[1])
        out_tok.append(tok.cpu().numpy())
        g = torch.cat([tok.mean(1), tok.max(1).values], dim=1)
        out_glob.append(g.cpu().numpy())
    return np.concatenate(out_tok), np.concatenate(out_glob)


def masks_to_tokens(masks, grid, size):
    """掩码 → grid×grid 的 token 标签。既接受全尺寸 0/1 掩码，也接受已是 grid×grid 的。"""
    blk = size // grid
    out = []
    for m in masks:
        m = np.asarray(m, dtype=np.float32)
        if m.shape == (grid, grid):
            out.append(m.reshape(-1))
            continue
        mm = m.reshape(grid, blk, grid, blk).max(axis=(1, 3))
        out.append(mm.reshape(-1))
    return np.stack(out)  # [B, grid*grid]


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "mvtec", "capsule"))
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "resnet18-f37072fd.pth"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--n-good", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 25, 50, 100, 200])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gens", nargs="+", default=["cutpaste", "salient"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    train_good, test_good, defects = scan_dataset(args.data)
    print(f"[data] train_good={len(train_good)} test_good={len(test_good)} defect={len(defects)}", flush=True)
    n_with_mask = sum(1 for _, _, m in defects if m)
    print(f"[data] 缺陷图中有掩码的: {n_with_mask}/{len(defects)}", flush=True)
    if not train_good or not test_good or not defects:
        raise SystemExit("数据目录结构不对，检查 --data")

    # ---- 全部缺陷图特征（折内只做索引，避免重复前向）
    net = build_backbone(args.weights, device)
    defect_imgs = np.stack([load_rgb(p, args.size) for _, p, _ in defects])
    d_tok, _ = extract(net, defect_imgs, device, args.grid, args.size)
    d_masks = [
        load_mask(m, args.size, args.grid) if m else np.ones((args.grid, args.grid), dtype=np.float32)
        for _, _, m in defects
    ]
    d_mask_tok = masks_to_tokens(d_masks, args.grid, args.size)

    # ---- 训练用 good 图（固定）
    rng_np = np.random.RandomState(0)
    good_idx = rng_np.choice(len(train_good), size=min(args.n_good, len(train_good)), replace=False)
    good_imgs = np.stack([load_rgb(train_good[i], args.size) for i in good_idx])
    g_tok, _ = extract(net, good_imgs, device, args.grid, args.size)

    # ---- 评测用 good 图（固定）
    eval_good_imgs = np.stack([load_rgb(p, args.size) for p in test_good])
    ev_g_tok, _ = extract(net, eval_good_imgs, device, args.grid, args.size)

    # ---- 合成池：按生成方法各造一批（一次生成，各 cell 复用）
    rng = random.Random(0)
    src = train_good[:]
    rng.shuffle(src)
    pools = {}
    for gen in args.gens:
        fn = cutpaste if gen == "cutpaste" else cutpaste_salient
        imgs, masks = [], []
        for i in range(args.n_synth):
            base = load_rgb(src[i % len(src)], args.size)
            a, m = fn(base, rng, args.size)
            imgs.append(a)
            masks.append(m)
        s_tok, _ = extract(net, np.stack(imgs), device, args.grid, args.size)
        pools[gen] = (s_tok, masks_to_tokens(masks, args.grid, args.size))
        print(f"[feat] synth[{gen}]={len(s_tok)} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[feat] 总览: good={len(g_tok)} eval_good={len(ev_g_tok)} defect={len(d_tok)}", flush=True)

    # ---- 扫描：gen × folds × k × N（每折轮换评测集，避免单次划分失真）
    def image_score(clf, tok):
        p = clf.predict_proba(tok.reshape(-1, tok.shape[-1]))[:, 1]
        return p.reshape(len(tok), -1).max(axis=1)

    results = []
    for gen in args.gens:
        s_tok, s_mask_tok = pools[gen]
        for fold in range(args.folds):
            ev_idx = np.where(np.arange(len(defects)) % args.folds == fold)[0]
            pool_idx = np.where(np.arange(len(defects)) % args.folds != fold)[0]
            ev_d_tok = d_tok[ev_idx]
            y_eval = [0] * len(ev_g_tok) + [1] * len(ev_d_tok)
            for k in args.ks:
                for seed in args.seeds:
                    r = np.random.RandomState(1000 + seed)
                    kidx = r.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)
                    for N in args.ns:
                        t1 = time.time()
                        nidx = r.choice(len(s_tok), size=min(N, len(s_tok)), replace=False)

                        tok = np.concatenate([g_tok, d_tok[kidx], s_tok[nidx]])
                        tmask = np.concatenate([
                            np.zeros((len(g_tok), args.grid * args.grid)),
                            d_mask_tok[kidx], s_mask_tok[nidx],
                        ])
                        y = tmask.reshape(-1)
                        X = tok.reshape(-1, tok.shape[-1])
                        neg_idx = np.where(y == 0)[0]
                        pos_idx = np.where(y == 1)[0]
                        if len(pos_idx) == 0:
                            continue
                        keep = r.choice(neg_idx, size=min(len(neg_idx), max(2000, 6 * len(pos_idx))), replace=False)
                        use = np.concatenate([keep, pos_idx])

                        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
                        clf.fit(X[use], y[use])
                        sg, sd = image_score(clf, ev_g_tok), image_score(clf, ev_d_tok)
                        auc = roc_auc_score(y_eval, np.concatenate([sg, sd]))
                        results.append({"gen": gen, "k": k, "N": N, "seed": seed, "fold": fold,
                                        "auc": float(auc), "fit_s": time.time() - t1})
            print(f"  {gen} fold {fold} 完成 ({time.time()-t0:.1f}s)", flush=True)

    # ---- 汇总：按生成方法分组；均值、标准差、相对 N=0 的增量
    summary = {}
    for gen in args.gens:
        sub = [x for x in results if x["gen"] == gen]
        g = {}
        for k in args.ks:
            base = [x["auc"] for x in sub if x["k"] == k and x["N"] == 0]
            row = {}
            for N in args.ns:
                vals = [x["auc"] for x in sub if x["k"] == k and x["N"] == N]
                row[str(N)] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": float(np.std(vals)) if vals else float("nan"),
                    "n": len(vals),
                    "delta": float(np.mean(vals) - np.mean(base)) if vals and base else float("nan"),
                }
            g[str(k)] = row
        summary[gen] = g

    for gen in args.gens:
        g = summary[gen]
        print(f"\n[{gen}] AUROC 均值 (行=k 真实样本数, 列=N 合成样本数)")
        print("        N=" + "".join(f"{n:>9d}" for n in args.ns))
        for k, row in g.items():
            print(f"  k={k:>3s}   " + "".join(f"{row[str(n)]['mean']:9.4f}" for n in args.ns))
        print(f"[{gen}] 相对 N=0 的增量 ΔAUROC")
        print("        N=" + "".join(f"{n:>9d}" for n in args.ns))
        for k, row in g.items():
            print(f"  k={k:>3s}   " + "".join(
                ("      -- " if n == 0 else f"{row[str(n)]['delta']:+9.4f}") for n in args.ns))

    payload = {"args": vars(args), "summary": summary, "rows": results,
               "elapsed_s": time.time() - t0}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(args.out, "feats.npz"),
                        ev_g_tok=ev_g_tok, g_tok=g_tok, d_tok=d_tok, d_mask_tok=d_mask_tok,
                        **{f"s_tok_{g}": v[0] for g, v in pools.items()},
                        **{f"s_mask_tok_{g}": v[1] for g, v in pools.items()})
    print(f"\n[done] {time.time()-t0:.1f}s → {args.out}", flush=True)


if __name__ == "__main__":
    main()
