
"""公平对照：把监督基线也换成 wide_resnet50 特征，再和无监督 PatchCore 比一次。

评审说「你的监督基线连 PatchCore 都打不过，研究它有什么意义」——
但原来的监督基线用的是 resnet18 layer3（256 维），PatchCore 用的是 wide_resnet50（1536 维），
特征强度差一截，这个比较本身不公平。

本脚本：
  1. 两种方法**共用同一套 wide_resnet50 patch 特征**（/8 网格下采样到 16×16）；
  2. 无监督：PatchCore 式 kNN 到好图 memory bank（零缺陷标注）；
  3. 监督：k 张真缺陷 + N 张合成 → PCA 降到 128 维 → 线性探针；
  4. 看监督路线能否追平无监督。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_mvp import cutpaste, cutpaste_salient, load_mask, load_rgb, masks_to_tokens, scan_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def feats(net, imgs, device, bs=8):
    """[N, 16*16, 1536]：layer2+layer3 在 /8 网格拼接后 2×2 平均池化。"""
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
        f2 = net.layer2(x)
        f3 = net.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)
        f = F.avg_pool2d(f, 3, stride=1, padding=1)
        f = F.avg_pool2d(f, 2, stride=2)                       # 32×32 → 16×16
        out.append(f.permute(0, 2, 3, 1).reshape(len(t), -1, f.shape[1]).cpu().half())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "mvtec", "capsule"))
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "wide_resnet50_2-95faca4d.pth"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--n-good", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 25, 50, 100, 200])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--pca-dim", type=int, default=128)
    args = ap.parse_args()

    device = "cuda"
    t0 = time.time()
    rng = random.Random(0)
    train_good, test_good, defects = scan_dataset(args.data)
    print(f"[data] good={len(train_good)} test={len(test_good)} defect={len(defects)}", flush=True)

    net = models.wide_resnet50_2(weights=None)
    net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    # ---- 特征
    g_feat = feats(net, np.stack([load_rgb(p, args.size) for p in train_good]), device)
    print(f"[feat] train_good {tuple(g_feat.shape)} ({time.time()-t0:.1f}s)", flush=True)
    rng_np = np.random.RandomState(0)
    gi = rng_np.choice(len(train_good), size=args.n_good, replace=False)
    g_tok = g_feat[gi].float()

    ev_g_feat = feats(net, np.stack([load_rgb(p, args.size) for p in test_good]), device).float()
    d_feat = feats(net, np.stack([load_rgb(p, args.size) for _, p, _ in defects]), device).float()
    d_masks = [load_mask(m, args.size, args.grid) if m else np.ones((args.grid, args.grid), np.float32)
               for _, _, m in defects]
    d_mask_tok = torch.from_numpy(masks_to_tokens(d_masks, args.grid, args.size))
    print(f"[feat] eval {tuple(ev_g_feat.shape)} / defect {tuple(d_feat.shape)} ({time.time()-t0:.1f}s)", flush=True)

    src = train_good[:]
    rng.shuffle(src)
    pools = {}
    for gen, fn in (("cutpaste", cutpaste), ("salient", cutpaste_salient)):
        imgs, masks = [], []
        for i in range(args.n_synth):
            a, m = fn(load_rgb(src[i % len(src)], args.size), rng, args.size)
            imgs.append(a)
            masks.append(m)
        tok = feats(net, np.stack(imgs), device).float()
        pools[gen] = (tok, torch.from_numpy(masks_to_tokens(masks, args.grid, args.size)))
        print(f"[feat] {gen} {tuple(tok.shape)} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 无监督 PatchCore（同一套特征）
    bank = g_feat.reshape(-1, g_feat.shape[-1]).float()
    sub = rng_np.choice(len(bank), size=min(20000, len(bank)), replace=False)
    bank = bank[sub].to(device)
    ev_all = torch.cat([ev_g_feat, d_feat]).to(device)
    y_all = np.array([0] * len(ev_g_feat) + [1] * len(d_feat))
    scores = []
    for i in range(0, len(ev_all), 8):
        c = ev_all[i:i + 8]
        b, p, dim = c.shape
        d = torch.cdist(c.reshape(-1, dim), bank)
        dmin = d.min(dim=1).values.reshape(b, p)
        scores.append(dmin.max(dim=1).values.cpu())
    pc_score = torch.cat(scores).numpy()
    pc_auroc = roc_auc_score(y_all, pc_score)
    print(f"\n[无监督 PatchCore·同特征] AUROC = {pc_auroc:.4f}", flush=True)

    # ---- PCA（无监督拟合，用 good + 合成 patch，不看缺陷标签）
    pool = torch.cat([g_tok.reshape(-1, g_tok.shape[-1]),
                      pools["cutpaste"][0].reshape(-1, g_tok.shape[-1])])
    pick = rng_np.choice(len(pool), size=min(30000, len(pool)), replace=False)
    pca = PCA(n_components=args.pca_dim, random_state=0).fit(pool[pick].numpy())
    print(f"[pca] 解释方差 {pca.explained_variance_ratio_.sum():.3f}", flush=True)

    def proj(t):
        sh = t.shape
        return torch.from_numpy(pca.transform(t.reshape(-1, sh[-1]).numpy())).float().reshape(*sh[:-1], -1)

    g_p = proj(g_tok)
    evg_p = proj(ev_g_feat)
    d_p = proj(d_feat)
    pools_p = {k: (proj(v[0]), v[1]) for k, v in pools.items()}

    def probe_score(clf, t):
        p = clf.predict_proba(t.reshape(-1, t.shape[-1]).numpy())[:, 1]
        return p.reshape(len(t), -1).max(axis=1)

    rows = []
    for fold in range(args.folds):
        ev_idx = np.where(np.arange(len(defects)) % args.folds == fold)[0]
        pool_idx = np.where(np.arange(len(defects)) % args.folds != fold)[0]
        ev_d = d_p[ev_idx]
        for k in args.ks:
            for seed in args.seeds:
                r = np.random.RandomState(1000 + seed)
                kidx = r.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)
                for gen in ("cutpaste", "salient"):
                    st, sm = pools_p[gen]
                    for N in args.ns:
                        sel = r.choice(len(st), size=min(N, len(st)), replace=False) if N else np.zeros(0, int)
                        tok = torch.cat([g_p, d_p[kidx], st[sel]])
                        tm = torch.cat([torch.zeros(len(g_p), args.grid * args.grid),
                                        d_mask_tok[kidx], sm[sel]])
                        y = tm.reshape(-1).numpy()
                        X = tok.reshape(-1, tok.shape[-1]).numpy()
                        neg = np.where(y == 0)[0]
                        pos = np.where(y == 1)[0]
                        if not len(pos):
                            continue
                        keep = r.choice(neg, size=min(len(neg), max(2000, 6 * len(pos))), replace=False)
                        use = np.concatenate([keep, pos])
                        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
                        clf.fit(X[use], y[use])
                        sg, sd = probe_score(clf, evg_p), probe_score(clf, ev_d)
                        auc = roc_auc_score([0] * len(sg) + [1] * len(sd), np.concatenate([sg, sd]))
                        rows.append({"fold": fold, "seed": seed, "k": k, "N": N,
                                     "gen": gen, "auc": float(auc)})
        print(f"  fold {fold} ({time.time()-t0:.1f}s)", flush=True)

    print(f"\n=== 同一套特征（wide_resnet50 / 16×16 / PCA{args.pca_dim}）===")
    print(f"无监督 PatchCore（0 张标注）: {pc_auroc:.4f}\n")
    print(f"{'gen':10s}{'k':>4s}" + "".join(f"{n:>9d}" for n in args.ns))
    for gen in ("cutpaste", "salient"):
        for k in args.ks:
            cells = []
            for N in args.ns:
                v = [r["auc"] for r in rows if r["gen"] == gen and r["k"] == k and r["N"] == N]
                cells.append(f"{np.mean(v):9.3f}" if v else "     --  ")
            print(f"{gen:10s}{k:>4d}" + "".join(cells))

    cell = {}
    for r in rows:
        cell.setdefault((r["gen"], r["k"], r["N"]), []).append(r["auc"])
    best_key = max(cell, key=lambda c: float(np.mean(cell[c])))
    best = {"gen": best_key[0], "k": best_key[1], "N": best_key[2], "auc": float(np.mean(cell[best_key]))}
    print(f"\n最优监督配置: {best['gen']} k={best['k']} N={best['N']} → {best['auc']:.4f}"
          f"（无监督 {pc_auroc:.4f}，差 {pc_auroc - best['auc']:+.4f}）")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results_strongsup.json"), "w", encoding="utf-8") as f:
        json.dump({"patchcore_auroc": float(pc_auroc), "pca_dim": args.pca_dim,
                   "n_synth": args.n_synth, "rows": rows, "elapsed_s": time.time() - t0},
                  f, ensure_ascii=False, indent=2)
    print(f"[done] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

