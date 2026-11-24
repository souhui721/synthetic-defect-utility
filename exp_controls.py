
"""对照组实验：回应评审点名的三件事

1. 「真图增强」对照 —— 排除「只是正样本变多了」这个混杂因素。
   同样给 N 张"假"样本，但不用 CutPaste，而是把抽到的 k 张真缺陷图做翻转/旋转/缩放/颜色抖动。
   如果合成样本连普通增强都打不过，"合成数据有用"的前提就不成立。
2. 多重比较校正（Benjamini-Hochberg）—— 32 个格子里只报 p<0.05 是自欺。
3. seen / unseen 拆分 —— 区分「真缺陷少」与「没见过这种缺陷类型」两种解释。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
from PIL import Image
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_mvp import (build_backbone, cutpaste, cutpaste_salient, extract,  # noqa: E402
                     load_mask, load_rgb, masks_to_tokens, scan_dataset)

HERE = os.path.dirname(os.path.abspath(__file__))


def augment(img, mask, rng):
    """几何 + 颜色增强，掩码同步变换。返回 (aug_img, aug_mask)。"""
    h, w = img.shape[:2]
    # 随机翻转
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.3:
        img, mask = img[::-1, :], mask[::-1, :]
    # 旋转
    ang = rng.uniform(-35, 35)
    im = Image.fromarray(img).rotate(ang, resample=Image.BILINEAR)
    mk = Image.fromarray((mask * 255).astype(np.uint8)).rotate(ang, resample=Image.NEAREST)
    # 缩放 + 裁回原尺寸
    s = rng.uniform(0.8, 1.2)
    nw, nh = int(w * s), int(h * s)
    im = im.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    mk = mk.resize((nw, nh), Image.NEAREST).resize((w, h), Image.NEAREST)
    a = np.asarray(im, dtype=np.float32)
    # 颜色抖动
    a = a * rng.uniform(0.9, 1.1) + rng.uniform(-8, 8)
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a, (np.asarray(mk) > 127).astype(np.float32)


def bh_correct(pvals):
    """Benjamini-Hochberg FDR。返回校正后的 q 值（保持原顺序）。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "mvtec", "capsule"))
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "resnet18-f37072fd.pth"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--n-synth", type=int, default=2000)
    ap.add_argument("--n-good", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--ns", type=int, nargs="+", default=[0, 25, 50, 100, 200])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--aug-per-defect", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    t0 = time.time()
    rng = random.Random(0)

    train_good, test_good, defects = scan_dataset(args.data)
    d_types = [d for d, _, _ in defects]
    print(f"[data] good={len(train_good)} test_good={len(test_good)} defect={len(defects)}", flush=True)

    net = build_backbone(args.weights, device)

    # ---- 固定特征
    rng_np = np.random.RandomState(0)
    good_idx = rng_np.choice(len(train_good), size=min(args.n_good, len(train_good)), replace=False)
    g_tok, _ = extract(net, np.stack([load_rgb(train_good[i], args.size) for i in good_idx]),
                       device, args.grid, args.size)

    ev_g_tok, _ = extract(net, np.stack([load_rgb(p, args.size) for p in test_good]),
                          device, args.grid, args.size)

    d_tok, _ = extract(net, np.stack([load_rgb(p, args.size) for _, p, _ in defects]),
                       device, args.grid, args.size)
    d_masks = [load_mask(m, args.size, args.grid) if m else np.ones((args.grid, args.grid), np.float32)
               for _, _, m in defects]
    d_mask_tok = masks_to_tokens(d_masks, args.grid, args.size)

    # ---- 合成池（两种）
    src = train_good[:]
    rng.shuffle(src)
    pools = {}
    for gen, fn in (("cutpaste", cutpaste), ("salient", cutpaste_salient)):
        imgs, masks = [], []
        for i in range(args.n_synth):
            a, m = fn(load_rgb(src[i % len(src)], args.size), rng, args.size)
            imgs.append(a)
            masks.append(m)
        tok, _ = extract(net, np.stack(imgs), device, args.grid, args.size)
        pools[gen] = (tok, masks_to_tokens(masks, args.grid, args.size))
        print(f"[feat] {gen} {len(tok)} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 真图增强池：每个缺陷 8 个变体（按折建，避免跨折泄漏）
    aug_tok, aug_mask, aug_src = {}, {}, {}
    for fold in range(args.folds):
        pool_idx = np.where(np.arange(len(defects)) % args.folds != fold)[0]
        imgs, masks, srcidx = [], [], []
        for di in pool_idx:
            base = load_rgb(defects[di][1], args.size)
            m0 = d_masks[di]
            for _ in range(args.aug_per_defect):
                a, m = augment(base, m0, rng)
                imgs.append(a)
                masks.append(m)
                srcidx.append(di)
        tok, _ = extract(net, np.stack(imgs), device, args.grid, args.size)
        aug_tok[fold] = tok
        aug_mask[fold] = masks_to_tokens(masks, args.grid, args.size)
        aug_src[fold] = np.array(srcidx)
        print(f"[feat] augment fold{fold} {len(tok)} ({time.time()-t0:.1f}s)", flush=True)

    # ---- 扫描
    def score(clf, tok):
        p = clf.predict_proba(tok.reshape(-1, tok.shape[-1]))[:, 1]
        return p.reshape(len(tok), -1).max(axis=1)

    rows = []
    for fold in range(args.folds):
        ev_idx = np.where(np.arange(len(defects)) % args.folds == fold)[0]
        pool_idx = np.where(np.arange(len(defects)) % args.folds != fold)[0]
        ev_d_tok = d_tok[ev_idx]
        ev_types = np.array([d_types[i] for i in ev_idx])
        for k in args.ks:
            for seed in args.seeds:
                r = np.random.RandomState(1000 + seed)
                kidx = r.choice(pool_idx, size=min(k, len(pool_idx)), replace=False)
                covered = set(d_types[i] for i in kidx)
                covered_mask = np.array([t in covered for t in ev_types])
                for source in ("cutpaste", "salient", "augment"):
                    for N in args.ns:
                        if N == 0 and source != "cutpaste":
                            continue
                        t1 = time.time()
                        if N == 0:
                            add_tok = np.zeros((0, args.grid * args.grid, d_tok.shape[-1]), np.float32)
                            add_mask = np.zeros((0, args.grid * args.grid), np.float32)
                        elif source == "augment":
                            cand = np.where(np.isin(aug_src[fold], kidx))[0]
                            sel = r.choice(cand, size=min(N, len(cand)), replace=False)
                            add_tok = aug_tok[fold][sel]
                            add_mask = aug_mask[fold][sel]
                        else:
                            st, sm = pools[source]
                            sel = r.choice(len(st), size=min(N, len(st)), replace=False)
                            add_tok, add_mask = st[sel], sm[sel]

                        tok = np.concatenate([g_tok, d_tok[kidx], add_tok])
                        tmask = np.concatenate([
                            np.zeros((len(g_tok), args.grid * args.grid)),
                            d_mask_tok[kidx], add_mask])
                        y = tmask.reshape(-1)
                        X = tok.reshape(-1, tok.shape[-1])
                        neg = np.where(y == 0)[0]
                        pos = np.where(y == 1)[0]
                        if len(pos) == 0:
                            continue
                        keep = r.choice(neg, size=min(len(neg), max(2000, 6 * len(pos))), replace=False)
                        use = np.concatenate([keep, pos])
                        clf = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced")
                        clf.fit(X[use], y[use])
                        sg, sd = score(clf, ev_g_tok), score(clf, ev_d_tok)
                        auc_all = roc_auc_score([0] * len(sg) + [1] * len(sd), np.concatenate([sg, sd]))
                        auc_cov = auc_unc = float("nan")
                        if covered_mask.sum() >= 5:
                            m = np.concatenate([np.zeros(len(sg), bool), covered_mask])
                            auc_cov = roc_auc_score(np.concatenate([np.zeros(len(sg)), np.ones(int(covered_mask.sum()))]),
                                                    np.concatenate([sg, sd[covered_mask]]))
                        if (~covered_mask).sum() >= 5:
                            auc_unc = roc_auc_score(np.concatenate([np.zeros(len(sg)), np.ones(int((~covered_mask).sum()))]),
                                                    np.concatenate([sg, sd[~covered_mask]]))
                        rows.append({"fold": fold, "seed": seed, "k": k, "N": N, "source": source,
                                     "auc": float(auc_all), "auc_seen": float(auc_cov),
                                     "auc_unseen": float(auc_unc), "n_types_covered": len(covered),
                                     "fit_s": time.time() - t1})
        print(f"  fold {fold} 完成 ({time.time()-t0:.1f}s)", flush=True)

    # ---- 汇总 + BH 校正
    summary, pvals = {}, []
    for source in ("cutpaste", "salient", "augment"):
        for k in args.ks:
            base = {(r["fold"], r["seed"]): r["auc"] for r in rows
                    if r["k"] == k and r["N"] == 0}
            if not base:
                continue
            for N in args.ns:
                if N == 0:
                    continue
                vals = [r["auc"] for r in rows if r["source"] == source and r["k"] == k and r["N"] == N]
                if not vals:
                    continue
                pairs = [r["auc"] - base[(r["fold"], r["seed"])] for r in rows
                         if r["source"] == source and r["k"] == k and r["N"] == N]
                b = [base[key] for key in sorted(base)]
                v = [r["auc"] for r in sorted([x for x in rows if x["source"] == source and x["k"] == k and x["N"] == N],
                                              key=lambda x: (x["fold"], x["seed"]))]
                p = float(stats.ttest_rel(b, v).pvalue) if len(b) == len(v) and np.std(v) > 0 else 1.0
                key = f"{source}|{k}|{N}"
                summary[key] = {"mean": float(np.mean(vals)), "delta": float(np.mean(pairs)),
                                "sem": float(stats.sem(vals)), "p": p,
                                "seen": float(np.nanmean([r["auc_seen"] for r in rows
                                                          if r["source"] == source and r["k"] == k and r["N"] == N])),
                                "unseen": float(np.nanmean([r["auc_unseen"] for r in rows
                                                            if r["source"] == source and r["k"] == k and r["N"] == N]))}
                pvals.append(p)

    qs = bh_correct(pvals) if pvals else []
    for (key, v), q in zip(summary.items(), qs):
        v["q"] = float(q)

    print("\n=== ΔAUROC（行=k，列=N）；cell=Δ(q)；三个来源并排 ===")
    for k in args.ks:
        base_rows = [r for r in rows if r["k"] == k and r["N"] == 0]
        base_auc = float(np.mean([r["auc"] for r in base_rows])) if base_rows else float("nan")
        base_seen = float(np.nanmean([r["auc_seen"] for r in base_rows])) if base_rows else float("nan")
        base_unseen = float(np.nanmean([r["auc_unseen"] for r in base_rows])) if base_rows else float("nan")
        print(f"\n-- k={k}  基线 AUROC={base_auc:.3f}（seen {base_seen:.3f} / unseen {base_unseen:.3f}）")
        print(f"{'source':12s}" + "".join(f"{n:>16d}" for n in args.ns if n > 0))
        for source in ("cutpaste", "salient", "augment"):
            cells = []
            for N in args.ns:
                if N == 0:
                    continue
                v = summary.get(f"{source}|{k}|{N}")
                cells.append(f"{v['delta']:+.3f}({v['q']:.2f})" if v else "      --      ")
            print(f"{source:12s}" + "".join(f"{c:>16s}" for c in cells))

    print("\n=== 绝对 AUROC：overall / seen / unseen ===")
    print(f"{'source':12s}{'k':>3s}{'N':>5s}{'overall':>9s}{'seen':>8s}{'unseen':>8s}")
    for source in ("cutpaste", "salient", "augment"):
        for k in args.ks:
            for N in args.ns:
                if N == 0:
                    if source != "cutpaste":
                        continue
                    rr = [r for r in rows if r["k"] == k and r["N"] == 0]
                    if not rr:
                        continue
                    print(f"{'-':12s}{k:>3d}{0:>5d}{np.mean([r['auc'] for r in rr]):9.3f}"
                          f"{np.nanmean([r['auc_seen'] for r in rr]):8.3f}"
                          f"{np.nanmean([r['auc_unseen'] for r in rr]):8.3f}")
                else:
                    v = summary.get(f"{source}|{k}|{N}")
                    if not v:
                        continue
                    print(f"{source:12s}{k:>3d}{N:>5d}{v['mean']:9.3f}{v['seen']:8.3f}{v['unseen']:8.3f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results_controls.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "summary": summary, "rows": rows,
                   "elapsed_s": time.time() - t0}, f, ensure_ascii=False, indent=2)
    print(f"\n[done] {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()

