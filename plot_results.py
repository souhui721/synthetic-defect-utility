
"""对 exp_mvp.py 的 results.json 做统计与出图（支持多种生成方法）。

产出：
  fig_curve.png       两种合成方法下 AUROC 随合成量的变化（±1 SEM）
  fig_map.png         有效性地图雏形：ΔAUROC(k, N) 热力图，按生成方法分面
  fig_preview.png     两种合成方法的样例（人眼检查"假缺陷"长什么样）
  stats.txt           每个格的均值 / SEM / 配对检验
"""
import glob
import json
import os
import random

import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))

OUT = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data", "mvtec", "capsule")
GEN_CN = {"cutpaste": "CutPaste（原位取块）", "salient": "CutPaste-S（物体→背景）"}

for f in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\Deng.ttf"]:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

with open(os.path.join(OUT, "results.json"), encoding="utf-8") as f:
    payload = json.load(f)
rows = payload["rows"]
ks, ns = payload["args"]["ks"], payload["args"]["ns"]
gens = payload["args"]["gens"]

lines = [f"总耗时 {payload['elapsed_s']:.1f}s；记录数 {len(rows)}", ""]
print("\n".join(lines))

tab = {}
for gen in gens:
    sub = [r for r in rows if r.get("gen") == gen]
    print(f"\n=== {gen} ===")
    print("k    N      mean     sem     Δmean    Δsem    p(paired t)")
    for k in ks:
        base = {(r["fold"], r["seed"]): r["auc"] for r in sub if r["k"] == k and r["N"] == 0}
        for N in ns:
            vals = [r["auc"] for r in sub if r["k"] == k and r["N"] == N]
            if not vals:
                continue
            mean, sem = float(np.mean(vals)), float(stats.sem(vals))
            if N == 0 or not base:
                delta, dsem, p = 0.0, 0.0, float("nan")
            else:
                pairs = [r["auc"] - base[(r["fold"], r["seed"])]
                         for r in sub if r["k"] == k and r["N"] == N]
                delta, dsem = float(np.mean(pairs)), float(stats.sem(pairs))
                b = [base[key] for key in sorted(base)]
                v = [r["auc"] for r in sorted([x for x in sub if x["k"] == k and x["N"] == N],
                                              key=lambda x: (x["fold"], x["seed"]))]
                p = float(stats.ttest_rel(b, v).pvalue) if len(b) == len(v) else float("nan")
            tab[(gen, k, N)] = (mean, sem, delta, dsem, p)
            line = f"{k:<4d}{N:<6d}{mean:.4f}  {sem:.4f}  {delta:+.4f}  {dsem:.4f}  {p:.3f}"
            print(line)
            lines.append(f"{gen} {line}")

with open(os.path.join(OUT, "stats.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ---------------- 图 1：曲线（按生成方法分面）
fig, axes = plt.subplots(1, len(gens), figsize=(6.2 * len(gens), 4.0), dpi=200, sharey=True)
axes = np.atleast_1d(axes)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ks)))
for ax, gen in zip(axes, gens):
    for c, k in zip(colors, ks):
        m = [tab[(gen, k, n)][0] for n in ns]
        e = [tab[(gen, k, n)][1] for n in ns]
        ax.errorbar(ns, m, yerr=e, marker="o", ms=3.5, lw=1.5, capsize=2, color=c,
                    label=f"真实缺陷 {k} 张")
    ax.set_title(GEN_CN.get(gen, gen), fontsize=10)
    ax.set_xlabel("合成缺陷样本数 N")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("图像级 AUROC")
axes[-1].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("合成量的边际效用（MVTec AD · capsule · 5 折 × 2 种子）", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_curve.png"))
plt.close(fig)

# ---------------- 图 2：Δ 热力图（按生成方法分面）
fig, axes = plt.subplots(1, len(gens), figsize=(5.0 * len(gens), 2.9), dpi=200)
axes = np.atleast_1d(axes)
allv = [abs(tab[(g, k, n)][2]) for g in gens for k in ks for n in ns if (g, k, n) in tab]
vmax = max(0.03, max(allv) if allv else 0.05)
for ax, gen in zip(axes, gens):
    delta = np.array([[tab[(gen, k, n)][2] for n in ns] for k in ks])
    im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_yticks(range(len(ks)), [f"{k}" for k in ks])
    ax.set_xlabel("合成缺陷样本数 N")
    ax.set_ylabel("真实缺陷样本数 k")
    ax.set_title(GEN_CN.get(gen, gen), fontsize=10)
    for i in range(len(ks)):
        for j in range(len(ns)):
            v = delta[i, j]
            ax.text(j, i, "—" if j == 0 else f"{v:+.3f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(v) > vmax * 0.55 else "black")
fig.suptitle("加合成样本后 AUROC 的变化（蓝=有用，红=有害）", fontsize=11)
fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
fig.savefig(os.path.join(OUT, "fig_map.png"), bbox_inches="tight")
plt.close(fig)

# ---------------- 图 3：合成样例（两种方法对照，直接调用实验里的生成函数）
import sys

sys.path.insert(0, HERE)
from exp_mvp import cutpaste, cutpaste_salient, load_rgb  # noqa: E402

rng = random.Random(11)
good = sorted(glob.glob(os.path.join(DATA, "train", "good", "*.png")))
sel = rng.sample(good, 4)
fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.2), dpi=160)
for j, p in enumerate(sel):
    base = load_rgb(p, 256)
    for i, (fn, name) in enumerate([(cutpaste, "CutPaste 原位取块"), (cutpaste_salient, "CutPaste-S 物体→背景")]):
        a, m = fn(base, random.Random(100 + j), 256)
        ys, xs = np.where(m > 0)
        axes[i, j].imshow(a)
        if len(xs):
            axes[i, j].add_patch(plt.Rectangle((xs.min(), ys.min()), xs.max() - xs.min(),
                                               ys.max() - ys.min(), fill=False, ec="lime", lw=1.2))
        axes[i, j].set_title(name if j == 0 else "", fontsize=8)
        axes[i, j].axis("off")
fig.suptitle("合成样例：绿框=模型被告知'这里有问题'。上排多数贴了等于没贴，下排才是可见的异物", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_preview.png"))
plt.close(fig)
print("\n图已保存到", OUT)
