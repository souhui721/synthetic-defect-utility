# -*- coding: utf-8 -*-
"""生成申报书用插图（基于 exp_controls / exp_strongsup 的结果）。

产出：
  fig_roadmap.png         图 2  总体技术路线（七环节 + 判据回灌）
  fig_delta_heatmap.png   图 3  Δ准确度热力图，三来源分面
  fig_method_compare.png  图 4  有标注（真缺陷+合成）vs 免标注
  fig_record.png          图 5  预实验的代码清单与运行规模（读真实文件生成）

设计约定：
  - 图内不再写标题，标题统一交给 Word 图注，避免"双标题"；
  - 字号按最终插入宽度反推（effective_pt ≈ design_pt × insert_in / figsize_in）；
  - 配色语义写进色标标签，不依赖图注说明。
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "results")   # 读 json
FIGS = os.path.join(HERE, "figures")   # 写 png
os.makedirs(FIGS, exist_ok=True)

# 五种样本来源的图内标注（去术语版）
SRC_CN = {
    "cutpaste": "原位剪贴\n（CutPaste）",
    "salient": "跨区剪贴\n（CutPaste-S）",
    "augment": "真实缺陷简单变换\n（翻转／旋转／缩放／调色）",
    "repeat": "直接重复\n（同一张复制）",
    "noise": "随机噪声贴块\n（无意义合成）",
}
# 后四种画的都不是合成缺陷，横轴标注必须分开
XLAB = {
    "cutpaste": "合成缺陷数量（张）",
    "salient": "合成缺陷数量（张）",
    "augment": "变换样本数量（张）",
    "repeat": "重复样本数量（张）",
    "noise": "噪声样本数量（张）",
}
ORDER = ["cutpaste", "salient", "augment", "repeat", "noise"]


def load(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f:
        return json.load(f)


# ============================================================ 图 2 技术路线
def draw_roadmap():
    fig, ax = plt.subplots(figsize=(5.0, 1.84), dpi=320)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0.30, 2.36)
    ax.axis("off")

    BW, BH = 1.12, 0.52
    X = [0.30, 1.62, 2.94, 4.26]      # 四列左边界
    Y1 = 2.02                          # 上排中心
    Y2 = 0.62                          # 下排中心

    boxes = [
        # (x, y, 序号, 标题, 副标题, 阶段色)
        (X[0], Y1, "①", "数据准备", "3 个公开数据集", 0),
        (X[1], Y1, "②", "少样本构造", "抽取 k 张真实缺陷", 0),
        (X[2], Y1, "③", "合成缺陷生成", "4 类合成方法", 0),
        (X[3], Y1, "④", "检测模型训练", "免标注／有标注", 1),
        (X[3], Y2, "⑤", "效果评测", "四条件网格 × 10 次", 1),
        (X[2], Y2, "⑥", "原因分析", "三条线索排查", 2),
        (X[1], Y2, "⑦", "成果交付", "地图·规则·代码·报告", 2),
    ]
    PHASE = [("#E8F0F8", "#2F6FA8", "#16324F"),   # 造数
             ("#E9F5EC", "#3E8E5A", "#22402F"),   # 评测
             ("#FBF0E4", "#C08434", "#4A331A")]   # 分析交付
    for x, y, no, title, sub, ph in boxes:
        fc, ec, tc = PHASE[ph]
        ax.add_patch(
            FancyBboxPatch(
                (x, y - BH / 2), BW, BH,
                boxstyle="round,pad=0.012,rounding_size=0.06",
                linewidth=0.9, edgecolor=ec, facecolor=fc,
            )
        )
        ax.text(x + BW / 2, y + 0.115, f"{no} {title}",
                ha="center", va="center", fontsize=7.2, color=tc,
                fontweight="bold")
        ax.text(x + BW / 2, y - 0.115, sub,
                ha="center", va="center", fontsize=5.6, color="#4A6076")

    def arrow(p1, p2, style="-|>", rad=0.0, color="#2F4F6F", lw=0.9, ls="-"):
        ax.add_patch(
            FancyArrowPatch(
                p1, p2, arrowstyle=style, mutation_scale=9,
                connectionstyle=f"arc3,rad={rad}",
                linewidth=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0,
            )
        )

    # 上排横向
    for i in range(3):
        arrow((X[i] + BW + 0.02, Y1), (X[i + 1] - 0.02, Y1))
    # ④ → ⑤ 下行
    arrow((X[3] + BW / 2, Y1 - BH / 2 - 0.02), (X[3] + BW / 2, Y2 + BH / 2 + 0.02))
    # 下排横向（自右向左）
    arrow((X[3] - 0.02, Y2), (X[2] + BW + 0.02, Y2))
    arrow((X[2] - 0.02, Y2), (X[1] + BW + 0.02, Y2))

    # ⑥ → ③ 判据回灌（同列垂直向上）
    arrow((X[2] + BW / 2, Y2 + BH / 2 + 0.02),
          (X[2] + BW / 2, Y1 - BH / 2 - 0.02),
          color="#B2483A", lw=1.0, ls=(0, (3, 2)))
    ax.text(X[2] + BW / 2 - 0.06, (Y1 + Y2) / 2, "分析结果反馈",
            ha="right", va="center", fontsize=5.6, color="#B2483A", rotation=90)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    out = os.path.join(FIGS, "fig_roadmap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved fig_roadmap.png")


# ============================================================ 图 3 热力图
def draw_heatmap():
    """主实验三种来源：原位剪贴、跨区剪贴、真图简单变换。"""
    d = load("results_controls.json")
    sm = d["summary"]
    ks = d["args"]["ks"]
    ns = [n for n in d["args"]["ns"] if n != 0]
    show = ["cutpaste", "salient", "augment"]

    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.26), dpi=300,
                             constrained_layout=True)
    allv = [abs(sm[f"{g}|{k}|{n}"]["delta"]) for g in ORDER for k in ks for n in ns]
    vmax = max(0.03, max(allv))

    for ax, g in zip(axes, show):
        delta = np.array([[sm[f"{g}|{k}|{n}"]["delta"] for n in ns] for k in ks])
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ns)), [str(n) for n in ns], fontsize=8)
        ax.set_yticks(range(len(ks)), [str(k) for k in ks], fontsize=8)
        ax.set_xlabel(XLAB[g], fontsize=8, labelpad=2)
        if g == show[0]:
            ax.set_ylabel("真实缺陷数量（张）", fontsize=8, labelpad=2)
        ax.set_title(SRC_CN[g], fontsize=8.2 if g == "augment" else 8.8, pad=4)
        for i, k in enumerate(ks):
            for j, n in enumerate(ns):
                v = delta[i, j]
                q = sm[f"{g}|{k}|{n}"]["q"]
                s = f"{v:+.3f}" + ("*" if q < 0.05 else "")
                ax.text(j, i, s, ha="center", va="center", fontsize=7.0,
                        color="white" if abs(v) > vmax * 0.55 else "#1a1a1a")

    cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.012)
    cb.set_label("变化量（红＝变好）", fontsize=7.6)
    cb.ax.tick_params(labelsize=7)
    out = os.path.join(FIGS, "fig_delta_heatmap.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved fig_delta_heatmap.png")


def draw_heatmap_ctrl():
    """两种「无意义」对照：直接重复与随机噪声贴块。"""
    d = load("results_controls.json")
    sm = d["summary"]
    ks = d["args"]["ks"]
    ns = [n for n in d["args"]["ns"] if n != 0]
    show = ["repeat", "noise"]

    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.10), dpi=300,
                             constrained_layout=True)
    allv = [abs(sm[f"{g}|{k}|{n}"]["delta"]) for g in ORDER for k in ks for n in ns]
    vmax = max(0.03, max(allv))

    for ax, g in zip(axes, show):
        delta = np.array([[sm[f"{g}|{k}|{n}"]["delta"] for n in ns] for k in ks])
        im = ax.imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ns)), [str(n) for n in ns], fontsize=8)
        ax.set_yticks(range(len(ks)), [str(k) for k in ks], fontsize=8)
        ax.set_xlabel(XLAB[g], fontsize=8, labelpad=2)
        ax.set_ylabel("真实缺陷数量（张）", fontsize=8, labelpad=2)
        ax.set_title(SRC_CN[g], fontsize=8.6, pad=4)
        for i, k in enumerate(ks):
            for j, n in enumerate(ns):
                v = delta[i, j]
                q = sm[f"{g}|{k}|{n}"]["q"]
                s = f"{v:+.3f}" + ("*" if q < 0.05 else "")
                ax.text(j, i, s, ha="center", va="center", fontsize=7.2,
                        color="white" if abs(v) > vmax * 0.55 else "#1a1a1a")

    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.020)
    cb.set_label("变化量（红＝变好）", fontsize=7.4)
    cb.ax.tick_params(labelsize=6.8)
    out = os.path.join(FIGS, "fig_delta_controls.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved fig_delta_controls.png")


# ============================================================ 图 4 方法对比
def draw_compare():
    ss = load("results_strongsup.json")
    pc = ss["patchcore_auroc"]
    rows = ss["rows"]
    gene = [r for r in rows if r["N"] >= 25]
    by_k = {}
    for k in sorted({r["k"] for r in gene}):
        vals = [r["auc"] for r in gene if r["k"] == k]
        by_k[k] = (float(np.mean(vals)), float(np.std(vals)))

    fig, ax = plt.subplots(figsize=(5.6, 2.62), dpi=260)
    labels = [f"{k} 张真实缺陷\n＋合成样本" for k in by_k]
    vals = [by_k[k][0] for k in by_k]
    errs = [by_k[k][1] for k in by_k]
    bars = ax.bar(labels, vals, color="#4C72B0", yerr=errs, capsize=3.5, width=0.58,
                  error_kw={"elinewidth": 0.9, "capthick": 0.9})
    ax.axhline(pc, color="#C44E52", linestyle="--", linewidth=1.3)
    ax.text(3.42, pc + 0.010, f"免标注方法：不用任何缺陷样本，{pc:.3f}",
            color="#C44E52", fontsize=7.6, ha="right", va="bottom")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v - 0.030, f"{v:.3f}",
                ha="center", va="top", fontsize=8.6, color="white")
    ax.set_ylim(0.58, 1.0)
    ax.set_ylabel("检测得分（越接近 1 越好）", fontsize=8)
    ax.tick_params(labelsize=7.6)
    ax.grid(axis="y", alpha=0.28, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.97, bottom=0.17)
    out = os.path.join(FIGS, "fig_method_compare.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved fig_method_compare.png  patchcore=%.4f" % pc)
    for k, (m, s) in by_k.items():
        print(f"  k={k}: {m:.4f} ± {s:.4f}")


# ============================================================ 图 2 研究框架
def rbox(ax, x0, y0, x1, y1, fc, ec, lw=0.9, r=0.06):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.012,rounding_size=%.3f" % r,
        linewidth=lw, edgecolor=ec, facecolor=fc))


def arrow(ax, p1, p2, color="#5B6B7C", lw=0.9):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=8,
                                 connectionstyle="arc3,rad=0",
                                 linewidth=lw, color=color, shrinkA=0, shrinkB=0))


def draw_framework():
    """三层研究框架：顶层目标 → 三个板块 → 交付物。"""
    fig, ax = plt.subplots(figsize=(5.0, 1.98), dpi=320)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 2.22)
    ax.axis("off")

    BLUE = ("#E8F0F8", "#2F6FA8")
    GREEN = ("#E9F5EC", "#3E8E5A")
    ORANGE = ("#FBF0E4", "#C08434")
    PURPLE = ("#EFEAF6", "#7A5FA8")

    # 顶层
    rbox(ax, 1.05, 1.80, 4.55, 2.16, *BLUE, lw=1.1)
    ax.text(2.80, 2.045, "造数地图 —— 合成缺陷效用评测", fontsize=9.0,
            color="#16324F", ha="center", va="center", fontweight="bold")
    ax.text(2.80, 1.895, "把“合成缺陷有用”从默认前提，变成可以检验、可以查表的问题",
            fontsize=6.6, color="#4A6076", ha="center", va="center")

    # 三条主线
    ax.plot([2.80, 2.80], [1.80, 1.70], color="#5B6B7C", lw=0.9)
    ax.plot([0.98, 4.62], [1.70, 1.70], color="#5B6B7C", lw=0.9)
    for x in (0.98, 2.80, 4.62):
        arrow(ax, (x, 1.70), (x, 1.60))

    panels = [
        (0.10, 1.85, "① 造什么", "合成方法与对照",
         ["原位剪贴 · 跨区剪贴", "纹理叠加 · AI 生成", "统一输出：图像＋位置"], BLUE),
        (1.92, 3.67, "② 怎么评", "控制变量 + 统计校正",
         ["三组对照：变换／重复／噪声", "80 组条件 × 10 次重复", "校正之后才谈“有提升”"], GREEN),
        (3.74, 5.50, "③ 怎么用", "查询表与判定规则",
         ["条件—效用查询表", "反作用判定规则", "全部在训练之前算出"], ORANGE),
    ]
    for x0, x1, title, sub, lines, (fc, ec) in panels:
        rbox(ax, x0, 0.74, x1, 1.60, fc, ec)
        cx = (x0 + x1) / 2
        ax.text(cx, 1.455, title, fontsize=8.4, color=ec, ha="center",
                va="center", fontweight="bold")
        ax.text(cx, 1.315, sub, fontsize=6.4, color="#4A6076", ha="center", va="center")
        ax.plot([x0 + 0.10, x1 - 0.10], [1.235, 1.235], color=ec, lw=0.5, alpha=0.55)
        for i, t in enumerate(lines):
            ax.text(cx, 1.135 - i * 0.135, t, fontsize=6.7, color="#2A2A2A",
                    ha="center", va="center")

    # 汇聚到交付
    for x in (0.98, 2.80, 4.62):
        ax.plot([x, x], [0.74, 0.64], color="#5B6B7C", lw=0.9)
    ax.plot([0.98, 4.62], [0.64, 0.64], color="#5B6B7C", lw=0.9)
    arrow(ax, (2.80, 0.64), (2.80, 0.54))

    rbox(ax, 0.35, 0.10, 5.25, 0.54, *PURPLE, lw=1.1)
    ax.text(2.80, 0.32,
            "交付：什么条件下值得造 · 加到多少就该停 · 哪些条件反而更差",
            fontsize=8.2, color="#4B3A6B", ha="center", va="center",
            fontweight="bold")

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
    out = os.path.join(FIGS, "fig_framework.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved fig_framework.png")


# ============================================================ 图 5 实验台账
def draw_record():
    """把 exp/mvp 目录里的真实文件和真实耗时画成一张台账（不是示意图）。"""
    from matplotlib.patches import Rectangle

    scripts = ["exp_mvp.py", "exp_controls.py", "exp_strongsup.py",
               "exp_patchcore.py", "plot_results.py", "plot_paper_figs.py"]
    results = ["results.json", "results_controls.json",
               "results_strongsup.json", "stats.txt"]
    files = [(n, os.path.getsize(os.path.join(BASE, n)) / 1024.0)
             for n in scripts + results]

    runs = [("results.json", "预实验"), ("results_controls.json", "对照实验"),
            ("results_strongsup.json", "强基线")]
    n_fit, t_all = 0, 0.0
    rows = []
    for fn, tag in runs:
        d = load(fn)
        n, t = len(d["rows"]), float(d["elapsed_s"])
        n_fit += n
        t_all += t
        rows.append((tag, "%d 次 / %d 秒" % (n, round(t))))
    rows = [("条件组合", "%d 组" % len(load("results_controls.json")["summary"])),
            ("对照来源", "3 种（变换／重复／噪声）"),
            ("每组重复", "10 次（5 折 × 2 种子）")] + rows + \
           [("合计", "%d 次 / %d 秒" % (n_fit, round(t_all)))]

    fig, ax = plt.subplots(figsize=(5.0, 2.02), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    LX, RX = 0.010, 0.516          # 左块 / 右块左边界
    LEND, REND = 0.470, 0.990      # 右对齐边界
    HEAD, RULE = 0.918, 0.880
    Y0, SL, SR = 0.826, 0.0665, 0.0945

    # 表头
    ax.text(LX, HEAD, "① 实验脚本与结果文件", fontsize=8.0, color="#16324F",
            fontweight="bold", va="center")
    ax.text(LEND, HEAD, "大小", fontsize=7.2, color="#4A6076", va="center", ha="right")
    ax.text(RX, HEAD, "② 实验规模与耗时", fontsize=8.0, color="#16324F",
            fontweight="bold", va="center")
    for x0, x1 in ((LX - 0.004, LEND + 0.004), (RX - 0.004, REND + 0.004)):
        ax.plot([x0, x1], [RULE, RULE], color="#2F4F6F", lw=0.7)

    # 左：文件清单（斑马纹）
    for i, (name, kb) in enumerate(files):
        y = Y0 - i * SL
        if i % 2 == 0:
            ax.add_patch(Rectangle((LX - 0.004, y - SL * 0.46), LEND - LX + 0.008,
                                   SL * 0.92, facecolor="#F4F7FB", edgecolor="none"))
        ax.text(LX, y, name, fontsize=7.3, color="#2A2A2A", va="center")
        ax.text(LEND, y, "%.1f KB" % kb, fontsize=6.9, color="#5A6B7C",
                va="center", ha="right")

    # 右：规模与耗时（最后一行加粗）
    for i, (k, v) in enumerate(rows):
        y = Y0 - i * SR
        last = i == len(rows) - 1
        if i % 2 == 0 and not last:
            ax.add_patch(Rectangle((RX - 0.004, y - SR * 0.46), REND - RX + 0.008,
                                   SR * 0.92, facecolor="#F4F7FB", edgecolor="none"))
        if last:
            ax.plot([RX - 0.004, REND + 0.004], [y + SR * 0.55, y + SR * 0.55],
                    color="#C9D6E4", lw=0.6)
        ax.text(RX, y, k, fontsize=7.2, color="#16324F" if last else "#2A2A2A",
                fontweight="bold" if last else "normal", va="center")
        ax.text(REND, y, v, fontsize=7.2, color="#16324F" if last else "#2A2A2A",
                fontweight="bold" if last else "normal", va="center", ha="right")

    ax.text(LX, 0.098,
            "数据：MVTec AD 胶囊类（219 张合格品、109 张缺陷）；运行记录见 stats.txt 与 results_*.json",
            fontsize=6.6, color="#5A6B7C", va="center")

    out = os.path.join(FIGS, "fig_record.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved fig_record.png  files=%d rows=%d 合计 %d 次 / %.0f 秒"
          % (len(files), len(rows), n_fit, t_all))


# ============================================================ 图 4 判定逻辑
def draw_logic():
    """对照组为什么必要：四条来源 → 同一套评测 → 三种结论。"""
    fig, ax = plt.subplots(figsize=(5.0, 2.02), dpi=320)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 2.26)
    ax.axis("off")

    BLUE = ("#E8F0F8", "#2F6FA8")
    GREEN = ("#E9F5EC", "#3E8E5A")
    GREY = ("#F2F4F7", "#8A97A6")
    PURPLE = ("#EFEAF6", "#7A5FA8")

    # 基线
    rbox(ax, 1.86, 1.94, 3.74, 2.22, *BLUE)
    ax.text(2.80, 2.08, "k 张真实缺陷（基线）", fontsize=7.6, color="#16324F",
            ha="center", va="center", fontweight="bold")

    # 四条来源
    cols = [
        (0.08, 1.34, "＋N 张合成缺陷", "原位／跨区剪贴", BLUE, 1),
        (1.50, 2.76, "＋N 张简单变换", "翻转·旋转·缩放", GREEN, 1),
        (2.92, 4.18, "＋N 张直接重复", "同一张复制", GREY, 0),
        (4.34, 5.60, "＋N 张随机噪声", "贴纯噪声块", GREY, 0),
    ]
    for x0, x1, t1, t2, (fc, ec), strong in cols:
        rbox(ax, x0, 1.26, x1, 1.66, fc, ec, lw=1.1 if strong else 0.8)
        cx = (x0 + x1) / 2
        ax.text(cx, 1.535, t1, fontsize=7.0, color=ec if strong else "#4A6076",
                ha="center", va="center", fontweight="bold" if strong else "normal")
        ax.text(cx, 1.375, t2, fontsize=6.0, color="#5A6B7C", ha="center", va="center")

    # 基线 → 四条来源
    ax.plot([2.80, 2.80], [1.94, 1.80], color="#5B6B7C", lw=0.9)
    ax.plot([0.71, 4.97], [1.80, 1.80], color="#5B6B7C", lw=0.9)
    for x in (0.71, 2.13, 3.55, 4.97):
        arrow(ax, (x, 1.80), (x, 1.66))

    # 四条来源 → 判定
    for x in (0.71, 2.13, 3.55, 4.97):
        ax.plot([x, x], [1.26, 1.12], color="#5B6B7C", lw=0.9)
    ax.plot([0.71, 4.97], [1.12, 1.12], color="#5B6B7C", lw=0.9)
    arrow(ax, (2.80, 1.12), (2.80, 1.00))

    # 判定
    rbox(ax, 0.08, 0.26, 5.52, 1.00, *PURPLE, lw=1.1)
    ax.text(0.22, 0.865, "同一个数据划分上比 Δ 得分", fontsize=6.8,
            color="#4B3A6B", ha="left", va="center", fontweight="bold")
    rows = [
        ("Δ合成明显高于 Δ变换", "提升来自合成内容本身", "#2F6FA8"),
        ("两者差不多", "提升只是样本数量变多", "#3E8E5A"),
        ("两者都不显著", "现有数据还分不出（不等于无效）", "#8A97A6"),
    ]
    for i, (a, b, c) in enumerate(rows):
        y = 0.660 - i * 0.155
        ax.text(0.22, y, "▪", fontsize=6.5, color=c, va="center")
        ax.text(0.40, y, a, fontsize=6.6, color="#2A2A2A", va="center")
        ax.text(2.42, y, "→", fontsize=6.6, color="#8A97A6", va="center")
        ax.text(2.66, y, b, fontsize=6.6, color=c, va="center", fontweight="bold")

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
    out = os.path.join(FIGS, "fig_logic.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved fig_logic.png")


# ============================================================ 图 8 进度时间轴
def draw_gantt():
    """12 个月、五个阶段、两个检查点。"""
    fig, ax = plt.subplots(figsize=(5.0, 1.62), dpi=320)
    ax.set_xlim(0, 5.6)
    ax.set_ylim(0, 1.72)
    ax.axis("off")

    X0, X1 = 1.32, 5.50
    M = (X1 - X0) / 12.0          # 每月宽度
    def mx(m):
        return X0 + m * M

    phases = [
        ("阶段一", "文献与固化", 0, 2, "#2F6FA8", "#E8F0F8"),
        ("阶段二", "强基线对齐", 2, 2, "#2F6FA8", "#E8F0F8"),
        ("阶段三", "主体实验", 4, 3, "#3E8E5A", "#E9F5EC"),
        ("阶段四", "分析与规则", 7, 2, "#C08434", "#FBF0E4"),
        ("阶段五", "验证与结题", 9, 3, "#C08434", "#FBF0E4"),
    ]
    for i, (no, name, s, w, ec, fc) in enumerate(phases):
        y = 1.44 - i * 0.235
        ax.text(0.06, y, no, fontsize=6.8, color=ec, va="center", fontweight="bold")
        ax.text(0.58, y, name, fontsize=6.0, color="#4A6076", va="center")
        ax.add_patch(FancyBboxPatch(
            (mx(s), y - 0.072), w * M, 0.144,
            boxstyle="round,pad=0.004,rounding_size=0.03",
            linewidth=0.8, edgecolor=ec, facecolor=fc))

    # 时间轴
    AY = 0.30
    ax.plot([X0, X1], [AY, AY], color="#5B6B7C", lw=0.9)
    for m, lab in [(0, "2026.12"), (3, "2027.03"), (6, "2027.06"),
                   (9, "2027.09"), (12, "2027.11")]:
        ax.plot([mx(m), mx(m)], [AY - 0.03, AY + 0.03], color="#5B6B7C", lw=0.8)
        ax.text(mx(m), AY - 0.10, lab, fontsize=5.8, color="#5A6B7C",
                ha="center", va="center")

    # 中期检查（外部硬节点）
    xm = mx(4.5)
    ax.plot([xm, xm], [0.24, 1.58], color="#C0392B", lw=0.9, linestyle=(0, (3, 2)))
    ax.text(xm + 0.05, 1.62, "4 月中旬 学校中期检查", fontsize=6.0,
            color="#C0392B", ha="left", va="center")

    # 阶段起点刻度
    for m in (2, 4, 7, 9):
        ax.plot([mx(m), mx(m)], [0.24, 1.52], color="#C9D6E4", lw=0.6,
                linestyle=(0, (2, 2)))

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
    out = os.path.join(FIGS, "fig_gantt.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved fig_gantt.png")


if __name__ == "__main__":
    draw_framework()
    draw_roadmap()
    draw_heatmap()
    draw_heatmap_ctrl()
    draw_compare()
    draw_record()
    draw_logic()
    draw_gantt()





