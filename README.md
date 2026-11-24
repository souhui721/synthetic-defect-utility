# 合成缺陷数据的效用评测（预实验代码与结果）

工业视觉质检里，缺陷样本极少。近五年的主流做法是"造数据"——用剪贴、纹理叠加或生成模型合成缺陷图，再和少量真实缺陷一起训练，文献普遍报告"有提升"。

这个仓库是**在把提升当真之前，先检验它**：把"加合成样本后检测得分的变化"放到受控条件下测一遍，并且把"提升来自合成内容"和"提升来自样本变多"这两件事分开。

> 预实验部分，配套大学生创新创业训练计划项目《"造数地图"——合成缺陷效用评测》。

---

## 主要发现

在 MVTec AD 胶囊类上，80 组条件（5 种样本来源 × 4 种真实缺陷数 × 4 种合成数量）各重复 10 次（5 折 × 2 种子）：

1. **没有一组提升通过多重比较校正**（Benjamini–Hochberg，80 组最小 q = 0.394）。以现有数据量，区分不出"合成有用"与"合成无效"。
2. **真实缺陷的简单变换**（翻转／旋转／缩放／调色）**与精心构造的合成缺陷效果无差别**。以 2 张真实缺陷为例，两种来源的得分增量分别为 +0.046 与 +0.029，校正后均不显著。这提示：观察到的变化更可能来自"缺陷样本变多了"，而不是"合成内容有用"。
3. **免标注方法仍然更高**。只用正常品训练的 PatchCore 得分为 0.9665；"16 张真实缺陷 + 合成样本"训练的最优配置为 0.9232（四组配置均值 0.908）。

第 2 条是这个仓库最想留下的东西：**如果没有"真实缺陷简单变换"这组对照，"合成数据有效"这两个字会被写进论文。**

---

## 环境

- Windows 11 / Python 3.12（其他平台未测试）
- 依赖见 `requirements.txt`。核心是 `torch` + `torchvision`（特征提取）、`scikit-learn`（线性探针）、`opencv-python`（跨区剪贴的梯度能量）、`scipy`（配对 t 检验）、`matplotlib`（作图）。

```bash
pip install -r requirements.txt
```

## 数据与权重

需要自己准备，不入库。

**MVTec AD**（capsule 类）：<https://www.mvtec.com/company/research/datasets/mvtec-ad>

```
data/mvtec/capsule/
├── train/good/          219 张合格品
├── test/good/            23 张合格品
└── test/<缺陷类型>/      109 张缺陷（crack / faulty_imprint / poke / scratch / squeeze）
```

**ImageNet 预训练权重**（torchvision 官方，放到 `weights/`）：

- `resnet18-f37072fd.pth`
- `wide_resnet50_2-95faca4d.pth`

## 复现

三条命令，本机（单卡笔记本）合计约 7 分钟：

```bash
# 1. 主网格：原位剪贴 + 跨区剪贴，400 次训练，约 147 s
python exp_mvp.py

# 2. 对照实验：五种来源（2 种合成 + 变换 / 重复 / 噪声），840 次训练，约 234 s
python exp_controls.py

# 3. 强基线：wide_resnet50_2 特征 + PCA + 有标注分类器，400 次训练，约 49 s
python exp_strongsup.py
```

结果写入 `results/`，其中 `stats.txt` 是主网格的逐条记录。重画论文图：

```bash
python plot_paper_figs.py     # 输出到 figures/
```

随机种子、数据划分、折数全部写死在脚本里并记录在结果的 `args` 字段中，同一台机器上重跑数字一致。

## 目录结构

```
.
├── exp_mvp.py            主网格：CutPaste / CutPaste-Salient × 真实缺陷数 × 合成数量
├── exp_controls.py       对照：真实缺陷简单变换、直接重复、随机噪声贴块
├── exp_strongsup.py      强基线：有标注分类器 vs PatchCore（同一套特征）
├── exp_patchcore.py      独立实现的 PatchCore 基线（像素级 AUROC）
├── plot_results.py       生成 stats.txt 与早期图表
├── plot_paper_figs.py    生成论文/申报书用图
├── results/              三份原始结果 json + stats.txt
├── figures/              已生成的图（可直接看）
├── data/                 数据集放这里（不入库）
└── weights/              预训练权重放这里（不入库）
```

## 结果摘要

主网格（`results/stats.txt`，Δ 为同一数据划分上"加合成"减"不加合成"）：

| 真实缺陷数 | 合成数 | 得分 | Δ 得分 | 配对 t 检验 p |
|---:|---:|---:|---:|---:|
| 2 | 0 | 0.6541 | — | — |
| 2 | 50 | 0.7045 | +0.0504 | 0.088 |
| 2 | 100 | 0.7093 | +0.0553 | 0.070 |
| 2 | 200 | 0.6965 | +0.0424 | 0.101 |
| 4 | 25 | 0.6944 | −0.0443 | 0.110 |

看得出"2 张真实缺陷时加上去有正向，但 p 值过不了 0.05"——这正是需要更大样本才能判定的区域。上表为**未校正**的 p 值；对全部 80 组做 Benjamini–Hochberg 校正后，**最小 q = 0.394**，没有一组通过。

强基线（`results/results_strongsup.json`，N ≥ 25 的均值 ± 标准差，每组 80 次）：

| 真实缺陷数 | 有标注（真缺陷 + 合成） | 免标注（PatchCore） |
|---:|---:|---:|
| 2 | 0.6853 ± 0.119 | 0.9665 |
| 4 | 0.7442 ± 0.095 | 0.9665 |
| 8 | 0.8649 ± 0.062 | 0.9665 |
| 16 | 0.9078 ± 0.048 | 0.9665 |

## 已知限制

- **跨区剪贴（CutPaste-Salient）已修过一轮，仍不是严格实现。** 早期版本把"低梯度"当成"背景"，而胶囊本体也是低梯度的；盒式模糊给出的窗口中心又被当成左上角使用，取块与贴块整体偏移半个窗口；取块位置由 `argmax` 唯一确定，同一张图的合成样本几乎一样。现已改为：以图像四条边的中位数估计背景色、用积分图按窗口左上角遍历打分、从 top-K 候选随机取块并在"高分带"内均匀随机贴块。它仍是一种启发式规则，落点只保证"靠近背景、避开取样区"，不保证语义上是缺陷。
- **只在单一类别（胶囊）上跑完 80 组网格**，跨类别结论尚未验证。
- **有标注一侧用线性探针**，不是经过完整调参的检测网络，得分偏低是预期的。
- 结果 json 里的 `args` 字段记录了作者本机的绝对路径，不影响复现。

## 引用

- Bergmann et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.
- Li et al. *CutPaste: Self-Supervised Learning for Anomaly Detection and Localization.* CVPR 2021.
- Roth et al. *Towards Total Recall in Industrial Anomaly Detection (PatchCore).* CVPR 2022.
- Zavrtanik et al. *DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection.* ICCV 2021.

## 许可

代码用于学术研究与教学，欢迎复现和讨论。引用或使用前请先联系作者。
