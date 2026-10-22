[English](README.md) | **中文**

# OCSR-complex：面向结构困难样本的封闭光学化学结构识别基准

**OCSR-complex** 是一套封闭评测基准，用来检验光学化学结构识别（OCSR，即从图像还原分子）在类药专利分子之外的表现。现有模型在 USPTO、Clef、UOB 等专利语料上常报告超过 90% 的精确匹配，但遇到立体化学密集、拓扑缠结、多中心配位的图时，准确率会明显下降。这类图在天然产物、全合成和有机金属文献里很常见。

为避免训练集污染和针对基准过拟合，**评测图像、抽样清单、化合物名录和标准答案保持不公开**。本仓库公开的是评测方法、11 个模型的 v2.0 排行榜与诊断分析（并与 v1.1 对照）、示例图（`samples/`），以及推理与评分代码（`tools/`）。

![示例图](samples/gallery.png)

*图 1：`samples/` 中的有机示例，用 v2.0 绘图管线重绘。图例沿用 v1.1 的有机分子（期刊样式与几何姿态），不含配合物。这些图不属于封闭评测集。*

---

## 1. 基准设计与 v2.0 评测协议

### 1.1 针对的结构失败模式
基准有意提高五类二维化学图的比例。这些图在视觉感知和拓扑重建上都容易含糊：

1. **桥环多环与笼状结构**。键画在桥头后方，或处在高张力多环笼中，二维上会严重交叉遮挡（`has_bridged_ring`、`has_cage`）。
2. **季碳螺环**。一个或多个螺结使局部键角拥挤，顶点度数难以判断（`has_spiro`）。
3. **大环与大型生物聚合物**。大环内酯、多烯、环肽和聚醚离子载体的关环与构象简并，会考验序列解码器和图解码器（`has_macrocycle`、`is_large`）。
4. **立体密集的天然产物**。大量楔形键、虚线键和有构型的烯烃，既要看清局部记号，也要在全局上与 CIP / SMILES 手性一致（`is_stereo_dense`、`has_e_z`）。
5. **配位与有机金属配合物**。单独成套（`coordination_benchmark/`），包括 η⁵/η⁶ 夹心配合物、金属大环空腔（卟啉、二氢卟吩、酞菁、可啉），以及 σ 键或螯合的过渡金属配合物（`is_organometallic`）。

### 1.2 从 v1.1 到 v2.0
在 v1.1 的经验和审计之上，**OCSR-complex v2.0** 做了四项改动：

* **有机集扩大，配合物分开评**。有机评测集的化学多样性增加了一倍以上。多中心配合物改到独立轨道（`coordination_benchmark/`），避免金属成键惯例混进有机准确率。
* **`acs1996` 下五个姿态全测**。v1.0/v1.1 的全网格实验表明，期刊样式只让准确率变动约 0.8–3.7 个百分点（七种模板的标准差为 2.2–3.2%），而再随机抽模板和姿态会引入抽样方差。v2.0 的正式协议是：在美国化学会标准样式 `acs1996` 下，**每个化合物的五个几何变体都测**（正位 `v0_orig`、面内旋转 `rz_a`/`rz_b`、面外透视倾斜 `tilt_x`/`tilt_y`）。
* **按 IUPAC 约束的算法绘图**。结构来自 [PubChem](https://pubchem.ncbi.nlm.nih.gov/)，并用 [Wikidata](https://www.wikidata.org/) 交叉核对。布局经过多种子二维搜索和三维投影修复，对齐 IUPAC 2006/2008 图示规范：保留 CIP 立体中心和有构型的 E/Z 双键，处理螺环氧与端基标签碰撞，再由 **ChemDraw** 栅格化，不做破坏性自动清理。
* **立体化学指标按分母校准**。没有手性中心或没有 E/Z 双键的分子，不再算进对应准确率的分母。四面体（`Chiral %`）只在有指定四面体立体中心的化合物上计算，双键（`E/Z %`）只在有构型双键的化合物上计算，另有忽略 E/Z、只看四面体的化合物加权指标（`Chirality No-E/Z %`）。

### 1.3 评分指标
* **有机主指标（`Organic Exact %`）**：先对每个化合物的五个姿态取平均，再对化合物取分层加权平均。紧凑的基线结构降权，使复杂骨架主导主指标。预测与标准答案的 RDKit 正则异构 SMILES 完全相同，才记为 `Exact`。
* **四面体手性（`Chiral %`）与双键几何（`E/Z %`）**：分别只在带有指定四面体立体中心、或带有构型 E/Z 烯烃的化合物上计算。
* **保留手性、忽略 E/Z（`Chirality No-E/Z %`）**：去掉双键方向记号（`/` 和 `\`），保留全部四面体记号（`@`/`@@`）后再比正则 SMILES。用来把 E/Z 惯例差异和四面体翻转分开。
* **互变异构（`Tautomer %`）**：在可枚举的重原子范围内，用 RDKit `TautomerEnumerator` 比较正则互变异构 InChIKey。酮–烯醇、酰胺–亚胺醇等合理互变异构不算拓扑错误。
* **连接性（`Connectivity %`）**：忽略立体化学的正则 SMILES 是否相同，相当于有机分子上 InChIKey 前 14 位骨架块是否相同。衡量二维分子图是否恢复。
* **结构相似度（`ECFP4 Tanimoto %`）**：Morgan 指纹（半径 2，2048 位）的 Tanimoto 相似度，按化合物加权。无法解析的输出记 0。用来衡量近失预测还保留了多少结构。
* **配合物配体骨架分（`Coordination Parts %`）**：在独立的配合物模块上，用规范的 **part key**（`coord_depict.py`）核对金属中心、非 haptic 的 σ/螯合配位层，以及各配体经电荷和自由基归一后的骨架。

---

## 2. 评测结果（11 个模型）

### 2.1 v2.0 有机排行榜及与 v1.1 的对照
11 个开放权重模型分三类：

1. **一维化学线性记法自回归解码器**：**DECIMER**、**COMO**、**MolParser-Mobile-V2**、**OCSRGlyph**、**MolSight**、**MolSight-stereo**。
2. **二维分子图重建混合模型**：**MolScribe**、**MolNexTR**、**MolGrapher**。
3. **视觉–语言模型（VLM）**：**TinyChemVL**、**MarkushGlyph**。

| 模型 | 范式与骨干 | 有机可解析 % | **v2.0 有机精确 %**（95% CI） | v1.1 精确 % | **连接性 %**（95% CI） | 手性 % | E/Z % | 手性（忽略 E/Z）% | 互变异构 % | ECFP4 Tanimoto % | 姿态标准差 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DECIMER** (v2.7.1) | EfficientNetV2 + Transformer → **SELFIES** | **95.4** | **59.1** (±3.4) | 58.6 | **78.4** (±2.8) | **61.2** | **66.1** | **62.1** | **64.3** | **90.5** | 0.64 |
| **MolScribe** | Swin-B + **图**自回归解码器 | 74.0 | 45.1 (±3.3) | 41.1 | 63.5 (±3.2) | 50.9 | 39.0 | 51.6 | 52.1 | 70.3 | 1.33 |
| **MolNexTR** | ConvNeXt + ViT + **图/SMILES** 解码器 | 76.0 | 44.0 (±3.2) | 41.1 | 59.4 (±3.1) | 49.0 | 35.3 | 49.5 | 49.0 | 69.7 | 0.62 |
| **COMO** (Tanimoto-RL) | Swin-B + 6 层解码器 + 图立体后处理 | 78.3 | 37.5 (±3.3) | 35.5 | 66.8 (±3.2) | 56.4 | 0.0 | 57.3 | 42.8 | 75.2 | **0.31** |
| **MolParser-Mobile-V2** | ViT-Tiny + 6 层解码器（10.0M）→ **E-SMILES** | 80.1 | 25.6 (±2.8) | 27.9 | 43.9 (±3.2) | 26.7 | 26.3 | 27.7 | 29.8 | 66.9 | 0.71 |
| **OCSRGlyph** | Swin-B/384 + Transformer 解码器 → **SMILES** | 77.1 | 19.2 (±2.5) | 18.0 | 49.3 (±3.3) | 21.0 | 19.8 | 23.5 | 22.7 | 68.4 | 0.59 |
| **MolSight** | EfficientViT-L1 + 6 层解码器（67.5M，Char+Edge） | 77.8 | 19.0 (±2.6) | 20.2 | 45.2 (±3.4) | 18.7 | 25.8 | 20.9 | 22.7 | 66.6 | 0.46 |
| **MolSight-stereo** | EfficientViT-L1 + 6 层解码器 + Stereo-GRPO LoRA | 79.0 | 18.2 (±2.6) | 19.1 | 44.5 (±3.4) | 18.0 | 25.3 | 20.3 | 22.2 | 66.6 | 0.52 |
| **MolGrapher** | ResNet-18 关键点 + **GCN** 分类器 + OCR | 85.5 | 8.1 (±1.8) | 6.1 | 44.2 (±3.3) | 6.5 | 9.8 | 9.6 | 10.1 | 57.9 | 0.63 |
| **TinyChemVL** | InternViT-6B + Qwen2.5-3B（约 9B VLM）→ **SMILES** | 91.4 | 7.2 (±1.7) | 14.6 | 13.8 (±2.2) | 6.9 | 1.8 | 8.2 | 8.7 | 46.3 | 0.43 |
| **MarkushGlyph** | Qwen3.5-2B-Base + LoRA VLM → **CXSMILES** | 58.7 | 2.7 (±1.0) | 5.2 | 13.6 (±2.3) | 1.4 | 3.0 | 3.0 | 4.6 | 38.7 | 0.36 |

> **统计分辨力与紧凑层校准：** 置信区间来自 10,000 次化合物级配对 bootstrap。v2.0 `Organic Exact %` 的 55 对模型比较中，**50 对（90.9%）在 Holm 逐步校正后于 α = 0.05 显著**（v1.1 为 45/55）。仍不可区分的五对是：**MolScribe** 与 **MolNexTR**（Δ = +1.1%，p_Holm = 1.00），**OCSRGlyph** / **MolSight** / **MolSight-stereo** 这一簇（Δ ≤ 1.0%，p_Holm = 1.00），以及精确匹配上的 **MolGrapher** 与 **TinyChemVL**（Δ = +0.9%，p_Holm = 1.00；但 **MolGrapher** 的连接性高出 **TinyChemVL** 30.4 个百分点）。紧凑有机层上精确率高得多（**DECIMER** 67.1%，**MolScribe** 58.8%，**MolParser-Mobile-V2** 54.1%，**COMO** 50.6%，**MolNexTR** 47.1%，**OCSRGlyph** 41.2%，**MolGrapher** 41.2%，**MolSight** 35.3%，**MolSight-stereo** 35.3%，**TinyChemVL** 32.9%，**MarkushGlyph** 22.4%）。给简单结构降权，是为了不让容易的类药分子盖住复杂骨架上的失败。

---

### 2.2 不同结构难度上的表现（v2.0）
表 2 是 v2.0 有机集上各难度标签的化合物加权准确率（`精确 % / 连接性 %`）。

| 模型 | 桥环 | 螺环 | 大环 | 多环笼 | 立体密集 | 大分子 | 有构型 E/Z | 桥环且立体密集 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DECIMER** | **44.7 / 69.2** | **50.4 / 74.6** | **43.1 / 67.2** | **25.8 / 55.7** | **55.0 / 76.2** | **42.1 / 64.2** | **46.9 / 73.8** | **39.2 / 67.5** |
| **MolScribe** | 29.2 / 46.9 | 29.9 / 50.1 | **25.3** / 53.4 | 17.5 / 28.3 | **40.5** / 59.6 | 27.0 / 45.3 | **30.4** / 58.7 | 23.6 / 43.0 |
| **MolNexTR** | **30.7** / 46.5 | **33.4** / 49.7 | 24.0 / 51.3 | **18.1 / 28.6** | 38.8 / 56.1 | **27.4** / 45.6 | 28.0 / 55.8 | **26.1 / 44.1** |
| **COMO** | 19.1 / **48.2** | 28.8 / **50.4** | 7.3 / **60.8** | 7.1 / **31.8** | 31.5 / **63.1** | 20.1 / **50.5** | 0.0 / **66.9** | 13.3 / 43.5 |
| **MolParser-Mobile-V2** | 14.0 / 30.2 | 14.7 / 34.6 | 13.8 / 26.6 | 4.7 / 10.6 | 18.1 / 37.3 | 5.4 / 11.5 | 15.3 / 32.0 | 8.5 / 24.8 |
| **OCSRGlyph** | 8.0 / 33.4 | 11.7 / 35.6 | 8.7 / 36.9 | 3.1 / 12.7 | 12.3 / 44.6 | 9.4 / 22.7 | 8.8 / 44.8 | 3.8 / 29.7 |
| **MolSight** | 9.0 / 29.2 | 8.5 / 31.3 | 7.4 / 35.2 | 1.8 / 6.6 | 12.1 / 39.3 | 7.8 / 19.2 | 11.5 / 38.2 | 5.2 / 24.5 |
| **MolSight-stereo** | 8.7 / 28.7 | 8.5 / 29.5 | 8.1 / 34.0 | 2.5 / 8.2 | 10.5 / 38.5 | 6.1 / 16.7 | 12.1 / 37.1 | 4.9 / 24.0 |
| **MolGrapher** | 4.5 / 30.8 | 6.2 / **35.7** | 1.4 / 32.1 | 1.0 / 17.0 | 4.0 / 38.6 | 1.7 / 14.3 | 1.8 / 36.2 | 1.3 / 27.0 |
| **TinyChemVL** | 3.9 / 6.7 | 2.9 / 7.1 | 0.0 / 2.2 | 0.9 / 1.1 | 4.1 / 8.7 | 0.6 / 1.7 | 0.2 / 5.3 | 1.2 / 4.1 |
| **MarkushGlyph** | 0.7 / 4.1 | 0.8 / 5.7 | 0.0 / 3.9 | 0.3 / 0.5 | 0.2 / 8.2 | 0.0 / 1.8 | 0.6 / 9.9 | 0.0 / 2.6 |

---

### 2.3 错误分类与立体化学保真度（v2.0）
表 3 把 v2.0 全部有机图像预测划成五个互斥结果，并给出条件立体精度（`Exact / Connectivity`，以及忽略 E/Z 的对应值）、拓扑错误但仍可解析时的近失相似度，以及骨架错误上的重原子数偏差中位数。

| 模型 | 精确匹配 % | E/Z 立体损失 % | 四面体或双重立体损失 % | 骨架 / 拓扑错误 % | 无法解析 % | 条件立体精度 % *（忽略 E/Z）* | 骨架错误平均 Tanimoto % | 骨架错误的 ΔN_heavy 中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **DECIMER** | **59.2** | 3.1 | 16.0 | 17.2 | **4.5** | **75.6** *（79.6）* | 70.0 | **1.0** |
| **MolScribe** | 45.3 | 6.4 | 11.7 | **10.6** | 26.0 | 71.5 *（81.6）* | 63.1 | **1.0** |
| **MolNexTR** | 44.0 | 5.5 | 9.7 | 16.8 | 24.0 | 74.4 *（83.6）* | 60.8 | **1.0** |
| **COMO** | 37.7 | 19.7 | **9.3** | 11.4 | 21.9 | 56.5 *（**86.0**）* | **73.1** | **1.0** |
| **MolParser-Mobile-V2** | 26.1 | 2.1 | 15.9 | 36.1 | 19.8 | 59.1 *（63.9）* | 63.6 | 2.0 |
| **OCSRGlyph** | 19.6 | 4.3 | 25.4 | 27.9 | 22.8 | 39.8 *（48.5）* | 68.4 | **1.0** |
| **MolSight** | 19.3 | 1.9 | 23.9 | 33.0 | 21.9 | 42.7 *（46.9）* | 65.1 | 2.0 |
| **MolSight-stereo** | 18.5 | 2.1 | 23.9 | 34.8 | 20.7 | 41.5 *（46.3）* | 63.7 | 2.0 |
| **MolGrapher** | 8.7 | 1.7 | 34.0 | 41.4 | 14.2 | 19.5 *（23.3）* | 33.0 | **1.0** |
| **TinyChemVL** | 7.6 | 1.1 | 5.6 | 77.2 | 8.5 | 53.2 *（60.9）* | 42.0 | 5.0 |
| **MarkushGlyph** | 3.0 | 0.4 | 10.6 | 45.0 | 41.0 | 21.5 *（24.7）* | 55.4 | 2.0 |

---

### 2.4 按成键类型划分的配合物识别
表 4 是独立配合物轨道上三类成键的表现（`Coordination Parts %` / `Valid %`）。

| 模型 | η⁵/η⁶ 夹心配合物 | σ 键 / 螯合物 | 金属大环空腔 | 配合物总分 % |
|---|---:|---:|---:|---:|
| **DECIMER** | **100.0** / 100.0 | 25.0 / 83.3 | 0.0 / 44.8 | **40.8** |
| **MolSight-stereo** | 91.7 / 7.1* | 25.0 / 91.7 | 0.0 / 9.4 | 37.7 |
| **OCSRGlyph** | 67.9 / 4.8* | **33.3** / **100.0** | 0.0 / 24.0 | 30.3 |
| **MolSight** | 64.3 / 7.1* | 22.2 / 94.4 | 0.0 / 12.5 | 27.2 |
| **MarkushGlyph** | 25.0 / 57.1 | **33.3** / 72.2 | 0.0 / 8.3 | 14.5 |
| **MolNexTR** | 10.7 / 7.1* | 16.7 / 69.4 | 0.0 / 24.0 | 6.6 |
| **COMO** | 7.1 / 7.1 | 16.7 / 66.7 | 0.0 / 7.3 | 5.3 |
| **MolParser-Mobile-V2** | 0.0 / 92.9 | 25.0 / 83.3 | 0.0 / 17.7 | 3.9 |
| **TinyChemVL** | 3.6 / 91.7 | 13.9 / 94.4 | 0.0 / 74.0 | 3.5 |
| **MolScribe** | 4.8 / 7.1* | 8.3 / 69.4 | 0.0 / 26.0 | 3.1 |
| **MolGrapher** | 0.0 / 100.0 | 0.0 / 97.2 | 0.0 / 42.7 | 0.0 |

*\*注：SMILES 解码器写出的未取代芳环茂（`c1cccc1`）通不过 RDKit 默认的凯库勒检查（`Valid %`），但配合物 `parts` 评分能正确识别。*

---

## 3. 主要结论（v1.1 → v2.0）

1. **姿态和字体几乎不影响结果，拓扑缠结才会。**  
   v1.1 比较了七种期刊样式，v2.0 对每个化合物测了五个姿态。排版和几何扰动对识别影响很小：v2.0 里，面内旋转和面外倾斜的标准差，11 个模型中有 10 个不超过 0.71%（**COMO** 0.31%，**MolSight** 0.46%，**MolNexTR** 0.62%，**DECIMER** 0.64%，**MolScribe** 1.33%）。从紧凑有机骨架换到多环笼或大环，各架构都会掉 30–60 个百分点。现有视觉编码器对姿态和字体是稳的；**瓶颈在化学图拓扑和立体化学解码**。

2. **无语法约束的一维表示（SELFIES）能避免大分子上的语法崩溃。**  
   **DECIMER** 在 v2.0 有机主指标上领先一截（精确 59.1%，连接性 78.4%，互变异构 64.3%，化合物加权 ECFP4 Tanimoto 90.5%）。v2.0 修了小环和螺环氧的拥挤，并对桥环笼使用三维 Schlegel/PCA 投影之后，**DECIMER** 的连接性比 v1.1 提高 7.4 个百分点（71.0% → 78.4%），笼状精确率近乎翻倍（14.3% → 25.8%；连接性 34.9% → 55.7%）。SMILES 序列模型和图构建模型在复杂天然产物上有 19.8–41.0% 无法解析，**DECIMER** 的 SELFIES 则保证 95.4% 语法有效，把可能的语法崩溃变成高相似度的近失（骨架错误上平均 Tanimoto 70.0%）。

3. **二维图重建混合模型从绘图修复中受益，骨架错误最少。**  
   有机集扩大、二维键间距收紧之后，**MolScribe**（精确 41.1% → 45.1%，连接性 53.8% → 63.5%）和 **MolNexTR**（精确 41.1% → 44.0%，连接性 49.7% → 59.4%）都上升。大分子（至少 60 个重原子：精确 8.3–12.4% → 27.0–27.4%）和大环（精确 16.4–22.9% → 24.0–25.3%）上的增益尤其大。**MolScribe** 的拓扑骨架错误率最低（10.6%），**COMO**（11.4%）和 **MolNexTR**（16.8%）次之。没有序列恢复、只做自下而上关键点和 GCN 的 **MolGrapher**（精确 8.1%，连接性 44.2%）仍怕键交叉和大环：预测图一旦断成多片段，就退回单碳哨兵 `C`。这占 v2.0 全部图的 16.9%（笼和大环 24.3%，大分子 39.9%，超过 100 个重原子的结构 83.8%）。

4. **按分母分开的 E/Z 与四面体指标，暴露了后处理取舍。**  
   `Chiral %` 和 `E/Z %` 只在各自有该立体特征的化合物上计算，四面体宇称和烯烃几何就分开了：
   * **COMO** 的官方后处理从辅助图头搬四面体楔形/虚线，同时故意丢掉顺反记号（`ignore_cistrans=True`）。因此 `E/Z %` 为 **0.0%**，全部图像里有 19.7% 只因 E/Z 丢失而失败。四面体上 **COMO** 达到 **56.4%**（仅次于 **DECIMER** 的 61.2%，高于 **MolScribe** 50.9% 和 **MolNexTR** 49.0%）。忽略 E/Z 后从 37.5% 升到 57.3%（+19.8 个百分点），条件无 E/Z 立体精度 **86.0%**、骨架错误 Tanimoto **73.1%**，都是全场最高。
   * 一维 SMILES 解码器里，**MolSight**（E/Z 25.8%）和 **MolParser-Mobile-V2**（E/Z 26.3%）比 **OCSRGlyph**（E/Z 19.8%）更能保住双键几何；**OCSRGlyph** 则恢复更多宪法骨架（连接性 49.3%，对照 45.2% 和 43.9%）。

5. **复杂集一扩大，通用 VLM 和窄域后训练的泛化上限就露出来。**  
   从 v1.1 到 v2.0，专用 OCSR 架构持平或上升，通用 VLM 和马库什专用 VLM 大约掉了一半：
   * **TinyChemVL**（约 9B）精确率从 14.6% 降到 7.2%，连接性从 24.1% 降到 13.8%（大环精确 0.0%，有构型 E/Z 烯烃 0.2%）。语法有效率仍有 91.4%，但 77.2% 的输出是拓扑幻觉，重原子数偏差中位数 |ΔN_heavy| = 5.0（**DECIMER**、**MolScribe**、**MolNexTR**、**COMO**、**OCSRGlyph**、**MolGrapher** 为 1.0）。
   * **MarkushGlyph** 精确率从 5.2% 降到 2.7%（无法解析 41.0%，骨架错误 45.0%）。面向专利马库什 CXSMILES 的微调，会在复杂天然产物拓扑上严重遗忘。
   * **MolSight** 与 **MolSight-stereo**：和 v1.1 一致，在合成立体异构体上做 GRPO LoRA，语法有效率略升（77.8% → 79.0%），夹心配合物也上升（27.2% → 37.7%），但有机精确率下降（19.0% → 18.2%，p_Holm = 1.00），立体密集骨架也下降（12.1% → 10.5%）。

6. **金属大环空腔仍然没有模型能做对。**  
   在独立配合物轨道上，序列模型能识别 η⁵/η⁶ 夹心结构（**DECIMER** 100.0%，**MolSight-stereo** 91.7%，**OCSRGlyph** 67.9%，**MolSight** 64.3%），也能部分恢复 σ 键/螯合过渡金属（最高 33.3%）。卟啉、二氢卟吩、酞菁和可啉上，**11 个模型都是 0.0%**。原因是中心 M–N₄ 配位键穿过共轭大环。

7. **不同架构互补，并集明显高于单模型。**  
   v2.0 有机网格上，11 个模型的并集精确解出 **73.8%** 的图（化合物加权精确同样 73.8%，图像级连接性 85.9%；至少一个姿态被精确解出的化合物占 **80.0%**，按连接性为 **88.8%**）。比最好的单模型 **DECIMER**（59.1%）高 14.7 个百分点。有 9 个模型贡献了其余 10 个模型都没有解出的精确预测：**DECIMER**、**MolScribe**、**MolNexTR**、**COMO**、**MolSight**、**MolParser-Mobile-V2**、**OCSRGlyph**、**MarkushGlyph**、**MolSight-stereo**。SELFIES 序列解码、二维图重建和辅助图立体后处理之间互补明显。

---

## 4. 模型来源与评测细节

全部模型使用官方发布的权重和后处理，确定性解码（贪心，beam size 为 1）：

* **DECIMER**（`v2.7.1`，[Kohulan/DECIMER-Image_Transformer](https://github.com/Kohulan/DECIMER-Image_Transformer)，MIT）：TensorFlow SavedModel（`DECIMER.predict_SMILES`），把预测的 SELFIES 转成正则 SMILES。
* **MolScribe**（[yujieq/MolScribe](https://huggingface.co/yujieq/MolScribe)，MIT）：权重 `swin_base_char_aux_1m680k.pth`，采用官方图到 SMILES 的输出。
* **MolNexTR**（[CYF200127/MolNexTR](https://huggingface.co/datasets/CYF200127/MolNexTR)，Apache 2.0）：权重 `molnextr_best.pth`，采用官方图到 SMILES 的输出。
* **COMO**（[Keylab/COMO](https://huggingface.co/Keylab/COMO)，CC BY-NC 4.0）：Tanimoto 奖励权重 `models/tanimoto/best.pth`。评分用官方 `postprocess` SMILES：从预测图搬四面体立体中心，正则化时忽略顺反；图立体恢复失败时退回 `decode_smiles`。
* **MolParser-Mobile-V2**（[UniParser/MolParser-Mobile-V2](https://huggingface.co/UniParser/MolParser-Mobile-V2)，CC BY-NC-SA 4.0）：1000 万参数，FP16，`max_length=384`。评分取 E-SMILES 里 `<sep>` 或 `|` 之前的 SMILES 主干。
* **OCSRGlyph**（`EdisonScientific/glyph`）：Swin-B/384 + Transformer 解码器，开启默认后处理。
* **MolSight 与 MolSight-stereo**（[hustvl/MolSight](https://github.com/hustvl/MolSight)，[Robert-zwr/MolSight](https://huggingface.co/Robert-zwr/MolSight)）：分别使用 `pubchem_uspto_smiles_edges_10.pth`（`--formats char,edges`）和 Stereo-200k GRPO LoRA 权重 `stereo_grpo_10_2.pth`（`--lora --formats char`）。评分前做上游官能团展开（`_postprocess_smiles`）。
* **MolGrapher**（[DS4SD/MolGrapher](https://github.com/DS4SD/MolGrapher)，MIT）：节点分类器用 `gc_stereo_model`，并加 `--assign-stereo`。经 `tools/setup_molgrapher_wsl.py` 修补后，在现行 PyTorch/CUDA（`torch 2.11+cu128`）上加载未改动的权重；标题和标签 OCR 用 CPU 上的 PaddleOCR。
* **TinyChemVL**（`InternViT-6B + Qwen2.5-3B`）：BF16。每个 batch 前把 `<image>` 拼进提示，并修正会被原地改写的 `num_image_token`（`tools/predict_tinychemvl.py`）。
* **MarkushGlyph**（[EdisonScientific/MarkushGlyph](https://huggingface.co/EdisonScientific/MarkushGlyph)，Apache 2.0）：在 `Qwen/Qwen3.5-2B-Base` 上的 LoRA，提示为 `OCSR_NO_EXTENSION_PROMPT`。评分取 `|` 扩展之前的 CXSMILES 核心。

---

## 5. 仓库结构与工具

```text
.
├── samples/
│   ├── gallery.png                 # 示例拼图
│   ├── ground_truth.csv            # 仅这些示例图的异构 SMILES、InChI 和结构描述
│   └── <image_id>.png              # 单张示例（不属于封闭评测集）
└── tools/
    ├── eval_ocsr.py                # v2.0 主评分（Exact、Chiral、E/Z、Tautomer、Connectivity、Parts）
    ├── eval_extra.py               # 错误分类、忽略 E/Z 的手性、ECFP4 Tanimoto
    ├── stats_ci.py                 # 化合物级配对 bootstrap 95% CI 与 Holm 校正
    ├── compare_models.py           # 多模型对照
    ├── make_random_split.py        # v1.1 的分层抽样切片
    ├── predict_decimer.py          # DECIMER 2.7.1
    ├── predict_molscribe.py        # MolScribe
    ├── predict_molnextr.py         # MolNexTR
    ├── predict_como.py             # COMO
    ├── predict_molparser.py        # MolParser-Mobile-V2
    ├── predict_ocsrglyph.py        # OCSRGlyph
    ├── predict_molsight.py         # MolSight 与 MolSight-stereo
    ├── predict_molgrapher.py       # MolGrapher（可分片、可断点续跑）
    ├── setup_molgrapher_wsl.py     # 在现行 PyTorch/CUDA 上跑 MolGrapher 的补丁
    ├── predict_tinychemvl.py       # TinyChemVL
    └── predict_markushglyph.py     # MarkushGlyph
```

### 推理与评分
所有 `predict_*.py` 接受清单（`--manifest`）、图像根目录（`--images-root`）和可选的模板过滤（`--template acs1996`），写出可续跑的 JSONL（`{"image_id": "<id>", "prediction": "<SMILES>"}`）：

```bash
# 按 v2.0 协议给一份 ACS 1996 预测评分
python tools/eval_ocsr.py \
    --manifest path/to/manifest_all.jsonl \
    --predictions predictions/v2_acs1996/decimer.jsonl \
    --template acs1996 \
    --out results/v2_acs1996_decimer

# 生成多模型对照
python tools/compare_models.py \
    --manifest path/to/manifest_all.jsonl \
    --template acs1996 \
    --preds decimer=predictions/v2_acs1996/decimer.jsonl,molscribe=predictions/v2_acs1996/molscribe.jsonl,molnextr=predictions/v2_acs1996/molnextr.jsonl \
    --out results/comparison_v2_acs1996.md
```

---

## 6. 方法说明与局限

* **配合物分开评**。haptic 键和配位键没有唯一的 SMILES，标准 InChI 对金属成键的归一也不好。配合物留在独立轨道，用 `parts` 键评分，不并进有机主指标。
* **版本之间不能直接当成同一次抽样**。v1.1 是多种模板的随机切片（`random_seed42`）。v2.0 在扩大后的有机集上、于 `acs1996` 下测全部五个几何变体，立体化学指标也按分母校准。v1.1 与 v2.0 的对照同时反映了化学难度的变化和绘图、评分协议的变化。
* **同系物混合物不收入**。PubChem/Wikidata 里以同系物混合物存放的记录不进入基准，只保留成分明确的单一化合物。
* **图是算法生成的**。全部图由经过校验的化学图生成，再按 ChemDraw 样式栅格化，不是从受版权保护的期刊 PDF 裁下来的。这样标准答案和画面一致，也把结构识别和版面分割分开。
* **许可**。本仓库的代码、评分脚本和示例文件采用 [MIT 许可证](LICENSE)。ChemDraw 是 Revvity Signals 的商业软件。分子记录来自 [PubChem](https://pubchem.ncbi.nlm.nih.gov/)（NCBI），并与 [Wikidata](https://www.wikidata.org/) 交叉核对。
