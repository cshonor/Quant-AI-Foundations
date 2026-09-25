# PSTAT_Basic — 茆诗松《概率论与数理统计教程》

**定位**：概率统计**底层基础轨**。优先补完各章推导、捡回遗忘部分，为计量、时序与 [`PSTAT_Quant`](../PSTAT_Quant/README.md)（Shiryaev 随机金融）打底。

---

## 仓库说明

本目录为茆诗松《概率论与数理统计教程》的学习记录，包含各章节笔记、关键公式推导、习题解答与 Python 实现。

**公式与上下标**：优先使用 HTML **`<sub>`**、**`<sup>`**（不是 LaTeX）；说明见仓库根目录 [`书写约定-上下标与公式.md`](../书写约定-上下标与公式.md)。第 4 章极限定理等已按此约定排版。

---

## 目录结构

```text
PSTAT_Basic/
├── 01_RANDOM_EVENTS                  # 第1章 随机事件与概率
├── 02_RANDOM_VARIABLES               # 第2章 随机变量及其分布
├── 03_MULTIVARIATE_DISTRIBUTIONS     # 第3章 多维随机变量及其分布
├── 04_LIMIT_THEOREMS                 # 第4章 大数定律与中心极限定理
├── 05_STATISTICS_DISTRIBUTIONS       # 第5章 统计量及其分布
├── 06_PARAMETER_ESTIMATION           # 第6章 参数估计
├── 07_HYPOTHESIS_TESTING             # 第7章 假设检验
├── 08_ANOVA_REGRESSION               # 第8章 方差分析与回归分析
└── README.md                         # 本说明文件
```

**第5章** 入口：`PSTAT_Basic/05_STATISTICS_DISTRIBUTIONS/README.md`  
**第6章** 入口：`PSTAT_Basic/06_PARAMETER_ESTIMATION/README.md`  
**第7章** 入口：`PSTAT_Basic/07_HYPOTHESIS_TESTING/README.md`  
**第8章** 入口：`PSTAT_Basic/08_ANOVA_REGRESSION/README.md`

---

## 章节学习目标

| 文件夹 | 核心目标 | 关联后续内容 |
| :--- | :--- | :--- |
| 01_RANDOM_EVENTS | 建立概率公理体系，掌握条件概率、贝叶斯公式 | 所有概率模型的基础 |
| 02_RANDOM_VARIABLES | 掌握离散/连续分布，理解随机变量函数分布 | 时序分布、量化模型基础 |
| 03_MULTIVARIATE_DISTRIBUTIONS | 理解联合/边缘/条件分布、协方差与相关系数 | 多变量时序、资产相关性分析 |
| 04_LIMIT_THEOREMS | 掌握大数定律与中心极限定理 | 统计推断、Shiryaev 极限理论 |
| 05_STATISTICS_DISTRIBUTIONS | 掌握样本统计量与三大抽样分布（卡方/t/F） | 假设检验、方差分析 |
| 06_PARAMETER_ESTIMATION | 掌握矩估计、极大似然估计与区间估计 | 时序参数估计、GARCH 基础 |
| 07_HYPOTHESIS_TESTING | 掌握显著性检验、拟合优度检验、非参数检验 | 平稳性检验、模型诊断 |
| 08_ANOVA_REGRESSION | 掌握方差分析与一元线性回归 | 趋势建模、协整前置 |

---

## 当前重点任务

1. **补推导**：各章定理与公式尽量写清证明要点，不只记结论。  
2. **捡遗忘**：按 01 → 08 顺序复习，薄弱章优先加习题与代码复现。  
3. **稳后再上**：第 1–4 章扎实后，再并行推进 `PSTAT_Quant`。

---

## 工具与依赖

- Python 3.x
- `numpy`、`scipy.stats`、`pandas`、`matplotlib`

---

## 学习顺序建议

1. **第一阶段**：01–04 章（概率论基础）  
2. **第二阶段**：05–07 章（数理统计核心）  
3. **第三阶段**：08 章；与 `Econometrics`、`TS` 对照复习  
4. **延伸**：[`PSTAT_Quant`](../PSTAT_Quant/README.md) — Shiryaev 随机金融；卡壳查 [`00_Advanced_Prob_Reference`](../PSTAT_Quant/00_Advanced_Prob_Reference/README.md)（非必学）

---

## 参考资料

- 茆诗松, 程依明, 濮晓龙. 《概率论与数理统计教程》
- 同仓库：[`PSTAT_Quant`](../PSTAT_Quant/README.md)、[`TS`](../TS/README.md)、[`Econometrics`](../Econometrics/README.md)
