# Shiryaev — 随机过程与随机金融

**教材主线**：A. N. Shiryaev，《Probability》及随机金融相关卷章（量化向笔记）；**完整证明**回查 [`00_Advanced_Prob_Reference`](../00_Advanced_Prob_Reference/README.md)（《概率论》经典卷，非必学）。

**目标**：从滤波、鞅、布朗运动出发，建立**风险中性定价**与**随机积分**直觉，对接期权/期货量化（仓库路线三）。

---

## 学习阶段（规划）

| 阶段 | 主题 | 与 `PSTAT_Basic` 的衔接 |
| :---: | :--- | :--- |
| 0 | 测度论概率回顾（σ-代数、条件期望） | 第 1–3 章直观 → `RFA` 严格化 |
| 1 | 滤波、停时、鞅 | 第 4 章极限定理 → 收敛与期望交换 |
| 2 | 布朗运动、二次变差 | 连续时间随机过程入门 |
| 3 | 随机积分（Itô）、Itô 公式 | 定价模型的核心运算 |
| 4 | Girsanov 定理、风险中性测度 | 无套利定价框架 |
| 5 | 期权定价、波动率、数值模拟 | 对接 [`Related_PDE`](../Related_PDE/README.md)（Wilmott：BS、美式期权 PDE）、`CO` |

---

## 笔记目录（待建）

```text
Shiryaev_Random_Finance/
├── 01_Measure_Prob_Review/          # 测度论概率要点（桥接 RFA）
├── 02_Martingales_Filtrations/      # 鞅与滤波
├── 03_Brownian_Motion/              # 布朗运动
├── 04_Stochastic_Integration/       # 随机积分与 Itô 公式
├── 05_Girsanov_Risk_Neutral/        # 测度变换与风险中性
└── 06_Option_Pricing_Applications/  # 期权定价与应用
```

各子目录暂以 `.gitkeep` 占位；有推导笔记时再按「一节一文件」补入。

---

## 当前任务

1. 先确保 `PSTAT_Basic` 第 4 章（极限定理、特征函数）推导完整。  
2. 并行阅读 `RFA` 测度论章节，为 Shiryaev 第 0 阶段做准备。  
3. 阶段 5 起并行 [`Related_PDE`](../Related_PDE/README.md)：Wilmott 补 PDE 表述，Shiryaev 补概率推导。  
4. 本目录笔记优先写：**定义 → 定理 → 证明要点 → 量化场景**；严格证明缺步时查 [`00_Advanced_Prob_Reference`](../00_Advanced_Prob_Reference/README.md)，不必中断主线通读概率卷。
