# 00_Advanced_Prob_Reference — 概率论经典卷（终极补充包）

**定位**：**非必学主线**，而是仓库的「终极补充包」——平时走 [`PSTAT_Basic`](../../PSTAT_Basic/README.md) → [`Shiryaev_Random_Finance`](../Shiryaev_Random_Finance/README.md) → [`Related_PDE`](../Related_PDE/README.md)；**仅在**啃随机金融、卡壳于纯数学证明时，来此处**按主题查对应章节**补证明。

**教材**：施利亚耶夫（Shiryaev）**《概率论》**多卷体系（测度论概率经典）；与主线 [`Shiryaev_Random_Finance`](../Shiryaev_Random_Finance/README.md) 同源，但本目录按**完整概率卷**组织，供深度查阅，不要求通读。

---

## 为什么单独成包

| 对比 | 主线 | 本补充包 |
| :--- | :--- | :--- |
| 目标 | 对接量化、随机金融 | 补全严格证明与理论细节 |
| 读法 | 按学习阶段顺序推进 | **按需取学**，查哪章读哪章 |
| 节奏 | 不因纯数学停滞 | 不打乱量化学习进度 |
| 传承 | 路径 + 脚手架 | 后人可见「卡住 → 回查经典」的思路 |

> 比把整套概率卷堆在书架上更有用：**知道何时打开、查哪一节**。

---

## 何时打开（查表）

| 主线卡住的位置 | 本包可查方向 |
| :--- | :--- |
| σ-代数、条件期望严格定义 | 测度论概率基础卷 |
| 依收敛、大数定律、CLT 证明 | 极限定理与弱收敛 |
| 鞅、停时、可选停时定理 | 随机过程卷 · 鞅论 |
| 布朗运动构造、二次变差 | 随机过程卷 · 连续时间 |
| Itô 积分、Itô 公式证明 | 随机分析卷 |
| Girsanov、测度变换 | 随机分析 · 绝对连续测度 |
| 最优停时、美式期权概率侧 | 鞅与停时 · Snell 包络等 |

具体章节号随笔记补入时标注（格式：**主线文件 ↔ 概率卷卷/章/节**）。

---

## 笔记目录（待建 · 按卷占位）

```text
00_Advanced_Prob_Reference/
├── 01_Measure_and_Probability/        # 测度、随机变量、条件期望
├── 02_Convergence_Limit_Theorems/     # 收敛性、大数定律、CLT
├── 03_Martingales_and_Stochastic_Proc/ # 鞅、马尔可夫、随机过程
├── 04_Stochastic_Calculus/            # 布朗运动、Itô、Girsanov
└── 05_Optional_Advanced_Topics/       # 其它查证明时零散收录
```

各子目录暂以 `.gitkeep` 占位。有查阅记录时再写：**问题来源（主线哪一节）→ 概率卷定位 → 证明要点 → 是否需回 RFA**。

---

## 使用原则

1. **默认不读**：没有卡壳就不开卷。  
2. **一次只查一个问题**：补完证明即回到主线。  
3. **与 RFA 分工**：测度/积分基础优先 [`RFA`](../../RFA/)；概率卷专注概率论定理链。  
4. **笔记可薄**：此处允许只记「卷·章·页 + 关键一步」，不必重写全书。

---

## 与主线的关系

```text
PSTAT_Basic（茆诗松，必走）
       ↓
Shiryaev_Random_Finance（量化主线，必走）
       ↓                    ↘
Related_PDE（Wilmott，按需）   00_Advanced_Prob_Reference（概率卷，查证明时才开）
```

**建议顺序**：主线优先；本包**并行挂载**，永不插队到 Basic 之前。
