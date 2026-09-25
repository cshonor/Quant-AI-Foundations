# Route 1: CTA Futures

> 主攻方向。CTA 策略为核心。

## 当前问题诊断

原路线直接照搬源仓库的"期权与期货量化（随机金融）"路径，存在两个根本问题：

### 1. 方向偏差——为期权定价设计的路线，不是为 CTA 设计的

原路线的内核是：测度论 → 概率 → 随机微积分 → PDE → 期权定价。这是**期权定价工程师**的路线。

CTA 策略的核心技能是：时间序列建模 → 信号构建 → 回测验证 → 风险管理。两者需要的数学重叠不到 30%。

### 2. 颗粒度粗糙——每本教材整本塞进来，没有拆分

| 原文件夹 | 问题 |
|:---|:---|
| `01_RFA` | 整本实分析+泛函分析，CTA 只需要测度论前 3-4 章 |
| `02_PSTAT_Basic` | 8 章全收录，但 CTA 只需 Ch 1-4 概率 + Ch 5-6 统计推断 |
| `03_PSTAT_Quant` | Shiryaev 整本随机金融，CTA 只需要随机过程基础部分 |
| `04_ODE` | 整本 ODE，CTA 只需要为 SDE/PDE 数值方法服务的部分 |
| `05_CO` | 整本凸优化，CTA 只需要 QP + 风险平价 |

### 3. 严重遗漏

| 遗漏科目 | 对 CTA 的重要性 |
|:---|:---|
| **时间序列分析** | ⛔ 最核心——CTA 本质就是时间序列策略 |
| **回测方法学** | ⛔ 没有回测意识，CTA 必死在过拟合上 |
| **风险度量** | ⛔ VaR/ES/最大回撤是 CTA 的生命线 |
| **线性代数** | 🟡 PCA/因子模型/回归的矩阵形式 |
| **机器学习** | 🟡 现代 CTA 越来越多用 ML 做信号 |

---

## 细粒度方案（15 个模块，4 个阶段）

### Phase 1: 数学基础

| 序号 | 模块 | 内容 | 来源 |
|:---:|:---|:---|:---|
| 00 | `00_Grokking_Statistics` | 直觉层伴读：样本 vs 总体、描述统计、六大分布、蒙特卡洛模拟 | Nield《Grokking Statistics》 |
| 01 | `01_Probability_Foundations` | 概率空间、随机变量、分布、期望、矩、极限定理 | PSTAT_Basic Ch 1-4 |
| 02 | `02_Statistical_Inference` | 点估计、区间估计、假设检验、MLE | PSTAT_Basic Ch 5-6 |
| 03 | `03_Linear_Algebra` | 矩阵运算、特征值分解、PCA、回归的矩阵形式 | Alg（选读 05/07/12 章） |

### Phase 2: 时间序列（CTA 核心）

| 序号 | 模块 | 内容 | 来源 |
|:---:|:---|:---|:---|
| 04 | `04_Time_Series_Basics` | 平稳性、ACF/PACF、白噪声、Ljung-Box 检验 | TS |
| 05 | `05_ARMA_ARIMA` | Box-Jenkins 方法论、定阶、诊断 | TS |
| 06 | `06_GARCH_Volatility` | 波动率聚类、ARCH/GARCH、EWMA、已实现波动率 | TS |
| 07 | `07_Cointegration_Pairs` | Engle-Granger、Johansen、协整检验、配对/价差交易 | TS + Econometrics（选读） |
| 08 | `08_Kalman_Filter` | 状态空间模型、卡尔曼滤波、动态线性模型 | TS（如有）/ 新建 |

### Phase 3: 随机过程与随机微积分

| 序号 | 模块 | 内容 | 来源 |
|:---:|:---|:---|:---|
| 09 | `09_Stochastic_Processes` | 布朗运动、鞅、马尔可夫过程、停时 | PSTAT_Quant（Shiryaev 选读） |
| 10 | `10_SDE_Ito_Calculus` | 伊藤引理、SDE、Girsanov 定理 | PSTAT_Quant + ODE（选读） |

### Phase 4: 优化、风险与回测

| 序号 | 模块 | 内容 | 来源 |
|:---:|:---|:---|:---|
| 11 | `11_Convex_Optimization` | QP、组合优化、风险平价、Lasso 信号筛选 | CO（选读） |
| 12 | `12_Risk_Metrics` | VaR、Expected Shortfall、最大回撤、Sharpe/Sortino/Calmar | 新建（需补充材料） |
| 13 | `13_Backtesting_Methodology` | 过拟合、样本外检验、多重检验、Deflated Sharpe Ratio | 新建（需补充材料） |

### Phase 5: 进阶（可选）

| 序号 | 模块 | 内容 | 来源 |
|:---:|:---|:---|:---|
| 14 | `14_Measure_Theory` | σ-algebra、勒贝格积分、Radon-Nikodym | RFA（选读前 3-4 章） |
| 15 | `15_PDE_Numerical` | 有限差分、蒙特卡洛模拟（期权定价数值方法） | PSTAT_Quant/Related_PDE |

---

## CTA vs 期权定价：学什么、不学什么

| 数学领域 | CTA 趋势跟踪 | 期权定价 | 差异 |
|:---|:---|:---|:---|
| 时间序列 | ⛔ **绝对核心** | 🟡 辅助 | CTA 的命根子 |
| 统计推断 | 🔴 重要 | 🟡 辅助 | 信号显著性检验 |
| 随机微积分 | 🟡 理解价格动态 | ⛔ **绝对核心** | 期权定价的根基 |
| PDE 数值方法 | ⚪ 可选 | ⛔ **绝对核心** | 期权定价求解器 |
| 测度论 | ⚪ 可选 | 🔴 重要 | 严格概率基础 |
| 凸优化 | 🔴 重要 | 🔴 重要 | 共享 |
| 风险度量 | ⛔ **绝对核心** | 🟡 辅助 | CTA 生存线 |
| 回测方法学 | ⛔ **绝对核心** | ⚪ 不太需要 | CTA 防过拟合 |

---

## 实施建议

0. **先扫 00 建直觉**：`00_Grokking_Statistics` 是直觉层伴读，不必逐章精读——每进入一个正式模块前扫对应章节即可（如进 01 前读它的 Ch 2-3，进 02 前读 Ch 4+ 假设检验部分）
1. **Phase 1-2 优先**：概率 + 统计 + 时间序列，这是 CTA 能上手的最小数学集
2. **Phase 3 按需**：如果只做趋势跟踪不做期权，Phase 3 可以跳过或略读
3. **Phase 4 必学**：风险和回测是 CTA 的"保命"知识，不能省
4. **Phase 5 可选**：测度论和 PDE 是给想做期权定价的人准备的

---

## 教材来源对照表

| 模块 | 教材 | 作者/出处 | 章节 |
|:---|:---|:---|:---|
| 00_Grokking_Statistics | *Grokking Statistics*（深入理解统计学） | Thomas Nield（Manning） | 全书 10 章（伴读，按需） |
| 01_Probability_Foundations | 《概率论与数理统计教程》 | 茆诗松 等（高等教育出版社） | Ch 1-4 |
| 02_Statistical_Inference | 《概率论与数理统计教程》 | 茆诗松 等（高等教育出版社） | Ch 5-8 |
| 03_Linear_Algebra | 《线性代数应该这样学》 | Sheldon Axler | 全本（重点 Ch 5/7/12） |
| 04_Time_Series_Basics | 《时间序列分析及其应用（第4版）》 | Robert H. Shumway & David S. Stoffer | Ch 1-2, 4 |
| 05_ARMA_ARIMA | 《时间序列分析及其应用（第4版）》 | Robert H. Shumway & David S. Stoffer | Ch 3 |
| 06_GARCH_Volatility | 《时间序列分析及其应用（第4版）》 | Robert H. Shumway & David S. Stoffer | Ch 5 |
| 07_Cointegration_Pairs | — | 占位（参考 Hamilton《Time Series Analysis》+ Engle/Granger 论文） | — |
| 08_Kalman_Filter | 《时间序列分析及其应用（第4版）》 | Robert H. Shumway & David S. Stoffer | Ch 6-7 |
| 09_Stochastic_Processes | *Probability* | A. N. Shiryaev（Springer） | Ch 1-3 |
| 10_SDE_Ito_Calculus | *Probability* + 《常微分方程》 | Shiryaev（Springer）+ V. I. Arnold | Shiryaev Ch 4-5 + Arnold 全本 |
| 11_Convex_Optimization | *Convex Optimization* | Stephen Boyd & Lieven Vandenberghe（Cambridge） | 全本（重点 Part I + App） |
| 12_Risk_Metrics | — | 新建占位 | — |
| 13_Backtesting_Methodology | — | 新建占位 | — |
| 14_Measure_Theory | 实变函数与泛函分析 | 内部整理笔记 | 实变函数 Ch 1-5 |
| 15_PDE_Numerical | *Probability* Ch 6 + Related_PDE | Shiryaev（Springer）+ 内部整理 | — |

---

## 与其他路线的关系

- `01_Probability_Foundations` + `02_Statistical_Inference` 与 Equity_MultiFactor 共享（来源相同，PSTAT_Basic）
- `03_Linear_Algebra` 与 Equity_MultiFactor / AI_Agent 共享
- `11_Convex_Optimization` 三条路线共享
- `14_Measure_Theory` 与 AI_Agent 的 `04_RFA` 来源相同但侧重不同
