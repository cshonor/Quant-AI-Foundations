# MidLow_Freq — 中低频（策略层）

> **定位**：中低频量化策略研究的全部内容——信号构建、统计推断、时间序列建模、回测与风险管理。

## 两条策略主线

| 路线 | 目录 | 说明 |
|:--|:--|:--|
| **CTA 期货（主攻）** | [`1_CTA_Futures`](./1_CTA_Futures/README.md) | 15 个模块 / 5 阶段：概率 → 统计推断 → 时序（ARMA/GARCH/协整/卡尔曼）→ 随机过程 → 风险与回测 |
| **股票多因子** | [`2_Equity_MultiFactor`](./2_Equity_MultiFactor/README.md) | 多因子量化所需的基础模块镜像与路线 |

## 数学基础模块（两条主线共享）

| 模块 | 内容 | 主要教材 |
|:--|:--|:--|
| [`PSTAT_Basic`](./PSTAT_Basic/README.md) | 概率统计底层基础轨 | 茆诗松《概率论与数理统计教程》 |
| [`PSTAT_Quant`](./PSTAT_Quant/README.md) | 随机金融上层 + 关联 PDE + 概率卷补充包 | Shiryaev《Probability》 |
| [`TS`](./TS/README.md) | 时间序列分析 | Shumway & Stoffer |
| [`Econometrics`](./Econometrics/README.md) | 计量经济学入门 | 伍德里奇《导论》（第 7 版） |
| [`ECON-CSPD`](./ECON-CSPD/README.md) | 横截面与面板进阶 | 伍德里奇《CSPD》（第 2 版） |
| [`Alg`](./Alg/README.md) | 线性代数、数值方法 | Axler 等 |
| [`CO`](./CO/README.md) | 凸优化 | Boyd & Vandenberghe |
| [`ODE`](./ODE/README.md) | 常微分方程（SDE/PDE 前置） | V. I. Arnold |
| [`RFA`](./RFA/README.md) | 实分析与泛函分析 | 周民强等 |
| [`ML`](./ML) | 机器学习 / 深度学习笔记 | Grokking ML 风格 |

**读法**：不要单独啃基础模块——从 `1_CTA_Futures` 或 `2_Equity_MultiFactor` 的路线 README 进入，按需挂载对应模块。

## 公式排版

下标、上标约定见根目录 [`书写约定-上下标与公式.md`](../书写约定-上下标与公式.md)；批量处理：`python tools/format_md_subsup.py`。
