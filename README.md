# 量化与 AI 学习笔记仓库

个人量化学习笔记，按**交易频率**分为两大部分：高频（执行层）与中低频（策略层）。

- **高频（HFT）**：C/C++ + Linux 系统编程，解决交易系统"跑得快"的问题 → [`High_Freq/`](./High_Freq/README.md)
- **中低频**：统计 + 时序 + 计量，解决"信号从哪来、怎么验证"的问题 → [`MidLow_Freq/`](./MidLow_Freq/README.md)
- **暂缓**：AI Agent 开发线，内容保留、暂停推进 → [`Paused/`](./Paused/README.md)

---

## 仓库初衷

这里整理的不只是量化笔记，而是一套**知识脚手架**：把概率、线性代数、计量、随机金融、PDE 应用等教材之间的缝隙填上，串成**能走通的学习路径**。

与市面零散教材不同的是，本仓库带有**亲身踩坑与验证过的顺序**——例如先茆诗松打概率底、再 Shiryaev 接随机金融、PDE 用 Wilmott 按需补应用而非从零啃纯数学；推导在持续补全，代码与实践在对齐挂载。笔记会随学习更新，后人可以沿此入门，也可以在空白处续写自己的版本。

**几条原则**（写进目录设计里）：

1. **先打牢基础，再定向延伸**——不因贪多而停滞。  
2. **教材是骨架，笔记是血肉**——补推导、易混对比、做题步骤。  
3. **概率侧与 PDE 侧双线对照**——同一结论既写鞅期望，也写边值问题。  
4. **留空可续**——占位目录与待建章节，表示「此处曾卡住、此处待补」。  
5. **按需取学**——经典概率卷、PDE 教程作补充包，查证明时再开，不插队主线。

若你走理工或金融交叉方向，希望这套脚手架比单本教科书更省迷路的时间。

---

## 高频 / 中低频，怎么分工

| | 高频（HFT） | 中低频 |
|:--|:--|:--|
| **目标** | 执行层：低延迟、高吞吐 | 策略层：信号、回测、风控 |
| **技能** | C/C++、Linux 内核、网络、eBPF | 概率统计、时序、计量、优化 |
| **时间尺度** | 纳秒 ~ 微秒 | 天 ~ 分钟 |
| **主阵地** | [`High_Freq/`](./High_Freq/README.md) + [hft-embedded-linux-study](https://github.com/cshonor/hft-embedded-linux-study) | [`MidLow_Freq/`](./MidLow_Freq/README.md) |

---

## MidLow_Freq 目录结构

| 文件夹 | 说明 |
| :--- | :--- |
| [`1_CTA_Futures`](MidLow_Freq/1_CTA_Futures/README.md) | **主攻**：CTA 期货 15 模块 / 5 阶段（含 00_Grokking_Statistics 直觉层伴读） |
| [`2_Equity_MultiFactor`](MidLow_Freq/2_Equity_MultiFactor/README.md) | 股票多因子量化路线 |
| [`PSTAT_Basic`](MidLow_Freq/PSTAT_Basic/README.md) | 底层：茆诗松《概率论与数理统计教程》 |
| [`PSTAT_Quant`](MidLow_Freq/PSTAT_Quant/README.md) | 上层：Shiryaev 随机金融 + 关联 PDE + 概率卷补充包（按需） |
| [`TS`](MidLow_Freq/TS/README.md) | 时间序列分析 |
| [`Econometrics`](MidLow_Freq/Econometrics/README.md) | 伍德里奇《计量经济学导论：现代观点》（第 7 版，19 章） |
| [`ECON-CSPD`](MidLow_Freq/ECON-CSPD/README.md) | 伍德里奇《横截面与面板数据的计量经济分析》（第 2 版，22 章） |
| [`Alg`](MidLow_Freq/Alg/README.md) | 线性代数、数值方法、基础算法 |
| [`CO`](MidLow_Freq/CO/README.md) | 凸优化（Boyd & Vandenberghe 体系提纲） |
| [`ODE`](MidLow_Freq/ODE) | 常微分方程 |
| [`RFA`](MidLow_Freq/RFA) | 实分析与泛函分析 |
| [`ML`](MidLow_Freq/ML) | 机器学习 / 深度学习（Grokking ML 风格章节笔记） |

**计量教材怎么选**：入门用 `Econometrics/`（OLS → 时间序列 → 面板 / IV）；截面与面板进阶用 `ECON-CSPD/`。

**公式排版**：下标、上标见 [`书写约定-上下标与公式.md`](./书写约定-上下标与公式.md)。批量处理可运行：`python tools/format_md_subsup.py`。

---

## 路线一：股票多因子量化（中低频）

**核心**：概率统计 + 计量经济学 + 线性模型 + 凸优化 + 时间序列

| 步骤 | 文件夹 | 重点 |
| :---: | :--- | :--- |
| 1 | `PSTAT_Basic` | 第 3–6 章：多维分布、极限定理、估计、假设检验 |
| 2 | `Econometrics` | 第一部分（第 1–9 章）：OLS、推断、异方差、模型设定 |
| 3 | `Alg` + `ML` | 矩阵运算、PCA；线性 / Ridge 回归（见 `ML/03`、`ML/04`） |
| 4 | `CO` | Lasso 因子筛选、二次规划组合优化 |
| 5 | `TS` | 因子稳定性、衰减、收益时序性质 |
| 6 | `Econometrics` | 第二、三部分（第 10–19 章）；可衔接 `ECON-CSPD` |

**建议顺序**：`00_Grokking_Statistics（直觉层伴读，可穿插）→ PSTAT_Basic → Econometrics（第一部分）→ Alg → ML（回归）→ CO → TS → Econometrics（第二、三部分）→ ECON-CSPD（可选）`

> 直觉层伴读 [`MidLow_Freq/1_CTA_Futures/00_Grokking_Statistics`](MidLow_Freq/1_CTA_Futures/00_Grokking_Statistics/README.md)：Nield《Grokking Statistics》笔记，建立样本/总体、分布与不确定性的直觉后再攻茆诗松。

---

## 路线二：CTA 期货量化（中低频，主攻）

**核心**：概率统计 + 时间序列 + 随机过程 + 风险与回测

详见 [`MidLow_Freq/1_CTA_Futures/README.md`](MidLow_Freq/1_CTA_Futures/README.md)——15 个模块、5 个阶段，含完整的"CTA vs 期权定价学什么"对照表与教材来源对照表。

**最小数学集**：`01_Probability_Foundations → 02_Statistical_Inference → 04/05/06/07 时序四件套 → 12_Risk_Metrics → 13_Backtesting_Methodology`

---

## 暂缓：AI Agent 开发

原路线二（AI Agent：`Alg → CO → ML → RFA` + 大模型应用）**暂缓推进**，内容完整保留在 [`Paused/3_AI_Agent/`](Paused/3_AI_Agent/)，恢复时按 [`Paused/README.md`](Paused/README.md) 的说明处理链接。
