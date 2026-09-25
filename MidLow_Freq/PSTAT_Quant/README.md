# PSTAT_Quant — 量化定向延伸

**定位**：在 [`PSTAT_Basic`](../PSTAT_Basic/README.md)（茆诗松）打牢概率统计基础之后，用施利亚耶夫（Shiryaev）体系对接**随机过程与随机金融**；偏微分方程（PDE）不单独开全课程，只作为随机金融里的**关联模块**按需补推导与应用。

---

## 精力分配原则

| 优先级 | 轨道 | 目标 |
| :---: | :--- | :--- |
| **1** | `PSTAT_Basic` | 补完推导、捡回遗忘——全书 8 章稳步推进 |
| **2** | `Shiryaev_Random_Finance` | 鞅、布朗运动、风险中性定价等量化核心 |
| **3** | `Related_PDE` | Wilmott《金融工程中的偏微分方程》+ PDE 教程速查；BS、美式期权等按需 |
| **—** | `00_Advanced_Prob_Reference` | **非主线**：Shiryaev《概率论》经典卷，卡壳查证明时用 |

> 不贪多：PDE 与概率卷均「按需取学」，避免主线停滞。

---

## 目录结构

```text
PSTAT_Quant/
├── 00_Advanced_Prob_Reference/  # 终极补充包：概率论经典卷（查证明，非必学）
├── Shiryaev_Random_Finance/     # 施利亚耶夫：随机过程 → 随机金融
├── Related_PDE/                 # 关联模块：随机金融中的 PDE 推导与应用
└── README.md                    # 本说明
```

---

## 前置要求

| 模块 | 建议已掌握 |
| :--- | :--- |
| `PSTAT_Basic` 第 1–4 章 | 概率公理、随机变量、多维分布、极限定理 |
| `PSTAT_Basic` 第 5–7 章 | 抽样分布、估计、假设检验（计量与模型诊断） |
| `RFA`（可选但推荐） | 测度、积分——读 Shiryaev 高阶内容时必备 |
| `ODE`（可选） | 常微分方程直觉——读 BS PDE 时有用 |

---

## 与仓库其他模块的衔接

| 后续 | 关联 |
| :--- | :--- |
| 路线三（期权期货） | 本目录 → `CO`（校准、优化）→ 定价实践 |
| `TS` | 平稳性、谱与随机过程的时间序列视角 |
| `Econometrics` / `ECON-CSPD` | 统计推断与面板/截面方法并行 |

**建议顺序**：`PSTAT_Basic（1–4 章稳）→ PSTAT_Basic（5–8 章）→ Shiryaev_Random_Finance → Related_PDE（Wilmott，与 Shiryaev 阶段 3–5 并行）`

---

## 书写约定

公式上下标优先 HTML `<sub>` / `<sup>`，见仓库根目录 [`书写约定-上下标与公式.md`](../../书写约定-上下标与公式.md)。
