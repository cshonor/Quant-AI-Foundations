# Related_PDE — 随机金融中的偏微分方程（关联模块）

**定位**：不是系统 PDE 课，而是**随机过程 / 随机金融里出现什么 PDE，就补什么推导与应用**。

**挂靠关系**：从属于 [`Shiryaev_Random_Finance`](../Shiryaev_Random_Finance/README.md)；Shiryaev 给出概率侧（鞅、Itô、风险中性），本模块补 **PDE 侧表述与数值直觉**。

---

## 教材选型

| 角色 | 书名 | 用法 |
| :---: | :--- | :--- |
| **主线** | Paul Wilmott，《**金融工程中的偏微分方程**》 | 为量化从业者写，**跳过纯数学证明**，直接讲 BS 方程、美式期权定价、有限差分等你后续会用到的应用；与 Shiryaev 随机金融**无缝衔接** |
| **工具书** | 《**偏微分方程教程**》（薄册，公式速查） | **不必通读**；随机金融推导里缺某类 PDE 公式/边界条件时，**回头翻对应章节**即可 |

**阅读原则**

1. **Wilmott 跟场景走**：Shiryaev 读到 Itô / 风险中性 / 期权定价时，同步打开 Wilmott 对应节。  
2. **教程书只当字典**：热方程、边值问题、Green 函数等基础式子查不到时再翻。  
3. **不阻塞主线**：先记 PDE 形式与经济含义，严格存在唯一性等纯数学可后补。

---

## 与 Shiryaev 的衔接

| Shiryaev 侧（概率） | Related_PDE 侧（Wilmott） |
| :--- | :--- |
| 布朗运动、Itô 公式 | 热方程、扩散 PDE |
| 风险中性期望定价 | Black–Scholes PDE 推导 |
| Feynman–Kac 型表示 | PDE ⟺ 条件期望 |
| 美式期权（最优停时） | 自由边界 / 线性互补问题 |
| 数值模拟直觉 | 有限差分、显式/隐式格式 |

---

## 收录范围

| 主题 | 典型用途 | 前置 |
| :--- | :--- | :--- |
| 热方程与布朗运动 | 转移密度、扩散过程 | 布朗运动、Itô 公式 |
| Black–Scholes PDE | 欧式期权定价 | 风险中性测度 |
| Feynman–Kac | PDE ⟺ 期望表示 | 随机积分、鞅 |
| 边界/终值条件 | 看涨/看跌、障碍期权 | BS 框架 |
| **美式期权 PDE** | 提前行权、自由边界 | 最优停时（Shiryaev） |
| 有限差分要点 | 与 `ODE`/数值实现衔接 | BS 或热方程离散化 |

**暂不收录**：广义函数论、完整 Sobolev 空间、椭圆/双曲 PDE 系统课内容。

---

## 笔记目录（待建）

```text
Related_PDE/
├── 01_Heat_Equation_and_Brownian/       # 热方程 ↔ 布朗；Wilmott 扩散部分
├── 02_Black_Scholes_Derivation/         # BS PDE：复制组合 / 风险中性
├── 03_Feynman_Kac/                      # 概率表示与 PDE 解
├── 04_Boundary_Conditions_Options/      # 欧式：边界/终值与期权类型
├── 05_American_Options_PDE/             # 美式：自由边界、LCP（Wilmott 重点）
└── 06_Finite_Difference_Numerics/       # 有限差分与实现要点
```

各子目录暂以 `.gitkeep` 占位；笔记格式：**PDE 写出 → 经济含义 → 与 Shiryaev 公式对照 → 数值/实现一句**。

---

## 学习策略

1. **先概率后 PDE**：Shiryaev 阶段 3–4（Itô、Girsanov）完成后再系统读 Wilmott BS 章。  
2. **双线对照**：同一结论写两栏——「鞅期望」与「PDE 边值问题」。  
3. **教程书按需查**：推导卡在某步 PDE 技术（如分离变量、边值分类）时再翻《偏微分方程教程》。  
4. **不阻塞主线**：缺细节先记结论与用法，回头再补。

---

## 建议顺序（相对 Shiryaev）

`03_Brownian_Motion` → `04_Stochastic_Integration` → **01 热方程** → `05_Girsanov` → **02 BS** → **03 Feynman–Kac** → **04 欧式边界** → **05 美式** → **06 差分**
