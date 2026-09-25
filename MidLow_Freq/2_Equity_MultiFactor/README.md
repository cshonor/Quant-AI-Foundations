# Route 2: Equity Multi-Factor Quant

> 概率统计 + 计量经济学 + 线性模型 + 凸优化 + 时间序列。

## 学习顺序

| 序号 | 文件夹 | 内容 | 重点章节 |
|:---:|:---|:---|:---|
| 1 | `01_PSTAT_Basic` | 茆诗松《概率论与数理统计教程》 | **第 3-6 章**（多维分布、极限定理、估计、假设检验） |
| 2 | `02_Econometrics` | 伍德里奇《计量经济学导论》第 7 版 | **第一部分（Ch 1-9）**：OLS、推断、异方差、模型设定 |
| 3 | `03_Alg` | 线性代数 | 矩阵运算、特征值、PCA |
| 4 | `04_ML` | 机器学习 | **03_Linear_Regression、04_Training_Optimization**（线性/Ridge 回归） |
| 5 | `05_CO` | 凸优化 | Lasso 因子筛选、二次规划组合优化 |
| 6 | `06_TS` | 时间序列分析 | 因子稳定性、衰减、收益时序性质 |
| 7 | `07_ECON-CSPD` | 伍德里奇《横截面与面板数据》第 2 版 | 截面与面板进阶（可选） |

## 学习要点

- **Econometrics 分两段学**：先学第一部分（Ch 1-9）打 OLS 底子，学完 TS 后再回来学第二、三部分（Ch 10-19）。
- **ML 只需回归部分**：`04_ML` 中的 `03_Linear_Regression` 和 `04_Training_Optimization` 是重点，其他章节（分类、SVM、树等）在此路线中非核心。
- **Alg 侧重矩阵计算**：`12_Matrix_Computation` 和 `05_Eigen_Subspace`（PCA）是因子分析的工具基础。
- **CO 侧重 Lasso 和 QP**：Lasso 做因子筛选，QP 做组合优化（Markowitz）。

## 与其他路线的关系

- `01_PSTAT_Basic` 与 Route 1 (CTA) 共享，但本路线需第 3-6 章，CTA 路线需第 1-6 章。
- `03_Alg`、`04_ML`、`05_CO` 与 Route 3 (AI Agent) 共享。
