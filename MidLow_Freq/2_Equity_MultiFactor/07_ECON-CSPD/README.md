# ECON-CSPD

**Wooldridge**《横截面与面板数据的计量经济分析（第二版）》学习笔记。

书写约定：[`书写约定-上下标与公式.md`](../书写约定-上下标与公式.md)。

| 缩写 | 含义 |
| :--- | :--- |
| **ECON** | 计量经济学（Econometrics） |
| **CSPD** | 横截面与面板数据（**C**ross **S**ection **&** **P**anel **D**ata） |

全书共 **22 章**，按原著 **四篇** 分文件夹；每章一个子目录。尚无笔记的章目录内保留 `.gitkeep`。

---

## 《横截面与面板数据的计量经济分析（第二版）》全书目录

### 第 I 篇 引论与背景

1. 引论
2. 计量经济学中条件期望与相关概念
3. 基本渐近理论

### 第 II 篇 线性模型

4. 单方程线性模型与普通最小二乘法估计
5. 单方程线性模型的工具变量估计
6. 附加的单方程专题
7. 利用普通最小二乘法与广义最小二乘法估计方程组
8. 利用工具变量的系统估计
9. 联立方程模型
10. 基本线性不可观测效应面板数据模型
11. 线性不可观测效应模型的更多专题

### 第 III 篇 非线性估计的一般方法

12. M 估计、非线性回归以及分位数回归
13. 极大似然法
14. 广义矩方法与最小距离估计

### 第 IV 篇 非线性模型与相关专题

15. 二值响应模型
16. 多项响应与有序响应模型
17. 角点解响应
18. 计数响应、分数响应及其他非负响应
19. 截取数据、样本选择及损耗
20. 分层抽样与整群抽样
21. 估计平均处理效应
22. 期限分析

---

## 仓库目录结构（四篇 × 22 章）

```text
ECON-CSPD/
├── README.md
├── 01_Part_I_Introduction_Background/
│   ├── 01_Introduction/
│   ├── 02_Conditional_Expectation/
│   └── 03_Asymptotic_Theory/
├── 02_Part_II_Linear_Models/
│   ├── 04_OLS_Single_Equation/ … 11_Panel_More_Topics/
│   └── 08_System_IV/.gitkeep          # 第 8 章占位
├── 03_Part_III_Nonlinear_Estimation_Methods/
│   ├── 12_M_Estimator_Nonlinear_Regression/
│   ├── 13_Maximum_Likelihood/
│   └── 14_GMM_Minimum_Distance/
└── 04_Part_IV_Nonlinear_Models_and_Applications/
    ├── 15_Binary_Response/ … 22_Duration_Analysis/
    ├── 17_Corner_Solution_Response/.gitkeep
    └── 18_Count_Fractional_Nonnegative/.gitkeep
```

### 各章笔记

| 章 | 路径 |
| :---: | :--- |
| 1 | [`01_Introduction`](01_Part_I_Introduction_Background/01_Introduction/Ch01-Econometrics-Causality-DataStructure.md) |
| 2 | [`02`](01_Part_I_Introduction_Background/02_Conditional_Expectation/Ch02-Linear-Model-Matrix-Algebra.md) |
| 3 | [`03`](01_Part_I_Introduction_Background/03_Asymptotic_Theory/Ch03-Large-Sample-Asymptotic-Theory.md) |
| 4 | [`04`](02_Part_II_Linear_Models/04_OLS_Single_Equation/Ch04-Classical-OLS-Assumptions.md) |
| 5 | [`05`](02_Part_II_Linear_Models/05_IV_Single_Equation/Ch05-Endogeneity-IV-2SLS.md) |
| 6 | [`06`](02_Part_II_Linear_Models/06_Single_Equation_Topics/Ch06-Control-Function-Proxies.md) |
| 7 | [`07`](02_Part_II_Linear_Models/07_SUR_GLS_Systems/Ch07-Multivariate-Regression.md) |
| 8 | （占位，待写） |
| 9 | [`09`](02_Part_II_Linear_Models/09_Simultaneous_Equations/Ch09-Simultaneous-Equation-Model.md) |
| 10 | [`10`](02_Part_II_Linear_Models/10_Basic_Panel_Unobserved_Effects/Ch10-Panel-Data-Basics.md)、[FE](02_Part_II_Linear_Models/10_Basic_Panel_Unobserved_Effects/Ch10-Fixed-Effect-First-Difference.md) |
| 11 | [`11`](02_Part_II_Linear_Models/11_Panel_More_Topics/Ch11-Random-Effect-Hausman-Test.md)、[GMM](02_Part_II_Linear_Models/11_Panel_More_Topics/Ch11-Dynamic-Panel-GMM.md) |
| 12 | [`12`](03_Part_III_Nonlinear_Estimation_Methods/12_M_Estimator_Nonlinear_Regression/Ch12-M-Estimator.md) |
| 13 | [`13`](03_Part_III_Nonlinear_Estimation_Methods/13_Maximum_Likelihood/Ch13-Maximum-Likelihood-MLE.md) |
| 14 | [`14`](03_Part_III_Nonlinear_Estimation_Methods/14_GMM_Minimum_Distance/Ch14-Generalized-Method-of-Moments-GMM.md) |
| 15 | [`15`](04_Part_IV_Nonlinear_Models_and_Applications/15_Binary_Response/Ch15-Logit-Probit-Binary-Choice.md) |
| 16 | [`16`](04_Part_IV_Nonlinear_Models_and_Applications/16_Multinomial_and_Ordered_Response/Ch16-Ordered-Discrete-Choice.md) |
| 17–18 | （占位，待写） |
| 19 | [`19`](04_Part_IV_Nonlinear_Models_and_Applications/19_Censoring_Sample_Selection_Attrition/Ch19-Tobit-Censored-Truncated-Data.md)、[Heckman](04_Part_IV_Nonlinear_Models_and_Applications/19_Censoring_Sample_Selection_Attrition/Ch19-Heckman-Sample-Selection.md) |
| 20 | [`20`](04_Part_IV_Nonlinear_Models_and_Applications/20_Stratified_and_Cluster_Sampling/Ch20-Sampling-Clustered-Inference.md) |
| 21 | [`21`](04_Part_IV_Nonlinear_Models_and_Applications/21_Average_Treatment_Effects/Ch21-Treatment-Effect-DID.md)、[RDD](04_Part_IV_Nonlinear_Models_and_Applications/21_Average_Treatment_Effects/Ch21-Regression-Discontinuity-RDD.md) |
| 22 | [`22`](04_Part_IV_Nonlinear_Models_and_Applications/22_Duration_Analysis/Ch22-Duration-Analysis.md) |

说明：此前误用 `Ch23` 表示第 22 章（期限分析），已改为 `Ch22`；RDD 归入第 21 章目录。

---

## 先修与学习顺序

概率与统计 → 回归入门 → 本书第 I–II 篇（OLS、IV、面板）→ 按需第 III–IV 篇。

相关资源：英文原著 *Econometric Analysis of Cross Section and Panel Data*（2nd ed.）；同仓库 `PSTAT_Basic/`、`ML/`。
