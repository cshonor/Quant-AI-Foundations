# 机器学习与凸优化笔记

主要基于《机器学习图解》整理，并补充凸优化相关内容。ML 为机器学习，CO 为凸优化。

## 缩写说明

- **ML**：机器学习 (Machine Learning)，对应 `ML/` 目录
- **CO**：凸优化 (Convex Optimization)，对应 `CO/` 目录

## ML 与 CO 的关系

**凸优化是机器学习的数学基础之一。** 二者关系可概括为：

| 层面 | 说明 |
|------|------|
| **问题形式** | 很多 ML 任务可写成优化问题：最小化损失函数、最大化似然等。线性回归、逻辑回归、SVM 等都可视为凸优化问题。 |
| **算法工具** | 梯度下降、拟牛顿法、坐标下降等均来自优化理论，是训练 ML 模型的核心算法。 |
| **理论保证** | 凸优化提供全局最优、收敛性等理论；理解 CO 有助于分析 ML 算法的行为与局限。 |
| **超越凸性** | 神经网络等模型多为非凸，但凸优化的思想（下降方向、步长、正则化）仍是设计和理解训练过程的基础。 |

**建议学习顺序**：先掌握 CO 中的梯度下降、凸函数、约束优化等概念，再学 ML 会更容易理解「为什么这样训练」「为什么会收敛」等问题。

## 目录结构

### ML 机器学习（01–13 章）

| 目录 | 内容 |
|------|------|
| `ML/01.What_is_Machine_Learning` | 机器学习基础概念 |
| `ML/02.Types_of_Machine_Learning` | 监督/无监督/强化学习等 |
| `ML/03.Linear_Regression` | 线性回归、损失函数、梯度下降 |
| `ML/04.Training_Optimization_Underfitting_Overfitting_Testing_Regularization` | 欠拟合、过拟合、测试、正则化 |
| `ML/05.Classification_with_Lines_Perceptron` | 感知器 |
| `ML/06.Continuous_Classification_Logistic_Classifier` | 逻辑分类器 |
| `ML/07.Evaluating_Classification_Models` | 准确率等评估指标 |
| `ML/08.Naive_Bayes` | 朴素贝叶斯模型 |
| `ML/09.Decision_Trees` | 决策树 |
| `ML/10.Neural_Networks` | 神经网络 |
| `ML/11.Support_Vector_Machines_and_Kernels` | 支持向量机与核方法 |
| `ML/12.Gradient_Boosting_and_Ensemble_Learning` | 集成方法 |
| `ML/13.Real_World_Data_Engineering_and_ML_Practice` | 数据工程与真实示例 |

### CO 凸优化（CMU 10-725）

以 **CMU 机器学习中的凸优化** 为主，按三阶段组织，面向 ML/量化落地。

| 阶段 | 目录 | 内容 |
|------|------|------|
| 阶段1 | `CO/CMU-阶段1.入门与核心工具/` | 导论、凸性、标准形式、梯度下降、次梯度/近端梯度（讲座1–8） |
| 阶段2 | `CO/CMU-阶段2.进阶理论与复杂算法/` | 对偶、KKT、牛顿法、ADMM 等（讲座9–22） |
| 阶段3 | `CO/CMU-阶段3.拓展与特殊场景/` | 坐标下降、混合整数规划（讲座23–25） |

详见 `CO/README.md`。

## 使用方式

1. **ML**：按章节进入 `ML/` 下对应文件夹（如 `ML/03.Linear_Regression`），阅读 `README.md` 获取概要。
2. **CO**：按阶段进入 `CO/CMU-阶段*/`，各阶段内有二级讲座目录（如 `01.导论`、`05-06.梯度下降法`）。
3. 在各目录中补充知识点、公式与示例，持续迭代。
