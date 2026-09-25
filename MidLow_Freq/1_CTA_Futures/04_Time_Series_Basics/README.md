# 《时间序列分析及其应用（第4版）》量化向学习笔记

> **教材来源**：Robert H. Shumway & David S. Stoffer《时间序列分析及其应用（第4版）》Ch 1-2, 4


公式中的上下标与指数：见仓库根目录 [书写约定-上下标与公式.md](../书写约定-上下标与公式.md)（推荐 HTML `<sub>` / `<sup>`，便于预览区分层级）。

> 面向中低频量化、因子分析与策略开发的结构化学习笔记
> 目标：用时间序列工具解决因子 IC 分析、收益预测、风险控制问题

## 目录结构
```text
TS/
├── 01_TS_CHARACTERISTICS/
├── 02_REGRESSION_EDA/
├── 03_ARIMA_MODELS/
├── 04_SPECTRAL_ANALYSIS/
├── 05_ADVANCED_TOPICS/
├── 06_STATE_SPACE_MODELS/
└── 07_SPECTRAL_METHODS/
```

## 学习目标
1. 掌握平稳性、自相关、ACF/PACF 等基础时序分析工具
2. 学会用 ARIMA/GARCH 模型分析因子 IC 和波动率序列
3. 理解卡尔曼滤波、状态空间模型在动态因子管理中的应用
4. 掌握频谱分析和滤波方法，优化因子信号质量

## 前置知识
- 高等代数基础（向量、矩阵、协方差）
- 概率论与数理统计（相关性、假设检验、最大似然估计）
- Python 基础（pandas、numpy、matplotlib、statsmodels）
