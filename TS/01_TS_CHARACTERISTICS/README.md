# 第1章 时间序列的特征
> 量化核心：因子IC、收益率序列的“体检报告”，判断序列是否适合建模

## 本章核心目标
理解时间序列的基本性质，掌握平稳性判断、自相关分析方法，为后续所有时序模型打基础。

## 关键概念
1. **时间序列的性质**：趋势、季节性、周期性、噪声、异方差
2. **平稳性**：严平稳 vs 宽平稳，均值/方差/自协方差的稳定性条件
3. **白噪声与随机游走**：
   - 白噪声：无自相关、均值为0的序列（因子失效的典型特征）
   - 随机游走：非平稳序列的代表，需差分处理
4. **自相关函数（ACF）**：序列与自身滞后项的相关性，衡量序列的“记忆性”
5. **偏自相关函数（PACF）**：剔除中间滞后影响后的相关性，用于识别 AR 模型阶数
6. **多维时间序列**：交叉相关、协方差矩阵，多因子分析基础

## 量化应用场景
- 分析因子 IC 序列的自相关性，判断因子有效性的衰减速度
- 检验股票收益率序列是否平稳，决定后续建模方法
- 多因子之间的交叉相关性分析，剔除冗余因子

## 实战练习
```python
# 1. 收益率序列可视化与ACF/PACF分析
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from jqdata import get_price

# 获取数据
df = get_price('000001.XSHE', start_date='2020-01-01', end_date='2025-01-01',
               frequency='daily', fields=['close'])
df['ret'] = df['close'].pct_change().dropna()

# 画ACF/PACF图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df['ret'].dropna(), lags=20, ax=ax1)
plot_pacf(df['ret'].dropna(), lags=20, ax=ax2)
plt.show()
```
