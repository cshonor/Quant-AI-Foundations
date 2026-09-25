# 第5章 其他的时域主题
> 量化核心：单位根检验、GARCH波动率建模，解决风控与非平稳问题

## 本章核心目标
掌握单位根检验、长记忆模型、GARCH模型等进阶时域分析工具。

## 关键概念
1. **长记忆 ARMA 模型与分数阶差分**：处理自相关缓慢衰减的序列
2. **单位根检验（ADF）**：判断序列是否平稳的核心方法
3. **GARCH 模型**：条件异方差模型，捕捉波动率聚集效应
4. **阈值模型**：非线性模型，处理序列在不同状态下的不同行为
5. **滞后回归与传递函数建模**：分析变量间的滞后影响

## 量化应用场景
- 用 ADF 检验判断股价、IC 序列是否平稳
- 用 GARCH 模型预测波动率，用于风控和期权定价
- 分析宏观数据对股市收益的滞后影响

## 实战练习
```python
# 1. ADF单位根检验
from statsmodels.tsa.stattools import adfuller

result = adfuller(ret.dropna())
print(f'ADF统计量: {result[0]}, p值: {result[1]}')
# p值<0.05则拒绝原假设，序列平稳

# 2. GARCH模型拟合
from arch import arch_model
model = arch_model(ret.dropna(), vol='Garch', p=1, q=1)
res = model.fit()
print(res.summary())
```
