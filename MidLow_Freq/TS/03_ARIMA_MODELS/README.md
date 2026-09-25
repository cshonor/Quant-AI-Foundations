# 第3章 ARIMA模型
> 量化核心：因子IC预测、收益率序列建模，经典时序预测框架

## 本章笔记文件
- [3.1 自回归移动平均模型（ARMA）](./3.1%20自回归移动平均模型（ARMA）.md)
- [3.2 差分方程与ARMA模型](./3.2%20差分方程与ARMA模型.md)
- [3.3 自相关系数与偏自相关系数（ACF与PACF）](./3.3%20自相关系数与偏自相关系数（ACF与PACF）.md)
- [3.4 模型预测（ARMA预测）](./3.4%20模型预测（ARMA预测）.md)
- [3.5 模型估计（参数估计）](./3.5%20模型估计（参数估计）.md)
- [3.6 非平稳数据的差分模型（ARIMA）](./3.6%20非平稳数据的差分模型（ARIMA）.md)
- [3.7 建立ARIMA模型的完整流程](./3.7%20建立ARIMA模型的完整流程.md)
- [3.8 使用自相关误差进行回归（ARMAX与相关）](./3.8%20使用自相关误差进行回归（ARMAX与相关）.md)
- [3.9 乘法季节ARIMA模型（SARIMA）](./3.9%20乘法季节ARIMA模型（SARIMA）.md)

## 本章核心目标
掌握 AR、MA、ARMA、ARIMA 模型的识别、估计、预测流程，解决非平稳序列建模问题。

## 关键概念
1. **AR(p)模型**：自回归模型，用自身滞后项预测
2. **MA(q)模型**：移动平均模型，用误差项滞后项预测
3. **ARMA(p,q)模型**：AR与MA的组合
4. **差分与 ARIMA(p,d,q)**：d阶差分将非平稳序列转为平稳序列
5. **模型识别**：通过 ACF/PACF 图判断 p 和 q
6. **模型选择**：AIC/BIC 准则，避免过拟合
7. **季节性 ARIMA（SARIMA）**：处理带季节性的序列

## 量化应用场景
- 预测因子 IC 的未来走势，提前判断因子是否会失效
- 对股价序列做 ARIMA 预测，辅助中低频策略
- 捕捉市场的季节性效应（月度/季度效应）

## 实战练习
```python
# 1. 用ARIMA模型拟合并预测IC序列
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# 假设ic_series是因子IC的时间序列
model = ARIMA(ic_series, order=(1, 0, 1))  # ARIMA(1,0,1)
result = model.fit()
print(result.summary())

# 预测未来10期IC值
forecast = result.get_forecast(steps=10)
print(forecast.predicted_mean)
```
