# 第6章 状态空间模型
> 量化核心：卡尔曼滤波、动态因子管理，机构级高阶工具

## 本章核心目标
理解状态空间模型的框架，掌握卡尔曼滤波的基本原理和应用。

## 关键概念
1. **线性高斯模型**：状态方程 + 观测方程的基本框架
2. **卡尔曼滤波**：预测步 + 更新步，迭代估计隐状态
3. **极大似然估计**：模型参数估计方法
4. **缺失数据修正**：卡尔曼滤波处理时序缺失值
5. **平滑样条和卡尔曼平滑器**：回溯修正状态估计
6. **贝叶斯状态空间模型**：贝叶斯视角下的动态建模

## 量化应用场景
- 动态因子权重调整，实时更新多因子模型
- 因子 IC 序列的卡尔曼平滑，去除噪声
- 补全停牌、数据缺失的序列
- 构建动态跟踪模型，用于配对交易

## 实战练习
```python
# 1. 简单卡尔曼滤波示例
import numpy as np

def kalman_filter(measurements, process_noise, measurement_noise):
    # 初始化
    x = measurements[0]
    p = 1.0
    estimates = []
    for z in measurements:
        # 预测
        x_pred = x
        p_pred = p + process_noise
        # 更新
        K = p_pred / (p_pred + measurement_noise)
        x = x_pred + K * (z - x_pred)
        p = (1 - K) * p_pred
        estimates.append(x)
    return estimates
```
