# 07 Cointegration & Pairs Trading

> **教材来源**：占位（参考 Hamilton《Time Series Analysis》+ Engle/Granger 论文）


> 协整检验与配对/价差交易。CTA 统计套利的核心数学工具。

## 学习目标

- 理解伪回归（spurious regression）问题——两个不相关的随机趋势可以回归出"显著"的 R²
- 掌握平稳性检验（ADF test）和协整检验（Engle-Granger / Johansen）
- 能构建配对交易策略：找到协整对 → 估计价差 → 设定交易阈值 → 动态对冲

## 核心知识点

### 1. 平稳性回顾
- 严格平稳 vs 弱平稳（stationarity in mean/variance/autocovariance）
- 单位根过程（random walk = non-stationary）
- ADF 检验（Augmented Dickey-Fuller）：H0 = 有单位根（不平稳）

### 2. 协整理论
- **定义**：两个或多个非平稳序列的线性组合是平稳的，则称它们协整
- **经济直觉**：存在长期均衡关系的价格序列，短期偏离会被拉回
- **Engle-Granger 两步法**：
  1. OLS 回归 y = α + βx + ε
  2. 对残差 ε 做 ADF 检验——如果残差平稳，则 y 和 x 协整
- **Johansen 检验**：多变量场景，基于 VECM（向量误差修正模型），能检验多个协整向量

### 3. 配对交易策略
- 选择协整对（同行业/同品种不同月份）
- 估计价差 spread = y - βx
- 计算价差的 z-score：z = (spread - μ) / σ
- 交易规则：z > +阈值 做空价差；z < -阈值 做多价差；z 回归 0 平仓
- 动态更新 β（滚动窗口或 Kalman filter）

### 4. CTA 应用
- **跨期套利**：同一品种不同到期月份合约的协整
- **跨品种套利**：相关品种（如豆油/棕榈油）的协整关系
- **统计套利**：一篮子品种的统计偏离

## 参考材料

- 源仓库 `TS/05_ADVANCED_TOPICS`（如有协整内容）
- 源仓库 `Econometrics` 中非平稳时间序列章节
- 推荐补充：Hamilton《Time Series Analysis》Ch 19
- 推荐补充：Alexander《Market Risk Analysis》Vol II Ch on Cointegration

## 待补充

此目录暂无完整笔记，需补充以下内容：
- [ ] ADF 检验的完整推导
- [ ] Engle-Granger 两步法示例代码（Python）
- [ ] Johansen 检验的 trace 和 max-eigen 统计量
- [ ] 配对交易完整回测模板
