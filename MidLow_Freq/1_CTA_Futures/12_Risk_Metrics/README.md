# 12 Risk Metrics

> **教材来源**：新建占位（待补充教材）


> 风险度量。CTA 策略的生存线——不管理风险，再好的信号也会爆仓。

## 学习目标

- 掌握主流风险度量指标的计算和适用场景
- 理解每种指标的局限性和陷阱
- 能构建完整的风险监控仪表盘

## 核心知识点

### 1. 波动率类
- **年化波动率**：σ_annual = σ_daily × √252（注意：CTA 交易天数可能不是 252）
- **已实现波动率（Realized Volatility）**：高频收益的平方和
- **隐含波动率（Implied Volatility）**：期权市场反推的波动率
- **波动率锥（Volatility Cone）**：不同窗口波动率的分位数分布

### 2. 下行风险
- **最大回撤（Max Drawdown）**：peak 到 trough 的最大跌幅
  - 计算方法：维护 running max，逐 bar 计算 (price - running_max) / running_max
  - CTA 策略回撤 > 20% 通常需要停止交易审查
- **Calmar Ratio**：年化收益 / 最大回撤
- **MAR Ratio**：类似 Calmar，但用滚动 36 个月

### 3. 风险价值（VaR）
- **定义**：在置信水平 α 下，投资组合在未来持有期 Δt 内的最大损失
- **历史模拟法**：用历史收益分布的分位数
- **参数法（方差-协方差法）**：假设正态分布，VaR = μ + z_α × σ
- **蒙特卡洛法**：模拟未来收益路径
- **陷阱**：VaR 不满足次可加性（不是一致风险度量），且对尾部风险不敏感

### 4. 期望短缺（Expected Shortfall / CVaR）
- **定义**：超过 VaR 的损失的期望值
- **ES_α = E[Loss | Loss > VaR_α]**
- 优点：满足次可加性，是一致风险度量（coherent risk measure）
- Basel III 已用 ES 替代 VaR

### 5. 绩效比率
| 指标 | 公式 | 含义 |
|:---|:---|:---|
| Sharpe Ratio | (R - R_f) / σ | 单位总风险的超额收益 |
| Sortino Ratio | (R - R_f) / σ_downside | 单位下行风险的超额收益（只惩罚亏损） |
| Calmar Ratio | R_annual / MaxDD | 单位回撤风险的收益 |
| Information Ratio | (R_p - R_b) / TE | 相对基准的超额收益 / 跟踪误差 |
| Omega Ratio | E[max(0, r-threshold)] / E[max(0, threshold-r)] | 收益/损失的概率加权比 |

### 6. 压力测试
- 历史情景回放（2008 金融危机、2020 疫情、2022 加息）
- 蒙特卡洛压力情景生成
- 敏感性分析（Delta/Gamma/Vega 对标的价格变动的敏感度）

## CTA 特有的风险考量

- **杠杆风险**：期货保证金交易，杠杆可达 10-50 倍
- **展期风险（Roll Risk）**：主力合约切换时的价差损失
- **流动性风险**：远月合约流动性差，大单冲击成本高
- **隔夜跳空**：CTA 持仓过夜可能面临大幅跳空
- **相关性崩溃**：危机时品种间相关性趋向 1，分散化失效
- **模型风险**：策略回测过拟合，实盘表现远不如回测

## 参考材料

此目录暂无源笔记。推荐补充：
- Hull《Risk Management and Financial Institutions》
- Alexander《Market Risk Analysis》Vol I-IV
- Bailey & López de Prado 论文系列

## 待补充

- [ ] VaR 三种计算方法的 Python 实现
- [ ] ES 的计算和回测
- [ ] CTA 风险监控仪表盘模板
- [ ] 压力测试历史情景数据
