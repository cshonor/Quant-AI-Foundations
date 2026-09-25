# High_Freq — 高频（HFT 执行层）

> **定位**：高频方向的**执行层**——交易系统的低延迟实现。核心技能：C/C++、Linux 系统编程、内核与网络栈、低延迟优化。

## 与主力学习仓库的关系

高频线的**主体内容不放在本仓库**，托管在姊妹仓库：

- **[cshonor/hft-embedded-linux-study](https://github.com/cshonor/hft-embedded-linux-study)**
  TLPI 全 64 章逐节笔记、LKD3rd 对照笔记、内核/并发/内存、嵌入式 Linux、eBPF（cshonor/ebpf-gate）。

本目录只放与高频交易直接相关的**量化侧增量内容**（如：市场数据协议、订单簿结构、低延迟策略模式），随学习增补。

## 两条线怎么配合

| 线 | 频率 | 关心的东西 | 主战场 |
|:--|:--|:--|:--|
| **高频（本目录 + hft 仓库）** | 高频 | 纳秒~微秒：执行、网络、内核 bypass | C/C++ / Linux / eBPF |
| **[`MidLow_Freq`](../MidLow_Freq/README.md)** | 中低频 | 天~分钟：信号、回测、风控 | 统计 / 时序 / 计量 |

策略研究在 `MidLow_Freq` 做出来后，若需要更高执行频率，才下沉到本目录的高频工程线。
