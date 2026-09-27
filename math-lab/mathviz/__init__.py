"""mathviz —— 数学书逐小节实验的最小工具层。

刻意只做三件事，别长成一个框架：
1. `Section`    统一"小节"的结构：书上结论 / 待验证命题 / 检查项 / 图 / 观察。
2. `num`        数值实验常用的几个判据（相对误差、收敛阶、数值秩）。
3. `site`       把跑完的小节汇总成一个可点开的 HTML 索引。

设计原则：**图是副产品，断言才是正主**。
每个小节必须至少有一条 `check(...)`，跑不出来 PASS 就等于这一节没学明白。
"""
from .section import Section, Check
from . import num, plot, site

__all__ = ["Section", "Check", "num", "plot", "site"]
__version__ = "0.1.0"
