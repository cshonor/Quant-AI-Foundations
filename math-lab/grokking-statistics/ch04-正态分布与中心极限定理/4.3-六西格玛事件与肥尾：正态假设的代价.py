"""4.3 「这是一起六西格玛事件！」——正态假设的代价

书上讲了个真事：2008 年高盛 CFO 把亏损归咎于「25 个标准差」事件。
作者的态度是怀疑：这类说法往往是模型错了，而不是运气差。
这一节把它量化：25σ 在正态下到底是什么量级？真实金融数据的尾部又肥多少？
顺带量一件更要紧的事——用正态算 VaR，误差方向在不同分位上是**相反**的。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "data", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import grokking_data as gd  # noqa: E402
from mathviz import Section, plot  # noqa: E402

SEC = Section(
    number="4.3",
    chapter="第 4 章 · 正态性的宏大展览",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="六西格玛事件与肥尾：正态假设的代价",
    claim="经验法则把 ±3σ 之外当成「几乎不可能」（0.3%）；有人把崩盘归为「六西格玛事件」，"
          "高盛 CFO 甚至说过「25 个标准差」。作者的怀疑：大概率是模型错了，不是运气差。",
    proposition="① 正态下 P(|Z|>6)=1.97e-9、P(|Z|>25)=6.1e-138，「25σ 事件」在数量级上就该被否掉；"
                "② 真实 BTC 日收益超额峰度 4.0，|z|>4 的实测频率是正态预测的 70 倍以上；"
                "③ 用正态算 VaR 的误差方向会翻转：95% 分位上偏保守，99% 分位上偏乐观；"
                "④ 龙卷风数据的最宽纪录在正态模型下概率 1.7e-21，但它就躺在 3210 条记录里。",
    source="第 4 章「经验法则 / 六西格玛事件」（高盛 25 标准差引文）；"
           "数据：BTCUSDT 日线 1998 根对数收益、NOAA 龙卷风宽度 3210 条",
)


# ------------------------------------------------- ① 六西格玛到底有多小


def experiment_six_sigma(sec: Section) -> None:
    p6 = float(2 * stats.norm.sf(6))
    p25 = float(2 * stats.norm.sf(25))
    sec.check(
        "「六西格玛事件」≈ 5 亿分之一；25σ 事件在数量级上根本不可能",
        abs(p6 - 1.973e-9) / 1.973e-9 < 0.01 and p25 < 1e-100,
        f"P(|Z|>6) = {p6:.3e}（约 {1/p6:,.0f} 分之一）；"
        f"P(|Z|>25) = {p25:.2e}——若每个交易日发生一次独立事件，"
        f"要等 {1/p25/365:.1e} 年才「该」见到一次；宇宙年龄只有 1.38e10 年",
        tolerance="P(|Z|>6) 与 1.973e-9 差 < 1%，P(|Z|>25) < 1e-100",
    )

    # 换个角度：一天一个观测，多久能见到一次 6σ / 25σ
    days6 = 1 / p6
    sec.info(
        "换成「多久见一次」",
        f"日频数据下 6σ 事件平均每 {days6:,.0f} 天（{days6/365:,.0f} 年）一次；"
        f"25σ 事件平均每 {1/p25:.1e} 天一次。把 2008 年称作 25σ，"
        f"等价于承认自己的模型离现实差了不止一个数量级。",
    )


# ------------------------------------------------- ② 真实收益的尾部


def experiment_fat_tail(sec: Section) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = gd.btc_daily_returns()
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    exkurt = float(stats.kurtosis(r))
    z = (r - mu) / sd

    ks = np.array([2.0, 3.0, 4.0, 5.0])
    emp = np.array([float(np.mean(np.abs(z) > k)) for k in ks])
    nor = np.array([float(2 * stats.norm.sf(k)) for k in ks])
    ratio = emp / nor

    sec.check(
        "真实日收益显著肥尾：超额峰度 > 3，且 |z|>4 的实测频率是正态预测的 50 倍以上",
        exkurt > 3.0 and ratio[2] > 50,
        f"BTCUSDT 日收益 n={len(r)}：超额峰度 {exkurt:.2f}（正态应为 0）；"
        + "；".join(f"|z|>{k:g}：实测 {e:.3%}（{int(e*len(r))} 次）vs 正态预测 {n:.2e}"
                    f"（{n*len(r):.2f} 次），放大 {rr:.0f}×"
                   for k, e, n, rr in zip(ks, emp, nor, ratio)),
        tolerance="超额峰度 > 3 且 |z|>4 放大 > 50 倍",
    )
    sec.check(
        "放大倍数随 k 迅速增长（越往尾部，正态错得越离谱）",
        bool(np.all(np.diff(ratio) > 0)),
        "→".join(f"|z|>{k:g}: {rr:.0f}×" for k, rr in zip(ks, ratio)),
        tolerance="放大倍数单调递增",
    )
    return r, ks, emp, nor


# ------------------------------------------------- ③ VaR 的误差方向会翻转


def experiment_var(sec: Section, r: np.ndarray) -> None:
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    rows = []
    for q in (0.95, 0.99):
        emp_q = float(np.quantile(r, 1 - q))
        nor_q = float(stats.norm.ppf(1 - q, mu, sd))
        rows.append((q, emp_q, nor_q, (nor_q / emp_q - 1)))
    q95, q99 = rows[0], rows[1]
    sec.check(
        "用正态算 VaR 的误差方向会翻转：95% 分位偏保守，99% 分位偏乐观",
        q95[3] > 0 and q99[3] < 0,
        "；".join(f"{q:.0%} VaR：经验 {e:.2f}% vs 正态 {n:.2f}%"
                  f"（损失幅度上正态{'偏保守' if d > 0 else '偏乐观'} {abs(d):.1%}）"
                  for q, e, n, d in rows),
        tolerance="95% 上正态的损失幅度更大、99% 上更小",
    )
    sec.observe(
        f"这条最反直觉：同一个正态模型，在 95% 分位上把风险算大了 {q95[3]:.1%}"
        f"（经验 {q95[1]:.2f}% vs 正态 {q95[2]:.2f}%），"
        f"到 99% 分位却算小了 {abs(q99[3]):.1%}（经验 {q99[1]:.2f}% vs 正态 {q99[2]:.2f}%）。"
        "原因就是肥尾：正态把中间部分抬高了，代价是尾部被削平。"
        "所以「用正态 VaR 保守一点」这句话只在中等分位成立，极端分位上它是**低估**风险的。"
    )


# ------------------------------------------------- ④ 龙卷风那个最大值


def experiment_tornado_max(sec: Section) -> None:
    w = gd.tornado_width()
    mu, sd = float(w.mean()), float(w.std(ddof=1))
    zmax = float((w.max() - mu) / sd)
    p = float(stats.norm.sf(zmax))
    sec.check(
        "真实数据里的最值，在正态模型下概率小到荒谬 —— 错的是模型不是数据",
        zmax > 6 and p * len(w) < 1e-15,
        f"龙卷风最宽纪录 {w.max():.0f} 码，距均值 {zmax:.2f}σ；"
        f"正态模型下 P(X ≥ max) = {p:.2e}，乘上样本量 {len(w)} 也只有 {p*len(w):.1e} 次——"
        f"但它实实在在发生了 1 次",
        tolerance="z > 6 且期望出现次数 < 1e-15",
    )


def run(sec: Section) -> None:
    experiment_six_sigma(sec)
    r, ks, emp, nor = experiment_fat_tail(sec)
    experiment_var(sec, r)
    experiment_tornado_max(sec)

    # ---- 图 1：Q-Q 图
    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.1))
    stats.probplot(r, dist="norm", plot=axes[0])
    axes[0].get_lines()[0].set(color=plot.ACCENT, ms=2.5)
    axes[0].get_lines()[1].set(color=plot.ACCENT2, lw=1.2)
    axes[0].set_title("BTC 日收益 Q-Q 图：两端脱离直线", fontsize=10)
    axes[0].tick_params(labelsize=8)

    # ---- 图 2：尾部频率（对数轴）
    axes[1].semilogy(ks, emp, "o-", color=plot.ACCENT, lw=1.3, ms=5, label="实测频率")
    axes[1].semilogy(ks, nor, "s--", color=plot.ACCENT2, lw=1.2, ms=4, label="正态模型预测")
    axes[1].set_xlabel("|z| 超过")
    axes[1].set_ylabel("频率（对数轴）")
    axes[1].set_title("越往尾部，正态错得越离谱", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].tick_params(labelsize=8)
    fig.suptitle("正态假设在尾部不是「有点误差」，是差几个数量级", fontsize=11)
    sec.figure("fat-tail", fig, caption="左图看形状，右图看数量级：|z|>5 实测 4 次，正态说 0.001 次")

    sec.observe(
        f"书上对「25 个标准差」的怀疑，实测支持得很干脆：正态下 P(|Z|>25) = "
        f"{float(2*stats.norm.sf(25)):.1e}，就算每分钟观测一次、观测到宇宙尽头也凑不出一次。"
        "所以出现这种量级的偏离时，正确动作是**改模型**（换肥尾分布、加机制），"
        "不是给运气找借口。这条对 CTA 直接可用：当策略回撤打到「多少个 σ」时，"
        "先怀疑波动率模型和样本期，而不是安慰自己「黑天鹅」。"
    )
    sec.pitfall(
        "别用「经验法则 + 3σ」给实盘设熔断阈值。实测 BTC 日收益在 "
        f"{len(r)} 天里就有 {int(np.sum(np.abs((r-r.mean())/r.std(ddof=1)) > 4))} 天超过 4σ"
        "（正态预测不到 0.2 天）。用正态定阈值，等于给风控装了一个会频繁误报、"
        "又在真崩盘时漏报的开关。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
