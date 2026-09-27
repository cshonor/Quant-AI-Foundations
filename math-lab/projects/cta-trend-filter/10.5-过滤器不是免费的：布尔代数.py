"""10.5 过滤器不是免费的：布尔代数与「被吞掉的平仓信号」

原代码的写法：

    entries_long  = raw_long  & (adx >= 20)
    entries_short = raw_short & (adx >= 20)
    exits         = entries_short        # ← 这里用的也是「过滤后」的信号
    short_exits   = entries_long

数学上这是集合运算：开仓用 A∩B，平仓也用 A∩B。于是 ADX < 20 那一天，
死叉既不反手、也不平仓——作者的注释写「过滤只拦截新开仓」，代码做的是反的。
本节不去争论意图，只把三种写法的信号集合和持仓时长数出来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "data", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cta_data  # noqa: E402
import _indicators as ind  # noqa: E402
from mathviz import Section, num, plot  # noqa: E402

SEC = Section(
    number="10.5",
    chapter="CTA 专题 · 信号逻辑",
    book="贯穿案例：CTA 趋势策略（BTCUSDT 日线）",
    title="过滤器不是免费的：布尔代数与被吞掉的平仓信号",
    claim="作者注释：「ADX < 20 时忽略信号，不开新仓；已持有的老仓位依旧会触发止损/反向平仓。」",
    proposition="① 容斥恒等式：原始交叉数 = 通过过滤 + 被过滤掉（集合划分必须精确成立）；"
                "② 把「过滤后的信号」同时当 exit 用，会吞掉持仓中的反手信号，使最长持仓异常拉长；"
                "③ 改成「开仓用过滤后信号、平仓用原始信号」后，被吞信号数归零，持仓时长回到基线。",
    source="布尔代数 · 集合划分 / 该结论来自对本仓库 audit_run.py 的实测回放",
)

FAST, SLOW, N = 20, 60, 14
ADX_THRESHOLD = 20.0
N_YEARS = 3.0
FEE = 0.001


def build_signals(high, low, close):
    fast, slow = ind.sma(close, FAST), ind.sma(close, SLOW)
    raw_long = ind.crossed_above(fast, slow)
    raw_short = ind.crossed_below(fast, slow)
    adx, _, _ = ind.wilder_adx(high, low, close, N)
    ok = np.where(np.isnan(adx), False, adx >= ADX_THRESHOLD)
    return raw_long, raw_short, adx, ok


# ------------------------------------------------------------------ ① 集合划分


def experiment_inclusion_exclusion(sec: Section, raw_long, raw_short, ok) -> None:
    n_raw = int(np.sum(raw_long) + np.sum(raw_short))
    n_kept = int(np.sum(raw_long & ok) + np.sum(raw_short & ok))
    n_dropped = int(np.sum(raw_long & ~ok) + np.sum(raw_short & ~ok))

    sec.check(
        "容斥恒等式：|原始信号| = |通过过滤| + |被过滤掉|",
        n_raw == n_kept + n_dropped,
        f"原始交叉 {n_raw} 次 = 通过 {n_kept} + 被拦 {n_dropped}"
        f"（拦截率 {n_dropped / n_raw * 100:.0f}%）",
        tolerance="整数精确相等",
    )
    sec.observe(
        f"过滤掉了 {n_dropped}/{n_raw} = {n_dropped/n_raw*100:.0f}% 的交叉信号。"
        "「交易次数变少」是这一步的直接后果，但真正的问题在第二步："
        "被拦掉的信号里，有一部分其实是**平仓/反手**信号。"
    )


# ------------------------------------------------------------------ ②③ 状态机回放三连


def experiment_replay(sec: Section, df, raw_long, raw_short, ok, adx) -> None:
    o, high, low, close = cta_data.ohlc(df)
    keep_long, keep_short = raw_long & ok, raw_short & ok

    # (a) 完全不过滤的基线
    base_trades, _ = ind.replay(o, raw_long, raw_short, raw_short, raw_long,
                                raw_long, raw_short, FEE)
    # (b) 原样：开仓和平仓都用过滤后信号
    orig_trades, eaten_orig = ind.replay(o, keep_long, keep_short, keep_short, keep_long,
                                         raw_long, raw_short, FEE)
    # (c) 修好：开仓用过滤后信号，平仓用原始信号
    fix_trades, eaten_fix = ind.replay(o, keep_long, raw_short, keep_short, raw_long,
                                       raw_long, raw_short, FEE)

    def stat(trades):
        if not trades:
            return 0, 0.0, float("nan")
        bars = [t["bars"] for t in trades]
        eq = float(np.prod([1.0 + t["ret"] for t in trades]))
        return len(trades), float(np.max(bars)), eq

    n_b, max_b, eq_b = stat(base_trades)
    n_o, max_o, eq_o = stat(orig_trades)
    n_f, max_f, eq_f = stat(fix_trades)

    sec.check(
        "原样写法会吞掉持仓中的反手信号（实测 > 0 次）",
        eaten_orig > 0,
        f"被吞掉的反手信号 {eaten_orig} 次；最长持仓从基线的 {max_b:.0f} 根拉长到 {max_o:.0f} 根"
        f"（+{max_o - max_b:.0f} 根）",
        tolerance="被吞次数 > 0",
    )
    sec.check(
        "改为「平仓用原始信号」后被吞次数归零，最长持仓回到基线",
        eaten_fix == 0 and max_f <= max_b,
        f"被吞掉的反手信号 {eaten_fix} 次；最长持仓 {max_f:.0f} 根（基线 {max_b:.0f} 根），"
        f"交易数 {n_f} 笔（原样 {n_o} 笔，无过滤基线 {n_b} 笔）",
        tolerance="被吞次数 = 0 且最长持仓 ≤ 基线",
    )
    sec.info(
        "三种写法的结果（同一数据、同一手续费 10bp）",
        f"无过滤基线：{n_b} 笔，最长持仓 {max_b:.0f} 根，净值 ×{eq_b:.3f}；"
        f"原样：{n_o} 笔，最长持仓 {max_o:.0f} 根，净值 ×{eq_o:.3f}；"
        f"修好：{n_f} 笔，最长持仓 {max_f:.0f} 根，净值 ×{eq_f:.3f}",
    )
    sec.observe(
        "「过滤器」在集合论上就是一个与运算，它不会区分这个信号是用来**开仓**还是**平仓**的。"
        "把同一个过滤后的信号同时接到 entries 和 exits 上，等于同时收紧了两头——"
        "这不是风格问题，是两个不同的策略。要表达作者的意图，必须是："
        "**开仓 = 原始信号 ∩ 趋势条件，平仓 = 原始信号**（平仓不该再加条件，"
        "除非那个条件本身就是风险条件，比如止损）。"
    )
    sec.pitfall(
        "写信号逻辑时养成习惯：把 entries 和 exits 当成两个独立的集合分别定义，"
        "并各打印一次 `sum()`。只打印「总交易数」看不出这类 bug——"
        "原样版本交易数更少、看起来更「谨慎」，实际上是持仓被锁住了。"
    )

    # 图 1：信号时间轴
    idx_all = np.arange(len(close))
    fig, ax = plot.newfig(figsize=(7.2, 3.0))
    ax.plot(df.index, close, lw=0.8, color="#c9ced6", label="Close")
    ax.plot(df.index[raw_long & ok], close[raw_long & ok], "^", ms=7, color=plot.ACCENT,
            label=f"开多（通过过滤，{int(np.sum(raw_long & ok))}）")
    ax.plot(df.index[raw_short & ok], close[raw_short & ok], "v", ms=7, color=plot.ACCENT2,
            label=f"开空（通过过滤，{int(np.sum(raw_short & ok))}）")
    ax.plot(df.index[(raw_long | raw_short) & ~ok], close[(raw_long | raw_short) & ~ok],
            "x", ms=7, color="#9ca3af", label=f"被 ADX 拦掉（{int(np.sum((raw_long|raw_short) & ~ok))}）")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    sec.figure("signal-map", fig,
               caption="灰色叉是被 ADX 拦掉的交叉——其中一部分发生在持仓中，本该触发平仓/反手")

    # 图 2：持仓时长对比
    fig, ax = plot.newfig(figsize=(6.6, 3.0))
    for trades, name, col in ((base_trades, "无过滤基线", "#c9ced6"),
                              (orig_trades, "原样（平仓也被过滤）", plot.ACCENT2),
                              (fix_trades, "修好（平仓用原始信号）", plot.ACCENT)):
        bars = [t["bars"] for t in trades]
        ax.plot(sorted(bars, reverse=True), marker="o", ms=3, lw=1.2, color=col,
                label=f"{name}：{len(bars)} 笔，最长 {max(bars) if bars else 0} 根")
    ax.set_xlabel("交易序号（按持仓时长降序）")
    ax.set_ylabel("持仓 bar 数")
    ax.legend(frameon=False, fontsize=9)
    sec.figure("holding", fig,
               caption="原样版本的持仓时长分布有一条明显更长的尾巴——那就是被吞掉的反手信号留下的")


def run(sec: Section) -> None:
    df = cta_data.load(N_YEARS)
    o, high, low, close = cta_data.ohlc(df)
    raw_long, raw_short, adx, ok = build_signals(high, low, close)
    experiment_inclusion_exclusion(sec, raw_long, raw_short, ok)
    experiment_replay(sec, df, raw_long, raw_short, ok, adx)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
