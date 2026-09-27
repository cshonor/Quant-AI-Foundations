"""修好的 ADX + 均线 CTA 策略（不依赖 vectorbt / TA-Lib，纯 numpy）。

    python strategy.py

对照原始贴出的那段代码，改了三处（其余保持不变）：
  1. ADX 用 Wilder 口径自己实现（`vbt.ADX` 在 vectorbt 里不存在）；
  2. 止损传**百分比**：sl_stop=(k·ATR)/close，而不是价格金额；high/low 也真正用于盘中触发；
  3. 平仓用**未过滤**的反向信号（原写法把过滤后的信号同时接到 entries 和 exits 上，
     会吞掉 ADX<20 时的反手/平仓信号，把持仓锁死 —— 见 10.5）。

⚠️ 仅学习演示，不构成投资建议。
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

FAST, SLOW = 20, 60
ATR_N, ADX_N = 14, 14
ADX_THRESHOLD = 20.0
ATR_SL_MULT = 1.5
INIT_CAPITAL = 100_000.0
FEE = 0.001
N_YEARS = 3.0


def signals(high, low, close):
    fast, slow = ind.sma(close, FAST), ind.sma(close, SLOW)
    raw_long = ind.crossed_above(fast, slow)
    raw_short = ind.crossed_below(fast, slow)
    adx, _, _ = ind.wilder_adx(high, low, close, ADX_N)
    ok = np.where(np.isnan(adx), False, adx >= ADX_THRESHOLD)
    atr = ind.wilder_atr(high, low, close, ATR_N)
    sl_pct = ATR_SL_MULT * atr / close            # ← 百分比！不是金额
    return raw_long, raw_short, ok, sl_pct


def backtest(o, high, low, close, long_entry, long_exit, short_entry, short_exit,
             sl_pct, use_stop=True):
    """按 bar 回放：信号在 t-1 收盘确认，t 开盘成交；盘中用 high/low 触发止损。"""
    n = len(close)
    eq = np.empty(n)
    eq[0] = INIT_CAPITAL
    pos, entry_px, stop_px = 0, 0.0, 0.0
    fills = 0
    for i in range(1, n - 1):
        s = i - 1                                   # 信号来自上一根收盘
        px = o[i]

        # 1) 平仓 / 反手
        if pos == 1 and long_exit[s]:
            eq[i - 1] *= (1 - FEE)
            pos, fills = 0, fills + 1
        elif pos == -1 and short_exit[s]:
            eq[i - 1] *= (1 - FEE)
            pos, fills = 0, fills + 1

        # 2) 开新仓（按当前 bar 的开盘价）
        if pos == 0:
            if long_entry[s]:
                pos, entry_px = 1, px
                stop_px = px * (1 - float(sl_pct[s])) if use_stop and np.isfinite(sl_pct[s]) else -np.inf
                fills += 1
            elif short_entry[s]:
                pos, entry_px = -1, px
                stop_px = px * (1 + float(sl_pct[s])) if use_stop and np.isfinite(sl_pct[s]) else np.inf
                fills += 1

        # 3) 盘中止损
        stopped = False
        if pos == 1 and low[i] <= stop_px:
            eq[i] = eq[i - 1] * (1 - FEE) * (stop_px / o[i])
            pos, stopped = 0, True
        elif pos == -1 and high[i] >= stop_px:
            eq[i] = eq[i - 1] * (1 - FEE) * (o[i] / stop_px)
            pos, stopped = 0, True

        # 4) 当日盈亏：持仓从 open[i] 到 open[i+1]
        if not stopped:
            eq[i] = eq[i - 1] * (1 + pos * (o[i + 1] / o[i] - 1.0))
    eq[-1] = eq[-2]
    return eq, fills


def stats(name: str, eq: np.ndarray) -> dict:
    r = eq[1:] / eq[:-1] - 1.0
    r = r[np.isfinite(r)]
    return {
        "变体": name,
        "总收益%": round((eq[-1] / INIT_CAPITAL - 1) * 100, 1),
        "最大回撤%": round(ind.max_drawdown(eq) * 100, 1),
        "Sharpe": round(ind.sharpe(r), 2),
    }


def main() -> int:
    df = cta_data.load(N_YEARS)
    o, high, low, close = cta_data.ohlc(df)
    raw_long, raw_short, ok, sl_pct = signals(high, low, close)
    keep_long, keep_short = raw_long & ok, raw_short & ok

    variants = [
        ("买入持有", None),
        ("无过滤基线（交叉即反手）", (raw_long, raw_short, raw_short, raw_long)),
        ("原样（开仓平仓都被过滤 + ATR 止损）", (keep_long, keep_short, keep_short, keep_long)),
        ("修好（开仓过滤 / 平仓不过滤 + ATR 止损）", (keep_long, raw_short, keep_short, raw_long)),
    ]

    curves, rows = {}, []
    for name, sig in variants:
        if sig is None:
            eq, fills = INIT_CAPITAL * close / close[0], 1
        else:
            eq, fills = backtest(o, high, low, close, *sig, sl_pct=sl_pct)
        curves[name] = eq
        row = stats(name, eq)
        row["成交次数"] = fills
        rows.append(row)

    print(f"数据：{len(df)} 根 BTCUSDT 日线 {df.index.min().date()} → {df.index.max().date()}"
          f"｜本金 {INIT_CAPITAL:,.0f}｜单边手续费 {FEE*1000:.1f}‰")
    w = max(len(r["变体"]) for r in rows)
    for r in rows:
        print(f"  {r['变体']:<{w}}  总收益 {r['总收益%']:>7.1f}%   "
              f"最大回撤 {r['最大回撤%']:>5.1f}%   Sharpe {r['Sharpe']:>5.2f}   成交 {r['成交次数']:>3} 次")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sys.path.insert(0, str(_ROOT))
        from mathviz import plot  # noqa: E402

        fig, ax = plot.newfig(figsize=(7.2, 3.4))
        for name, eq in curves.items():
            ax.plot(df.index, eq / INIT_CAPITAL, lw=1.2, label=name)
        ax.set_yscale("log")
        ax.set_ylabel("净值 / 本金（对数）")
        ax.legend(frameon=False, fontsize=9)
        out = _HERE / "equity.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        print(f"净值图：{out}")
    except Exception as exc:                       # 画图失败不影响回测结论
        print(f"（跳过画图：{exc}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
