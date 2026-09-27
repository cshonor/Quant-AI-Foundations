"""参数敏感性体检: 把 ADX 阈值与均线组合铺开，看收益是"规律"还是"噪声"。

每个格子只有十几笔交易，看的是分布形状而不是"最佳参数"。
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt

INIT_CAPITAL = 100000
FEES = 0.001
df_all = pd.read_csv("data/btc_usdt_1d.csv", index_col=0, parse_dates=True)
ohlc = df_all.loc[df_all.index > df_all.index.max() - pd.DateOffset(years=3)].copy()
close, high, low = ohlc["Close"], ohlc["High"], ohlc["Low"]

ADX_IND = vbt.IndicatorFactory.from_talib("ADX")
ATR_WINDOW = 14
atr = vbt.ATR.run(high, low, close, window=ATR_WINDOW).atr
sl_pct = atr * 1.5 / close

THRESHOLDS = [0, 15, 20, 25, 30, 35]
PAIRS = [(10, 30), (20, 60), (30, 90), (50, 150)]

rows_ret, rows_sharpe, rows_n = [], [], []
for fast, slow in PAIRS:
    mf = vbt.MA.run(close, window=fast)
    ms = vbt.MA.run(close, window=slow)
    raw_long = mf.ma_crossed_above(ms)
    raw_short = mf.ma_crossed_below(ms)
    adx_val = ADX_IND.run(high, low, close, timeperiod=14).real

    r_row, s_row, n_row = {}, {}, {}
    for th in THRESHOLDS:
        trend = adx_val >= th
        pf = vbt.Portfolio.from_signals(
            close,
            entries=raw_long & trend,
            exits=raw_short,
            short_entries=raw_short & trend,
            short_exits=raw_long,
            high=high, low=low,
            sl_stop=sl_pct,
            init_cash=INIT_CAPITAL, fees=FEES, direction="both",
        )
        st = pf.stats()
        key = f"ADX≥{th}"
        r_row[key] = round(float(st["Total Return [%]"]), 1)
        s_row[key] = round(float(st["Sharpe Ratio"]), 2)
        n_row[key] = int(pf.trades.count())
    rows_ret.append({"均线": f"{fast}/{slow}", **r_row})
    rows_sharpe.append({"均线": f"{fast}/{slow}", **s_row})
    rows_n.append({"均线": f"{fast}/{slow}", **n_row})


def to_md(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *body])


ret_df = pd.DataFrame(rows_ret)
sharpe_df = pd.DataFrame(rows_sharpe)
cnt_df = pd.DataFrame(rows_n)

bh = (close.iloc[-1] / close.iloc[0] - 1) * 100
bh_pf = vbt.Portfolio.from_holding(close, init_cash=INIT_CAPITAL, fees=FEES)
bh_maxdd = float(bh_pf.stats()["Max Drawdown [%]"])
bh_sharpe = float(bh_pf.stats()["Sharpe Ratio"])

print("== 标的: BTCUSDT 日线,", ohlc.index[0].date(), "->", ohlc.index[-1].date(), f"({len(ohlc)} 根)")
print(f"== 买入持有基准: 收益 {bh:.1f}% / 最大回撤 {bh_maxdd:.1f}% / Sharpe {bh_sharpe:.2f}")
print("\n== 总收益% ==\n" + ret_df.to_string(index=False))
print("\n== Sharpe ==\n" + sharpe_df.to_string(index=False))
print("\n== 交易笔数 ==\n" + cnt_df.to_string(index=False))

with open("sweep.md", "w", encoding="utf-8") as fh:
    fh.write(f"标的 BTCUSDT 日线 {ohlc.index[0].date()} -> {ohlc.index[-1].date()} ({len(ohlc)} 根)\n\n")
    fh.write(f"买入持有基准: 收益 {bh:.1f}% / 最大回撤 {bh_maxdd:.1f}% / Sharpe {bh_sharpe:.2f}\n\n")
    fh.write("### 总收益%\n\n" + to_md(ret_df) + "\n\n")
    fh.write("### Sharpe\n\n" + to_md(sharpe_df) + "\n\n")
    fh.write("### 交易笔数\n\n" + to_md(cnt_df) + "\n")
