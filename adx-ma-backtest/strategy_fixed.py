"""修复版策略。

相对原贴代码改了这些地方（每一条都对应 AUDIT.md 里的一个 bug）:

1. `vbt.ADX` 在 vectorbt 里根本不存在 -> 用 `IndicatorFactory.from_talib("ADX")`，
   参数名是 TA-Lib 的 `timeperiod`，不是 vbt 的 `window`。
2. `sl_stop` 是**百分比**（0.01 = 1%），不是金额 -> ATR 止损要除以价格。
3. `from_signals` 必须传 `high` / `low`，否则回测里 `low = min(open, close)`，
   盘中插针打不到止损。
4. ADX 只卡**开仓**，`exits` / `short_exits` 用未过滤的交叉信号，
   否则 `exits=entries_short` 会连平仓一起吞掉（原贴第 2 条的说法和代码是矛盾的）。
5. 明确写 `direction="both"`，别依赖版本相关的默认行为。
6. 明确写 `upon_long_conflict` / `upon_short_conflict`，别吃默认的 `"ignore"`。
7. 默认按下根开盘成交（`delay=1` + `price=open`），不用"同根收盘价算信号又按它成交"。
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import pandas as pd
import vectorbt as vbt

CONFIG = dict(
    data_path="data/btc_usdt_1d.csv",
    years=3,
    fast=20,
    slow=60,
    atr_window=14,
    adx_window=14,
    adx_threshold=20,
    atr_mult=1.5,
    fees=0.001,
    slippage=0.0,
    init_cash=100000,
    filter_exits=False,   # True 就是原贴写法（连平仓都过滤，不推荐）
    delay=1,              # 1 = 信号后移一根，按下根开盘成交；0 = 同根收盘成交
)


def load_ohlc(path: str, years: int) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    if years:
        df = df.loc[df.index > df.index.max() - pd.DateOffset(years=years)]
    return df


def run(cfg: dict | None = None):
    cfg = {**CONFIG, **(cfg or {})}
    ohlc = load_ohlc(cfg["data_path"], cfg["years"])
    close, high, low, opn = ohlc["Close"], ohlc["High"], ohlc["Low"], ohlc["Open"]

    ma_fast = vbt.MA.run(close, window=cfg["fast"]).ma
    ma_slow = vbt.MA.run(close, window=cfg["slow"]).ma
    atr = vbt.ATR.run(high, low, close, window=cfg["atr_window"]).atr

    ADX_IND = vbt.IndicatorFactory.from_talib("ADX")
    adx = ADX_IND.run(high, low, close, timeperiod=cfg["adx_window"]).real

    cross_up = (ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))
    cross_dn = (ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))
    trend = adx >= cfg["adx_threshold"]

    long_entry = cross_up & trend
    short_entry = cross_dn & trend
    # 平仓信号是否也要被 ADX 过滤，做成开关，便于对比原贴写法
    long_exit = (cross_dn & trend) if cfg["filter_exits"] else cross_dn
    short_exit = (cross_up & trend) if cfg["filter_exits"] else cross_up

    sl_pct = cfg["atr_mult"] * atr / close   # 百分比口径，这个是关键

    d = cfg["delay"]
    if d:
        shift = lambda s: s.shift(d).fillna(False).astype(bool)  # noqa: E731
        kwargs = dict(
            entries=shift(long_entry),
            exits=shift(long_exit),
            short_entries=shift(short_entry),
            short_exits=shift(short_exit),
            sl_stop=sl_pct.shift(d).bfill(),
            price=opn,                       # 下一根的开盘价
        )
    else:
        kwargs = dict(
            entries=long_entry, exits=long_exit,
            short_entries=short_entry, short_exits=short_exit,
            sl_stop=sl_pct,
        )

    pf = vbt.Portfolio.from_signals(
        close,
        high=high, low=low,                  # 不传则止损只认 open/close
        direction="both",                    # 别靠默认
        upon_long_conflict="exit",           # 默认是 ignore，同根多空信号会被丢掉
        upon_short_conflict="exit",
        fees=cfg["fees"],
        slippage=cfg["slippage"],
        init_cash=cfg["init_cash"],
        freq="1D",
        **kwargs,
    )
    return pf, cfg


if __name__ == "__main__":
    pf, cfg = run()
    bh = vbt.Portfolio.from_holding(
        pd.read_csv(cfg["data_path"], index_col=0, parse_dates=True)
        .sort_index()
        .loc[lambda d: d.index > d.index.max() - pd.DateOffset(years=cfg["years"]), "Close"],
        init_cash=cfg["init_cash"], fees=cfg["fees"], freq="1D",
    )
    keys = ["Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio", "Sortino Ratio",
            "Total Trades", "Win Rate [%]", "Calmar Ratio"]
    out = pd.DataFrame({"策略": pf.stats()[keys], "买入持有": bh.stats()[keys]})
    print(f"标的 BTCUSDT 日线  近 {cfg['years']} 年  {cfg['fast']}/{cfg['slow']} MA  "
          f"ADX≥{cfg['adx_threshold']}  ATR止损 {cfg['atr_mult']}x")
    print(out.to_string())
    print("\n※ 每个格子只有十几笔交易，以上数字仅用于验证代码行为，不构成任何策略结论。")
