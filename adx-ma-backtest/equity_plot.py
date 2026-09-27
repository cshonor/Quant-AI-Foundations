"""画一张对比图: 买入持有 / 原样代码 / 作者意图版 / 修复版。"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import pandas as pd
import plotly.graph_objects as go
import vectorbt as vbt

from strategy_fixed import CONFIG, load_ohlc, run

INIT = CONFIG["init_cash"]
FEES = CONFIG["fees"]

pf_fix, cfg = run()                                   # 修复版: 下根开盘成交, ATR 百分比止损
pf_intent, _ = run({**cfg, "delay": 0})               # 作者意图版: 同根收盘成交
ohlc = load_ohlc(cfg["data_path"], cfg["years"])
close, high, low = ohlc["Close"], ohlc["High"], ohlc["Low"]

# 完全复刻原样写法: 平仓信号也过 ADX 过滤 + sl_stop 直接给美元金额 + 不传 high/low
ATR = vbt.ATR.run(high, low, close, window=cfg["atr_window"]).atr
ADX_IND = vbt.IndicatorFactory.from_talib("ADX")
adx = ADX_IND.run(high, low, close, timeperiod=cfg["adx_window"]).real
trend = adx >= cfg["adx_threshold"]
mf = vbt.MA.run(close, window=cfg["fast"])
ms = vbt.MA.run(close, window=cfg["slow"])
up = mf.ma_crossed_above(ms)
dn = mf.ma_crossed_below(ms)
pf_raw = vbt.Portfolio.from_signals(
    close,
    entries=up & trend, exits=dn & trend,
    short_entries=dn & trend, short_exits=up & trend,
    sl_stop=ATR * cfg["atr_mult"],
    init_cash=INIT, fees=FEES, direction="both",
)

pf_bh = vbt.Portfolio.from_holding(close, init_cash=INIT, fees=FEES, freq="1D")

fig = go.Figure()
for name, pf, dash, width in [
    ("买入持有 BTCUSDT", pf_bh, "dot", 1.8),
    (f"原样代码（止损失效 + 平仓也被过滤）", pf_raw, "dash", 1.8),
    ("作者意图版（ADX 只卡开仓，ATR% 止损）", pf_intent, "dashdot", 1.8),
    ("修复版（+ 下根开盘成交）", pf_fix, "solid", 2.4),
]:
    val = pf.value()
    ret = float(pf.stats()["Total Return [%]"])
    dd = float(pf.stats()["Max Drawdown [%]"])
    fig.add_trace(go.Scatter(
        x=val.index, y=val / INIT * 100,
        name=f"{name} | {ret:+.1f}% / 回撤 {dd:.0f}%",
        line=dict(dash=dash, width=width),
    ))

fig.update_layout(
    template="plotly_dark",
    title=("BTCUSDT 日线 · MA20/60 + ADX≥20 开关 + ATR1.5x 止损"
           f"（{cfg['years']} 年，净值起点=100；样本仅 {int(pf_fix.trades.count())} 笔交易，不构成投资建议）"),
    xaxis_title="日期", yaxis_title="净值 (起点=100)",
    hovermode="x unified", height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.write_html("equity.html", include_plotlyjs="cdn", full_html=True)
print("wrote equity.html")
