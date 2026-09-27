"""对「MA 交叉 + ADX 开关 + ATR 止损」这段回测代码做逐条体检。

跑的是真实 K 线（BTCUSDT 日线，Binance 公开镜像），不是随机数据。
每个变体只改一处，好把每个 bug 的贡献单独隔离出来。
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import vectorbt as vbt

FAST_WINDOW = 20
SLOW_WINDOW = 60
ATR_WINDOW = 14
ADX_WINDOW = 14
ADX_THRESHOLD = 20
ATR_SL_MULTIPLIER = 1.5
INIT_CAPITAL = 100000
FEES = 0.001
YEARS = 3

OUT_RESULTS = "results.csv"
OUT_MD = "results.md"

# ----------------------------------------------------------------- 数据源
df_all = pd.read_csv("data/btc_usdt_1d.csv", index_col=0, parse_dates=True)
cutoff = df_all.index.max() - pd.DateOffset(years=YEARS)
ohlc = df_all.loc[df_all.index > cutoff].copy()
close, high, low, open_ = ohlc["Close"], ohlc["High"], ohlc["Low"], ohlc["Open"]
print(f"样本: {len(ohlc)} 根日线, {ohlc.index.min().date()} -> {ohlc.index.max().date()}")

# ------------------------------------------------------- ADX: vbt 里根本没有
try:
    vbt.ADX.run(high, low, close, window=ADX_WINDOW)
    print("[!] vbt.ADX 居然存在")
except AttributeError as e:
    print(f"[!] vbt.ADX 不存在 -> {type(e).__name__}: {e}")

ADX = vbt.IndicatorFactory.from_talib("ADX")
# 注意: TA-Lib 的参数名叫 timeperiod，不是 vbt 惯用的 window，传错直接 TypeError
adx_val = ADX.run(high, low, close, timeperiod=ADX_WINDOW).real

ma_fast = vbt.MA.run(close, window=FAST_WINDOW)
ma_slow = vbt.MA.run(close, window=SLOW_WINDOW)
atr = vbt.ATR.run(high, low, close, window=ATR_WINDOW)

raw_long = ma_fast.ma_crossed_above(ma_slow)
raw_short = ma_fast.ma_crossed_below(ma_slow)

# 手写一遍交叉，确认 vbt 的 ma_crossed_above 语义没歧义
manual_long = (ma_fast.ma > ma_slow.ma) & (ma_fast.ma.shift(1) <= ma_slow.ma.shift(1))
print(f"[check] ma_crossed_above 与手写实现一致: {bool((raw_long == manual_long).all())}")

trend = adx_val >= ADX_THRESHOLD
entries_long = raw_long & trend
entries_short = raw_short & trend

# ------------------------------------------------------------ 被 ADX 吃掉的信号
suppressed_entry = int((raw_long & ~trend).sum() + (raw_short & ~trend).sum())
suppressed_exit = int((raw_short & ~trend).sum() + (raw_long & ~trend).sum())
print(f"[check] 原始交叉信号 {int(raw_long.sum() + raw_short.sum())} 个, "
      f"其中 ADX<{ADX_THRESHOLD} 被吞掉 {suppressed_entry} 个 "
      f"(占 {suppressed_entry / max(int(raw_long.sum() + raw_short.sum()), 1):.1%})")
print("[check] 注意: 被吞掉的信号里包含『平仓/反手』信号 —— 详见报告第 3 节")

# 显式状态机: 数一数"手上确实有仓、却被 ADX 拦掉了平仓/反手信号"发生了几次
state = 0  # 0 flat, 1 long, -1 short
suppressed_while_in_pos = 0
blocked_events = []
for ts in close.index:
    if bool(raw_long.loc[ts]) and bool(trend.loc[ts]):
        state = 1
    elif bool(raw_short.loc[ts]) and bool(trend.loc[ts]):
        state = -1
    elif state != 0 and (bool(raw_long.loc[ts]) or bool(raw_short.loc[ts])):
        suppressed_while_in_pos += 1
        blocked_events.append((str(ts.date()), state,
                               "金叉->应反手做多" if bool(raw_long.loc[ts]) else "死叉->应反手做空"))
print(f"[check] 持仓中却被 ADX 拦掉的反手信号: {suppressed_while_in_pos} 次")
for ev in blocked_events:
    print(f"        {ev[0]}  持仓方向={ev[1]:+d}  {ev[2]}  => 该信号被丢弃, 仓位继续持有")

# ----------------------------------------------------------------- 变体构造
COMMON = dict(
    init_cash=INIT_CAPITAL,
    fees=FEES,
    direction="both",
)


def build(name: str, entries, exits, short_entries, short_exits, **kw):
    pf = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        **COMMON,
        **kw,
    )
    return name, pf


variants = []

# V0 基线：交叉即反手，无 ADX，无止损
variants.append(build(
    "V0 基线: 纯交叉反手 / 无ADX / 无止损",
    raw_long, raw_short, raw_short, raw_long,
))

# V1 用户原样：止损写成绝对金额 + 没传 high/low + 平仓信号也过了 ADX 过滤
variants.append(build(
    "V1 原样: sl_stop=ATR*1.5(当作金额) / 未传 high,low / 平仓也过滤",
    entries_long, entries_short, entries_short, entries_long,
    sl_stop=atr.atr * ATR_SL_MULTIPLIER,
))

# V2 只修止损单位（换算成百分比），其余保持 V1
variants.append(build(
    "V2 修止损单位: sl_stop=(ATR*1.5)/close / 未传 high,low / 平仓也过滤",
    entries_long, entries_short, entries_short, entries_long,
    sl_stop=(atr.atr * ATR_SL_MULTIPLIER) / close,
))

# V3 在 V2 基础上补传 high/low，让止损真能按盘中极值触发
variants.append(build(
    "V3 + 传 high,low / 平仓仍过滤",
    entries_long, entries_short, entries_short, entries_long,
    sl_stop=(atr.atr * ATR_SL_MULTIPLIER) / close,
    high=high, low=low,
))

# V4 作者口述的意图：ADX 只卡开仓，不卡平仓/反手
variants.append(build(
    "V4 作者意图: 平仓用未过滤的 raw_short/raw_long",
    entries_long, raw_short, entries_short, raw_long,
    sl_stop=(atr.atr * ATR_SL_MULTIPLIER) / close,
    high=high, low=low,
))

# V4b: 作者意图但干脆去掉止损，用来隔离"止损"本身的贡献
variants.append(build(
    "V4b 作者意图去止损",
    entries_long, raw_short, entries_short, raw_long,
    high=high, low=low,
))

# V5 在 V4 上做延时成交：信号整体后移一根，按下一根的开盘价成交
variants.append(build(
    "V5 去同期成交: V4 + shift(1) + price=Open",
    entries_long.shift(1).fillna(False).astype(bool),
    raw_short.shift(1).fillna(False).astype(bool),
    entries_short.shift(1).fillna(False).astype(bool),
    raw_long.shift(1).fillna(False).astype(bool),
    sl_stop=((atr.atr * ATR_SL_MULTIPLIER) / close).shift(1).bfill().astype(float),
    high=high, low=low, price=open_, upon_long_conflict="exit",
    upon_short_conflict="exit",
))

# ------------------------------------------------------------------- 指标汇总
days = (ohlc.index[-1] - ohlc.index[0]).days


def cagr(pf) -> float:
    end = float(pf.stats()["End Value"])
    return ((end / INIT_CAPITAL) ** (365.25 / days) - 1) * 100


def exposure(pf) -> float:
    return float((pf.asset_flow().cumsum() != 0).mean() * 100)


rows = []
for name, pf in variants:
    stats = pf.stats()
    n_trades = pf.trades.count()
    rec = pf.trades.records_readable
    rows.append({
        "变体": name,
        "总收益%": round(float(stats["Total Return [%]"]), 1),
        "CAGR%": round(cagr(pf), 1),
        "最大回撤%": round(float(stats["Max Drawdown [%]"]), 1),
        "Sharpe": round(float(stats["Sharpe Ratio"]), 3),
        "Sortino": round(float(stats["Sortino Ratio"]), 3),
        "交易数": int(n_trades),
        "胜率%": round(float(stats["Win Rate [%]"]), 1) if n_trades else np.nan,
        "平均持仓(天)": round(float((rec["Exit Timestamp"] - rec["Entry Timestamp"]).dt.days.mean()), 1) if n_trades else np.nan,
        "最长持仓(天)": int((rec["Exit Timestamp"] - rec["Entry Timestamp"]).dt.days.max()) if n_trades else np.nan,
        "敞口%": round(exposure(pf), 1),
    })

res = pd.DataFrame(rows)
res.to_csv(OUT_RESULTS, index=False)
print("\n" + res.to_string(index=False))


def to_md(frame: pd.DataFrame, path: str) -> None:
    header = "| " + " | ".join(frame.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join([header, sep, *body]) + "\n")


to_md(res, OUT_MD)
print(f"\n[stats 可用字段] {sorted(variants[0][1].stats().index.tolist())}")

# ------------------------------------------------------------- 止损到底触发过没
print("\n[止损触发核查]")
# V1 与同一套信号但完全无止损的版本对比，若资金流完全相同 => 止损一次都没生效
pf_v1_nostop = vbt.Portfolio.from_signals(
    close, entries=entries_long, exits=entries_short,
    short_entries=entries_short, short_exits=entries_long, **COMMON,
)
v1_identical = np.allclose(
    pf_v1_nostop.asset_flow().to_numpy(), variants[1][1].asset_flow().to_numpy()
)
print(f"    V1(ATR*1.5 当金额) 资金流 == 同信号无止损版本 ? {v1_identical}  "
      f"=> 止损 {'完全没生效' if v1_identical else '生效过'}")

# V4 vs V4b(同信号无止损): 平仓时点不同的那些交易 = 被止损打掉的
pf_v4 = next(p[1] for p in variants if p[0].startswith("V4 "))
pf_v4b = next(p[1] for p in variants if p[0].startswith("V4b"))
ex_v4 = set(pd.to_datetime(pf_v4.trades.records_readable["Exit Timestamp"]))
ex_v4b = set(pd.to_datetime(pf_v4b.trades.records_readable["Exit Timestamp"]))
print(f"    V4 共 {len(ex_v4)} 笔, 其中 {len(ex_v4 - ex_v4b)} 笔的离场日期与无止损版本不同"
      f" => 这些就是被止损打掉的交易")

# ------------------------------------------------- 持仓时间异常 longest trade
print("\n[最长持仓] 说明被 ADX 吞掉的平仓信号会把仓位锁死多久")
for name, pf in [variants[0], variants[3], variants[4]]:
    rec = pf.trades.records_readable
    if len(rec) == 0:
        continue
    days = (rec["Exit Timestamp"] - rec["Entry Timestamp"]).dt.days
    print(f"    {name}: 中位 {days.median():.0f} 天 / 最长 {days.max():.0f} 天")
