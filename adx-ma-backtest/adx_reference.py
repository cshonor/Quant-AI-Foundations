"""纯 pandas 手写 Wilder ADX / ATR，与 TA-Lib 逐点对齐。

目的: 后面要把策略移植到 Go/C++，手写的这份就是可照抄的参考实现。
只要这里和 talib 逐根 bar 对齐到 1e-8 以内，移植口径就有据可依。
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import talib
import vectorbt as vbt


def _true_range(high, low, close, n):
    """TR, index 0 置 NaN（没有 prev_close），后面所有计数从 index 1 开始。"""
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = np.nan
    return tr


def _wilder(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder 平滑，等价于 A[i] = (A[i-1]*(n-1) + x[i]) / n。

    种子取前 n 项**均值**并落在 index n，所以递推里 x[i] 也必须除以 n
    （用「前 n 项之和」做种子的写法才能省掉这个 /n）。
    """
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) <= n or np.isnan(x[1: n + 1]).any():
        return out
    out[n] = np.nanmean(x[1: n + 1])
    for i in range(n + 1, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def _wilder_sum(x: np.ndarray, n: int) -> np.ndarray:
    """TA-Lib 口径的 Wilder 平滑（保持"和"的量纲，DI 里用作比值会自动约掉）。

    种子 = 前 n-1 项之和 S，然后立刻做一次 S - S/n + x[n] 得到 index n 的值。
    """
    out = np.full_like(x, np.nan, dtype=float)
    if len(x) <= n or np.isnan(x[1:n]).any():
        return out
    s = np.nansum(x[1:n])
    out[n] = s - s / n + x[n]
    for i in range(n + 1, len(x)):
        out[i] = out[i - 1] - out[i - 1] / n + x[i]
    return out


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    return _wilder(_true_range(high, low, close, n), n)


def wilder_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)

    tr = _true_range(high, low, close, n)
    plus_dm[0] = minus_dm[0] = np.nan

    # 注意: TA-Lib 内部用的是「和」而不是「均值」做种子，且种子只累加前 n-1 项
    # (ta_ADX.c: `i = optInTimePeriod - 1; while(i--) prevXXX += ...`)，
    # 落第一个值之前还会先做一次 `-prev/n + x[n]`。照抄才能逐根 bar 对齐。
    sm_tr = _wilder_sum(tr, n)
    sm_plus = _wilder_sum(plus_dm, n)
    sm_minus = _wilder_sum(minus_dm, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * sm_plus / sm_tr
        minus_di = 100.0 * sm_minus / sm_tr
        di_sum = plus_di + minus_di
        dx = np.where(
            (sm_tr > 0.0) & (np.abs(di_sum) > 0.0),
            100.0 * np.abs(plus_di - minus_di) / di_sum,
            np.nan,
        )

    adx = np.full_like(dx, np.nan)
    first = 2 * n - 1                            # TA-Lib: lookback = 2n-1
    if len(dx) <= first or np.isnan(dx[n: first + 1]).any():
        return adx
    adx[first] = np.nansum(dx[n: first + 1]) / n  # ta_ADX.c: prevADX = sumDX / n
    for i in range(first + 1, len(dx)):
        adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx


if __name__ == "__main__":
    df = pd.read_csv("data/btc_usdt_1d.csv", index_col=0, parse_dates=True)
    df = df.loc[df.index > df.index.max() - pd.DateOffset(years=3)]
    h, l, c = df["High"].to_numpy(float), df["Low"].to_numpy(float), df["Close"].to_numpy(float)

    ref_adx = talib.ADX(h, l, c, timeperiod=14)
    my_adx = wilder_adx(h, l, c, 14)
    mask = np.isfinite(ref_adx) & np.isfinite(my_adx)
    print(f"ADX  有效点位 {mask.sum()} / {len(h)}")
    print(f"ADX  最大绝对误差 = {np.nanmax(np.abs(ref_adx[mask] - my_adx[mask])):.3e}")
    idx = int(np.argmax(np.isfinite(ref_adx)))
    print(f"ADX  talib 首值位置 = {idx} (期望 2n-1 = 27)")

    ref_atr = talib.ATR(h, l, c, timeperiod=14)
    my_atr = wilder_atr(h, l, c, 14)
    m2 = np.isfinite(ref_atr) & np.isfinite(my_atr)
    print(f"ATR  有效点位 {m2.sum()} / {len(h)}")
    print(f"ATR  最大绝对误差 = {np.nanmax(np.abs(ref_atr[m2] - my_atr[m2])):.3e}")

    # vbt.ATR 是简单移动平均口径，和 Wilder ATR 不是一回事，顺手量一下差多少
    vbt_atr = vbt.ATR.run(df["High"], df["Low"], df["Close"], window=14).atr.to_numpy()
    m3 = np.isfinite(ref_atr) & np.isfinite(vbt_atr)
    print(f"ATR  vbt(SMA口径) vs talib(Wilder口径) 平均相对偏差 = "
          f"{np.nanmean(np.abs(vbt_atr[m3] - ref_atr[m3]) / ref_atr[m3]) * 100:.2f}%")
