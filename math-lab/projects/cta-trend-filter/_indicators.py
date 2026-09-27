"""CTA 案例用到的最小指标与回测工具（不依赖 vectorbt / TA-Lib）。

刻意写成"能照抄到 Go/C++"的样子：全是显式循环或 numpy 基础运算，
没有 pandas 魔法。命名带下划线前缀，run_all.py 不会把它当小节收集。
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 平滑


def sma(x: np.ndarray, n: int) -> np.ndarray:
    """简单移动平均：前 n-1 个为 NaN。"""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def wilder(x: np.ndarray, n: int, scale: str = "sum") -> np.ndarray:
    """Wilder 平滑（等价于 α=1/n 的一阶递推），种子落在 index n。

    scale="sum"   → 输出是"n 项和"量纲（TA-Lib 口径）：v[i] = v[i-1] - v[i-1]/n + x[i]
    scale="mean"  → 输出是"均值"量纲：v[i] = (v[i-1]*(n-1) + x[i]) / n

    两者只差一个常数因子 n；在 ADX 这种比值场景里会被约掉。
    """
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    if len(x) <= n or np.isnan(x[1:n + 1]).any():
        return out
    if scale == "sum":
        # TA-Lib 口径：种子只累加前 n-1 项，落 index n 之前先做一次递推
        s = np.nansum(x[1:n])
        out[n] = s - s / n + x[n]
        for i in range(n + 1, len(x)):
            out[i] = out[i - 1] - out[i - 1] / n + x[i]
    else:
        out[n] = np.nanmean(x[1:n + 1])
        for i in range(n + 1, len(x)):
            out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def ema(x: np.ndarray, alpha: float, seed: float | None = None) -> np.ndarray:
    """标准 EMA：v[i] = (1-α)·v[i-1] + α·x[i]。"""
    x = np.asarray(x, float)
    out = np.full_like(x, np.nan)
    if len(x) == 0:
        return out
    out[0] = x[0] if seed is None else seed
    for i in range(1, len(x)):
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * x[i]
    return out


# ------------------------------------------------------------------ 波动 / 趋势


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """TR_t = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)，index 0 无定义。"""
    high, low, close = map(lambda a: np.asarray(a, float), (high, low, close))
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr = np.asarray(tr, float)
    tr[0] = np.nan
    return tr


def wilder_atr(high, low, close, n: int = 14) -> np.ndarray:
    return wilder(true_range(high, low, close), n, scale="mean")


def wilder_adx(high, low, close, n: int = 14, scale: str = "sum") -> tuple[np.ndarray, ...]:
    """返回 (adx, plus_di, minus_di)。DX = 100·|+DI − −DI| / (+DI + −DI)。"""
    high, low, close = map(lambda a: np.asarray(a, float), (high, low, close))
    up = np.diff(high, prepend=high[0])
    dn = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    plus_dm[0] = minus_dm[0] = np.nan

    sm_tr = wilder(true_range(high, low, close), n, scale=scale)
    sm_plus = wilder(plus_dm, n, scale=scale)
    sm_minus = wilder(minus_dm, n, scale=scale)

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
    first = 2 * n - 1                       # TA-Lib lookback
    if len(dx) > first and not np.isnan(dx[n: first + 1]).any():
        adx[first] = np.nansum(dx[n: first + 1]) / n
        for i in range(first + 1, len(dx)):
            adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx, plus_di, minus_di


# ------------------------------------------------------------------ 信号


def crossed_above(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 上穿 b（严格：昨天 a≤b，今天 a>b）。NaN 视为 False。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    prev_a, prev_b = np.roll(a, 1), np.roll(b, 1)
    out = (prev_a <= prev_b) & (a > b)
    out[:1] = False
    return np.where(np.isnan(a) | np.isnan(b) | np.isnan(prev_a) | np.isnan(prev_b), False, out)


def crossed_below(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return crossed_above(b, a)


# ------------------------------------------------------------------ 回测


def replay(open_px: np.ndarray, long_entry, long_exit, short_entry, short_exit,
           raw_long=None, raw_short=None, fee: float = 0.001) -> tuple[list[dict], int]:
    """显式状态机回放：信号在 bar t 收盘确认，bar t+1 开盘成交。

    返回 (逐笔交易, 被吞掉的反手信号数)。`raw_long/raw_short` 是未经 ADX 过滤的
    原始交叉信号——只有在持仓中出现了原始反向信号、却没有任何一条 exit 规则响应它，
    才算"被吞掉"。
    """
    n = len(open_px)
    long_entry, long_exit = map(lambda m: np.asarray(m, bool), (long_entry, long_exit))
    short_entry, short_exit = map(lambda m: np.asarray(m, bool), (short_entry, short_exit))
    raw_long = np.asarray(raw_long, bool) if raw_long is not None else long_entry
    raw_short = np.asarray(raw_short, bool) if raw_short is not None else short_entry

    pos, entry_i, entry_px = 0, -1, 0.0
    trades: list[dict] = []
    eaten = 0
    for i in range(1, n):
        px = float(open_px[i])
        s = i - 1                                        # 上一根收盘后生成的信号

        # 1) 先处理平仓（含反手）
        if pos == 1 and long_exit[s]:
            trades.append({"side": 1, "entry": entry_i, "exit": i,
                           "entry_px": entry_px, "exit_px": px, "bars": i - entry_i,
                           "ret": (px * (1 - fee)) / (entry_px * (1 + fee)) - 1.0})
            pos = 0
        elif pos == -1 and short_exit[s]:
            trades.append({"side": -1, "entry": entry_i, "exit": i,
                           "entry_px": entry_px, "exit_px": px, "bars": i - entry_i,
                           "ret": (entry_px * (1 - fee)) / (px * (1 + fee)) - 1.0})
            pos = 0

        # 2) 原始反手信号出现了，但没有任何 exit 规则响应 → 被过滤器吞掉
        if pos == 1 and raw_short[s] and not long_exit[s]:
            eaten += 1
        if pos == -1 and raw_long[s] and not short_exit[s]:
            eaten += 1

        # 3) 开新仓
        if pos == 0:
            if long_entry[s]:
                pos, entry_i, entry_px = 1, i, px
            elif short_entry[s]:
                pos, entry_i, entry_px = -1, i, px
    return trades, eaten


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, float)
    peak = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peak))


def sharpe(returns: np.ndarray, periods_per_year: int = 365) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if r.size < 2 or np.std(r) == 0:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))
