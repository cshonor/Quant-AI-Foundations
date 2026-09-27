"""《精通统计学》(Grokking Statistics, Thomas Nield) 用到的数据集。

- 龙卷风宽度：NOAA 2023–2024 真实数据，书中第 2/4 章的主角（严重右偏）。
  来源 CSV 已精简为单列，落在 data/tornado_width.csv。
- 咖啡师调查：书中第 4 章 78 位咖啡师给出的「16 oz 完美咖啡用粉量（克）」，照书抄录。
- BTCUSDT 日收益：第 4 章「六西格玛事件」的现实对照（金融收益是肥尾的）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent

# 书中第 4 章原文数据：78 位咖啡师给出的克数（书里算得 mean 27.6 / std 3.2）
BARISTA_GRAMS = np.array([
    27, 22, 28, 27, 26, 28, 31, 25, 25, 33, 24, 28, 28, 30,
    26, 25, 29, 28, 30, 30, 22, 28, 36, 27, 26, 31, 30, 25,
    27, 32, 25, 23, 27, 24, 28, 24, 27, 28, 30, 26, 33, 29,
    31, 29, 27, 31, 20, 23, 25, 29, 30, 24, 29, 28, 25, 25,
    30, 30, 27, 36, 26, 31, 24, 29, 34, 27, 23, 29, 27, 19,
    28, 30, 26, 29, 27, 30, 29, 28,
], dtype=float)


def tornado_width() -> np.ndarray:
    """龙卷风宽度（码）。n=3210，严重右偏：偏度约 3.3。"""
    s = pd.read_csv(DATA / "tornado_width.csv")["TOR_WIDTH"].astype(float)
    return s.to_numpy()


def barista() -> np.ndarray:
    """78 位咖啡师的用粉量（克）。"""
    return BARISTA_GRAMS.copy()


def btc_daily_returns() -> np.ndarray:
    """BTCUSDT 日线对数收益（%），剔除最后一根未闭合 K 线。"""
    df = pd.read_csv(DATA / "btc_usdt_1d.csv", index_col=0, parse_dates=True)
    close = df["Close"].to_numpy(float)
    r = np.diff(np.log(close)) * 100.0
    return r
