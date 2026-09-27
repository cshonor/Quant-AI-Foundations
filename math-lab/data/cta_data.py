"""CTA 案例数据加载：BTCUSDT 日线（币安公开数据，已剔除非闭合的最后一根）。

    python data/fetch_btc.py     # 重新抓取/更新 btc_usdt_1d.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).with_name("btc_usdt_1d.csv")


def load(n_years: float | None = 3.0) -> pd.DataFrame:
    """读 CSV；n_years 只保留最近 N 年（None = 全量）。"""
    if not CSV.exists():
        raise FileNotFoundError(f"缺少数据文件 {CSV}，先跑 `python data/fetch_btc.py`")
    df = pd.read_csv(CSV, index_col=0, parse_dates=True).sort_index()
    if n_years:
        end = df.index.max()
        df = df.loc[df.index > end - pd.DateOffset(days=int(round(365.25 * n_years)))]
    return df


def ohlc(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """返回 (open, high, low, close) 四个 float 数组。"""
    return tuple(df[c].to_numpy(float) for c in ("Open", "High", "Low", "Close"))
