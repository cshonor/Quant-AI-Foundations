"""CTA 案例数据抓取：币安公开 K 线（data-api.binance.vision），免密钥。

    python data/fetch_btc.py [天数上限]

输出 data/btc_usdt_1d.csv：Date,Open,High,Low,Close,Volume
最后一根未闭合的 K 线会被剔除——回测里放一根"还没走完的今天"是经典污染源。
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://data-api.binance.vision/api/v3/klines"
       "?symbol=BTCUSDT&interval=1d&limit=1000")
OUT = Path(__file__).with_name("btc_usdt_1d.csv")
COLS = ["open_time", "Open", "High", "Low", "Close", "Volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]


def fetch_one(end_time: int | None = None) -> pd.DataFrame:
    url = URL + (f"&endTime={end_time}" if end_time else "")
    req = urllib.request.Request(url, headers={"User-Agent": "math-lab/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=COLS)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    return df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def main() -> int:
    want_bars = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    frames, end = [], None
    while sum(len(f) for f in frames) < want_bars:
        batch = fetch_one(end)
        if batch.empty:
            break
        frames.append(batch)
        end = int(batch.index.min().timestamp() * 1000) - 1
        if len(batch) < 2:
            break
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()].iloc[:-1]      # 去掉未闭合的最后一根
    df.to_csv(OUT)
    print(f"写入 {OUT}：{len(df)} 根，{df.index.min().date()} → {df.index.max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
