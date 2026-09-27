"""抓取 Binance 现货 BTCUSDT 日线作为 BTC-USD 代理数据源。

为什么不用原脚本的 vbt.YFData:
  * Yahoo 当前对本机出口返回 429 / 反爬页面，YFData.download 直接取不到数据；
  * data-api.binance.vision 是 Binance 的公开只读镜像，无需密钥，可用。
  * 差异要说清楚：BTCUSDT 与 BTC-USD(Coinbase/Yahoo 口径) 报价略有价差，
    本文结论基于 BTCUSDT；换成 futures(如 BTCUSDT.PERP) 只需改 URL。
"""
import json
import urllib.request

import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"
OUT = "data/btc_usdt_1d.csv"

COLS = [
    "open_time", "Open", "High", "Low", "Close", "Volume",
    "close_time", "quote_volume", "trades", "taker_base",
    "taker_quote", "ignore",
]


def _page(end_ms=None) -> list:
    url = f"{BASE}?symbol=BTCUSDT&interval=1d&limit=1000"
    if end_ms is not None:
        url += f"&endTime={end_ms}"
    req = urllib.request.Request(url, headers={"User-Agent": "agent-audit/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fetch(n_batches: int = 2) -> pd.DataFrame:
    """服务器单页上限 1000 根，向前翻页拼出更长历史。"""
    rows: list = []
    end_ms = None
    for _ in range(n_batches):
        page = _page(end_ms)
        if not page:
            break
        rows = page + rows
        end_ms = int(page[0][0]) - 1  # 再往前一页

    df = pd.DataFrame(rows, columns=COLS)
    df["Date"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    df = (
        df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        .astype(float)
        .sort_index()
    )
    df = df[~df.index.duplicated(keep="last")]
    # 最后一根是尚未走完的当日 K 线，剔除，避免回测里出现"未闭合 bar"
    return df.iloc[:-1]


if __name__ == "__main__":
    df = fetch()
    df.to_csv(OUT)
    print(df.shape, df.index.min(), "->", df.index.max())
    print(df.tail(3))
