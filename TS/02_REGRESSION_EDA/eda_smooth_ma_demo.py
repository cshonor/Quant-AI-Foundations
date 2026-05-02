"""
2.3 平滑示例：原序列 vs 简单移动平均（SMA），保存对比图到 img/。
运行：python TS/02_REGRESSION_EDA/eda_smooth_ma_demo.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(7)
n = 400
idx = pd.date_range("2020-01-01", periods=n, freq="D")
trend = 0.08 * np.arange(n)
seasonal = 4 * np.sin(2 * np.pi * np.arange(n) / 25)
noise = np.random.normal(0, 1.8, n)
raw = pd.Series(trend + seasonal + noise, index=idx, name="raw")

ma7 = raw.rolling(window=7, center=True).mean()
ma30 = raw.rolling(window=30, center=True).mean()

# 图1：上下两幅，短窗 vs 长窗
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(raw.index, raw.values, color="gray", alpha=0.55, linewidth=0.9, label="原序列")
axes[0].plot(ma7.index, ma7.values, color="tab:blue", linewidth=1.4, label="SMA 窗宽=7")
axes[0].set_title("原序列 vs 移动平均（窗宽 7）")
axes[0].legend(loc="upper left")
axes[0].grid(alpha=0.25)

axes[1].plot(raw.index, raw.values, color="gray", alpha=0.55, linewidth=0.9, label="原序列")
axes[1].plot(ma30.index, ma30.values, color="tab:orange", linewidth=1.4, label="SMA 窗宽=30")
axes[1].set_title("原序列 vs 移动平均（窗宽 30）")
axes[1].legend(loc="upper left")
axes[1].grid(alpha=0.25)
axes[1].set_xlabel("时间")
plt.tight_layout()
fig.savefig(OUT / "eda_smooth_ma_compare.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# 图2：同轴三线，便于看滞后与光滑度
fig2, ax = plt.subplots(figsize=(11, 4))
ax.plot(raw.index, raw.values, color="gray", alpha=0.45, linewidth=0.9, label="原序列")
ax.plot(ma7.index, ma7.values, color="tab:blue", linewidth=1.2, label="SMA(7)")
ax.plot(ma30.index, ma30.values, color="tab:orange", linewidth=1.2, label="SMA(30)")
ax.set_title("同图对比：窗越大越光滑、拐点滞后越明显（示意）")
ax.legend(loc="upper left")
ax.grid(alpha=0.25)
ax.set_xlabel("时间")
plt.tight_layout()
fig2.savefig(OUT / "eda_smooth_ma_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

print("已保存:", OUT / "eda_smooth_ma_compare.png")
print("已保存:", OUT / "eda_smooth_ma_overlay.png")
