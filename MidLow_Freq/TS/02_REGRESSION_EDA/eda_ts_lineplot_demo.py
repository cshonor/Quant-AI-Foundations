"""
与 2.2 探索性数据分析.md 中第五节代码一致；无 GUI 时用 Agg 后端并保存 PNG。
运行：python TS/02_REGRESSION_EDA/eda_ts_lineplot_demo.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
n = 500
date_rng = pd.date_range(start="2020-01-01", periods=n, freq="D")

trend = 0.1 * np.arange(n)
seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 30)
noise = np.random.normal(0, 2, n)
ts1 = trend + seasonal + noise

ts2 = 0.8 * trend + 3 * np.cos(2 * np.pi * np.arange(n) / 30) + np.random.normal(0, 1.5, n)

df = pd.DataFrame({"date": date_rng, "value1": ts1, "value2": ts2})
df.set_index("date", inplace=True)

print("=== 数据前5行 ===")
print(df.head())

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(df["value1"], color="blue", linewidth=1.2, label="value1")
plt.title("单序列时序折线图", fontsize=14)
plt.ylabel("值")
plt.grid(alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df["value1"], color="blue", linewidth=1.2, label="value1")
plt.plot(df["value2"], color="orange", linewidth=1.2, label="value2")
plt.title("多序列对比时序折线图", fontsize=14)
plt.xlabel("时间")
plt.ylabel("值")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(OUT / "eda_01_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== 描述性统计 ===")
print(df.describe())

print("\n=== 偏度与峰度 ===")
print(f"value1 偏度: {df['value1'].skew():.3f}, 峰度: {df['value1'].kurtosis():.3f}")
print(f"value2 偏度: {df['value2'].skew():.3f}, 峰度: {df['value2'].kurtosis():.3f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df["value1"], bins=40, color="steelblue", edgecolor="white", alpha=0.85, density=True)
plt.title("value1 直方图（密度尺度）")
plt.xlabel("value1")

plt.subplot(1, 2, 2)
stats.probplot(df["value1"], plot=plt)
plt.title("value1 QQ 图")

plt.tight_layout()
plt.savefig(OUT / "eda_02_hist_qq.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot([df["value1"], df["value2"]], tick_labels=["value1", "value2"])
plt.title("箱线图：离群值")
plt.grid(alpha=0.3)
plt.savefig(OUT / "eda_03_boxplot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(df["value1"], df["value2"], alpha=0.35, s=12)
plt.xlabel("value1")
plt.ylabel("value2")
plt.title("value1 vs value2 散点图")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
corr = df.corr()
im = plt.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr.columns)), corr.columns)
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", color="black", fontsize=10)
plt.title("相关矩阵")

plt.tight_layout()
plt.savefig(OUT / "eda_04_scatter_corr.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n已保存图片到:", OUT.resolve())
for name in (
    "eda_01_timeseries.png",
    "eda_02_hist_qq.png",
    "eda_03_boxplot.png",
    "eda_04_scatter_corr.png",
):
    print(" -", name)
