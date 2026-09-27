"""统一的画图入口：中文字体 + 固定尺寸 + 统一配色。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

CJK_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Heiti TC", "Songti SC",
                  "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]

INK = "#1f2328"
MUTED = "#6b7280"
ACCENT = "#185FA5"
ACCENT2 = "#D85A30"
GRID = "#e5e7eb"


def setup() -> None:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    fonts = [c for c in CJK_CANDIDATES if c in available] or ["DejaVu Sans"]
    plt.rcParams.update({
        "font.sans-serif": fonts + ["DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "font.size": 11,
        "axes.titlesize": 12,
    })


setup()


def newfig(nrows: int = 1, ncols: int = 1, figsize=(7.2, 3.6)):
    """返回一个紧凑的 fig, ax（单子图时 ax 不是数组）。"""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    return fig, ax
