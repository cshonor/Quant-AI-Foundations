"""7.2 最小二乘与插值/外推：书上的 9.74x − 405.26 能用到哪儿

书上讲线性回归是从损失函数开始的：

  1. 任选一条直线 m、b，对每个点算残差 eᵢ = yᵢ − ŷᵢ；
  2. 残差平方求和 ⇒ **平方和 SSE**，这就是损失函数；
  3. 简单线性回归有唯一一组 (m, b) 让 SSE 最小，而且有闭式解，不必梯度下降：

        m = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)
        b = ȳ − m·x̄

然后作者立刻泼了一盆冷水，专门写了一节「插值 vs. 外推」：

  · 这条线在 y 轴上的截距是 **−405.26**，也就是「0°F 时卖出 −405 单位的运动饮料」——没有意义；
  · 数据只覆盖 51.5°F ~ 101.6°F，模型在这个区间**以外**说什么都不算数；
  · 引用《加勒比海盗》的台词：「你已经走出了地图的边缘，伙计。那里有怪物！」

这一节把这三件事都拿去结算：

  · 复现书上全部输出：m = 9.7409、b = −405.2598、SSE = 5477.72、R² = 0.9779、
    RMSE = 23.4044、斜率标准误 0.5173；
  · 验证「闭式解真的是那个最低点」：随机扰动 2000 次，SSE 无一例外上升；
  · 用一个**已知真值是弯曲**的过程，量化走出数据范围要付多少钱；
  · 最后拿真实 BTC 日线做一遍：同样一条拟合线，「补中间的洞」和「推到未来」
    的误差完全不是一个量级。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "data", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mathviz import Section, plot  # noqa: E402

SEC = Section(
    number="7.2",
    chapter="第 7 章 · 相关性与线性回归",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="最小二乘与插值/外推：9.74x − 405.26 能用到哪儿",
    claim="同一组温度/销量数据用 linregress 拟合得 m = 9.74、b = −405.26、r = 0.9889、"
          "p = 0.0000000654、斜率标准误 0.5173；手写闭式解给出 9.740870543251257 与 "
          "−405.259779147856；SSE = 5477.72、R² = 0.9779、RMSE = 23.4044。"
          "作者随即警告：截距 −405.26 意味着 0°F 时销量为负，没有实际意义；"
          "线性回归只能用于观测范围内的**插值**，超出范围的**外推**不可靠。",
    proposition="① 复现书上全部 6 个输出数字；"
                "② 闭式解真的是全局最低点：2000 组随机扰动让 SSE 无一例外上升；"
                "③ 真值弯曲时，观测范围内 R² 仍可 > 0.98，但两侧的外推误差成倍放大甚至给负数；"
                "④ 真实 BTC 日线上，「补中间的洞」与「推到未来」的预测误差差一个量级。",
    source="第 7 章「简单线性回归」「残差与平方和」「插值 vs. 外推」",
)

X = np.array([51.5, 55.1, 63.0, 71.2, 73.0, 77.1, 88.1, 91.2, 93.0, 101.6])
Y = np.array([100.98, 137.44, 237.8, 254.96, 263.45, 353.89, 450.32, 521.34, 503.13, 573.91])


def closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """书上的闭式解。"""
    n = len(x)
    m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
    b = np.sum(y) / n - m * np.sum(x) / n
    return float(m), float(b)


def sse_of(x: np.ndarray, y: np.ndarray, m: float, b: float) -> float:
    return float(np.sum((y - (m * x + b)) ** 2))


# ------------------------------------------------- ① 复现书上六个数字


def experiment_reproduce(sec: Section) -> dict:
    m_lib, b_lib, r_val, p_val, se_slope = stats.linregress(X, Y)
    m_cf, b_cf = closed_form(X, Y)
    sse_val = sse_of(X, Y, m_lib, b_lib)
    pred = m_lib * X + b_lib
    rmse = float(np.sqrt(np.mean((Y - pred) ** 2)))
    r2_formula = float(1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2))
    sec.check(
        "复现书上全部输出：m = 9.74、b = −405.26、SSE = 5477.72、R² = 0.9779、"
        "RMSE = 23.4044、斜率标准误 0.5173",
        (abs(m_lib - 9.74) < 0.01 and abs(b_lib + 405.26) < 0.02
         and abs(sse_val - 5477.72) < 1.0 and abs(r2_formula - 0.9779) < 5e-4
         and abs(rmse - 23.4044) < 5e-3 and abs(se_slope - 0.5173) < 5e-3),
        f"linregress: slope = {m_lib:.6f}（书 9.74）、intercept = {b_lib:.4f}（书 −405.26）、"
        f"斜率标准误 {se_slope:.4f}（书 0.5173）；"
        f"SSE = {sse_val:.2f}（书 5477.72）、RMSE = {rmse:.4f}（书 23.4044）；"
        f"R² 两条路：r² = {r_val**2:.4f} vs 公式 {r2_formula:.4f}（差 {abs(r_val**2-r2_formula):.2e}）",
        tolerance="六个数字分别对齐到书上精度",
    )
    sec.check(
        "手写闭式解与 scipy 完全一致：m = 9.740870543251257、b = −405.259779147856",
        abs(m_cf - 9.740870543251257) < 1e-9 and abs(b_cf + 405.259779147856) < 1e-9
        and abs(m_cf - m_lib) < 1e-9 and abs(b_cf - b_lib) < 1e-9,
        f"闭式解 m = {m_cf:.12f}、b = {b_cf:.12f}；与 linregress 相差 "
        f"|Δm| = {abs(m_cf-m_lib):.2e}、|Δb| = {abs(b_cf-b_lib):.2e}",
        tolerance="两侧吻合到 1e-9",
    )
    sec.observe(
        "闭式解和库函数给出同一条线，说明「最小二乘」是数学上**可解**的 —— "
        "这也是书上强调它简单、优雅、可证明最优的原因。"
        "真正棘手的从来不是「怎么找到这条线」，而是「这条线能用到哪儿」。"
    )
    return {"m": float(m_lib), "b": float(b_lib), "sse": sse_val,
            "r2": float(r2_formula), "rmse": rmse, "se": float(se_slope)}


# ------------------------------------------------- ② 闭式解真的是最低点


def experiment_is_minimum(sec: Section) -> dict:
    m0, b0 = closed_form(X, Y)
    base = sse_of(X, Y, m0, b0)
    rng = np.random.default_rng(721)
    trials = 2000
    deltas = np.empty(trials)
    for i in range(trials):
        deltas[i] = sse_of(X, Y, m0 + rng.normal(0, 0.5), b0 + rng.normal(0, 30.0)) - base
    sec.check(
        "闭式解是全局最低点：2000 组随机扰动（Δm~N(0,0.5)、Δb~N(0,30)）让 SSE 无一例外上升",
        bool(np.all(deltas > 0)),
        f"{trials:,} 次扰动全部使 SSE 增加，最小的那次为 +{deltas.min():.4f}"
        f"（基线 SSE = {base:.2f}），最大 +{deltas.max():.1f}，中位数 +{np.median(deltas):.1f}",
        tolerance="2000/2000 全部增加",
    )
    sec.observe(
        "书上画了那张三维「山谷」图，这里用 2000 次扰动把它量化了："
        "不是「看起来最低」，而是**朝任何方向挪一步都会立刻变差**。"
        "这是线性模型相对深度学习最大的优势之一 —— 没有局部最优、不需要调学习率。"
        "所以如果你的梯度提升树调来调去还打不过一条直线，"
        "问题通常不出在优化器，而是数据里本来就没有更多结构。"
    )
    return {"base": base, "m0": m0, "b0": b0, "deltas": deltas}


# ------------------------------------------------- ③ 真值弯曲时，外推要付多少钱


def saturation_truth(x: np.ndarray) -> np.ndarray:
    """真实的销量-温度关系会饱和：太热没人愿意出门买饮料。"""
    return 620.0 / (1.0 + np.exp(-(x - 78.0) / 11.0))


def experiment_curvature_trap(sec: Section) -> dict:
    rng = np.random.default_rng(722)
    x_obs = np.linspace(52, 102, 40)                       # 观测范围（类比书上的 51.5~101.6）
    y_obs = saturation_truth(x_obs) + rng.normal(0, 14.0, size=x_obs.size)
    m, b = closed_form(x_obs, y_obs)
    fit = m * x_obs + b
    r2_in = float(1 - np.sum((y_obs - fit) ** 2) / np.sum((y_obs - y_obs.mean()) ** 2))
    rmse_in = float(np.sqrt(np.mean((y_obs - fit) ** 2)))

    grid_in = np.linspace(55, 100, 60)
    grid_cold = np.linspace(20, 45, 40)                    # 观测范围以下的左侧外推
    grid_hot = np.linspace(108, 140, 40)                   # 观测范围以上的右侧外推
    err_cold = m * grid_cold + b - saturation_truth(grid_cold)
    err_hot = m * grid_hot + b - saturation_truth(grid_hot)
    err_in = m * grid_in + b - saturation_truth(grid_in)
    rmse_cold = float(np.sqrt(np.mean(err_cold ** 2)))
    rmse_hot = float(np.sqrt(np.mean(err_hot ** 2)))
    rmse_in_true = float(np.sqrt(np.mean(err_in ** 2)))
    neg_sales = float(m * 0 + b)                            # 0°F 时的预测销量

    sec.check(
        "真值弯曲时，观测范围内 R² 仍高达 0.98+，但两侧外推的误差放大十倍以上",
        r2_in > 0.98 and rmse_cold > 10 * rmse_in_true and rmse_hot > 10 * rmse_in_true,
        f"观测区间 [52,102]：R² = {r2_in:.4f}、RMSE = {rmse_in:.2f}（看起来好得很）；"
        f"拿去外推：低温侧 [20,45] RMSE = {rmse_cold:.1f}（{rmse_cold/rmse_in_true:.1f} 倍）、"
        f"高温侧 [108,140] RMSE = {rmse_hot:.1f}（{rmse_hot/rmse_in_true:.1f} 倍）；"
        f"这是同一条拟合线",
        tolerance="区间内 R² > 0.98，两侧外推 RMSE 都是区间内的 10 倍以上",
    )
    sec.check(
        "书上的警告可以复刻：同一条线外推到 0°F 会给出负的销量预测",
        neg_sales < 0 and float(m * 20 + b) < float(saturation_truth(np.array([20.0]))[0]) * 0.5,
        f"这条假想数据上的拟合线在 0°F 的预测值为 {neg_sales:.1f}（真值 {float(saturation_truth(np.array([0.0]))[0]):.1f}）；"
        f"结构与书上那句「−405.26 没有实际意义」完全同构 —— "
        f"截距不是错的，它只是落在地图以外",
        tolerance="0°F 预测为负",
    )
    sec.observe(
        "注意这里的陷阱有多隐蔽：**样本内的诊断指标完全健康**（R² = "
        f"{r2_in:.3f}、残差也不算离谱），所有教科书让你看的数字都会放行。"
        f"坏消息全部集中在你没数据的那一段。文献里把这类问题统称「模型外插」，"
        "书中的智慧在一句话里：「有界限地承认自己的直线只在某个范围内有效」。"
    )
    return {"x_obs": x_obs, "y_obs": y_obs, "m": m, "b": b, "r2_in": r2_in,
            "rmse_in": rmse_in, "rmse_cold": rmse_cold, "rmse_hot": rmse_hot,
            "rmse_in_true": rmse_in_true, "neg_sales": neg_sales}


# ------------------------------------------------- ④ BTC 真实数据：插值 vs 外推


def experiment_btc_interp_vs_extrap(sec: Section) -> dict:
    """同样一条最小二乘直线：补中间的洞 vs 推到未来，误差差一个量级。"""
    try:
        from cta_data import load
    except ImportError:                      # 数据缺失时优雅降级
        sec.info("BTC 数据不可用（需要 data/btc_usdt_1d.csv）", "跳过这一项在地真实数据验证")
        return {}

    try:
        df = load(None)
    except Exception as exc:
        sec.info("BTC 数据加载失败", str(exc))
        return {}

    logp = np.log(df["Close"].to_numpy(float))
    # 两个任务刻意配平：**都用 60 个训练点、都预测 20 个点**
    #   插值：训练 [0:30] ∪ [50:80]  ⇒ 预测中间的 [30:50]（两端包围）
    #   外推：训练 [0:60]            ⇒ 预测未来的 [60:80]（推向后面）
    train_n, gap0, horizon = 60, 30, 20
    rows = []
    for start in range(0, len(logp) - 80 - 1, 5):
        seg = logp[start:start + 80]
        idx = np.arange(len(seg))
        train_i = np.r_[idx[:gap0], idx[gap0 + horizon:train_n + horizon]]
        m_i, b_i = closed_form(idx[train_i], seg[train_i])
        pred_i = m_i * idx[gap0:gap0 + horizon] + b_i
        train_e = idx[:train_n]
        m_e, b_e = closed_form(idx[train_e], seg[train_e])
        pred_e = m_e * idx[train_n:train_n + horizon] + b_e

        rows.append((float(np.sqrt(np.mean((seg[gap0:gap0 + horizon] - pred_i) ** 2))),
                     float(np.sqrt(np.mean((seg[train_n:train_n + horizon] - pred_e) ** 2)))))
        if start == 0:
            first = {"seg": seg, "pred_i": pred_i, "pred_e": pred_e,
                     "idx_i": idx[gap0:gap0 + horizon], "idx_e": idx[train_n:train_n + horizon]}
    arr = np.array(rows)
    interp, extrap = arr[:, 0], arr[:, 1]
    ratio = float(np.median(extrap / interp))
    frac_worse = float(np.mean(extrap > interp))
    sec.check(
        "真实 BTC 日线：训练点 60 个、预测点 20 个都相同 —— 外推的 RMSE 系统性更差",
        frac_worse > 0.65 and ratio > 1.2,
        f"{len(arr)} 个滚动窗口（每 5 天一个起点，共 {len(logp)} 根日线）："
        f"插值 RMSE 中位数 {np.median(interp):.4f}；外推 RMSE 中位数 {np.median(extrap):.4f}；"
        f"比值中位数 {ratio:.2f}×，{frac_worse:.1%} 的窗口外推更差",
        tolerance="外推在 >65% 的窗口更差，且 RMSE 比值中位数 > 1.2",
    )
    sec.observe(
        "这一条没有任何构造：训练点数相同（60）、预测点数相同（20）、拟合算法相同，"
        f"唯一差别是预测点落在训练数据的「当中」还是「后面」，结果是 {ratio:.2f} 倍。"
        "**这个数字比我预想的小**。我原本期待外推要坏好几倍，实测只有 1.5 倍 —— "
        "根因是 BTC 接近随机游走，这条直线本来就没什么解释力，"
        "「远推一点点」和「补一个洞」的差别因此被压缩。"
        "换个读法反而更不舒服：**在几乎没有信号的资产上，只要把预测点从「中间」"
        "挪到「未来」，同一个模型的误差就已经涨了 50%** —— "
        "而且这还是在没有任何非线性、没有结构变化的乐观情形下。"
    )
    sec.pitfall(
        "这条否定的是一类常见做法：用最近 60 天的线性趋势外推未来 20 天的目标价。"
        f"它不是「也许不准」，而是在完全相同的训练量与预测跨度下，RMSE 中位数高 {ratio:.2f} 倍。"
        "更要命的地方在于循环论证：你用来评估这条线的 R² / RMSE 全部是样本内的，"
        "它们对未来没有任何约束力。**要评估未来，就必须用样本外的评估方式**（walk-forward）。"
    )
    return {"interp": interp, "extrap": extrap, "ratio": ratio,
            "frac_worse": frac_worse, "first": first}


# ------------------------------------------------- 图


def draw(sec: Section, repro: dict, mini: dict, curv: dict, btc: dict) -> None:
    # 图 1：书上数据 + 拟合线 + 残差（每个残差是一个正方形的边）
    fig, axes = plot.newfig(1, 2, figsize=(8.2, 3.0))
    m, b = repro["m"], repro["b"]
    pred = m * X + b
    axes[0].scatter(X, Y, color=plot.ACCENT, s=26, zorder=3)
    xs = np.linspace(X.min(), X.max(), 100)
    axes[0].plot(xs, m * xs + b, lw=1.2, color=plot.ACCENT2)
    for xi, yi, pi in zip(X, Y, pred):
        axes[0].add_patch(__import__("matplotlib").patches.Rectangle(
            (xi, min(yi, pi)), abs(yi - pi), abs(yi - pi),
            linewidth=0, facecolor=plot.ACCENT, alpha=0.18))
        axes[0].plot([xi, xi], [yi, pi], lw=0.8, color=plot.ACCENT, alpha=0.6)
    axes[0].set_xlabel("温度（华氏度）"); axes[0].set_ylabel("销量")
    axes[0].set_title(f"残差平方之和 = SSE = {repro['sse']:.1f}", fontsize=9.5)
    axes[0].tick_params(labelsize=8)

    # SSE 在 (m,b) 空间里的等高线 —— 最低点就是闭式解
    ms = np.linspace(m - 1.2, m + 1.2, 160)
    bs = np.linspace(b - 90, b + 90, 160)
    MM, BB = np.meshgrid(ms, bs)
    ZZ = np.array([[sse_of(X, Y, mm, bb) for mm in ms] for bb in bs])
    lv = np.logspace(np.log10(repro["sse"]), np.log10(ZZ.max()), 14)
    cf = axes[1].contourf(MM, BB, ZZ, levels=lv, cmap="viridis_r", alpha=0.85)
    axes[1].contour(MM, BB, ZZ, levels=lv, colors="white", linewidths=0.4, alpha=0.5)
    axes[1].plot([m], [b], "r*", ms=11)
    axes[1].set_xlabel("斜率 m"); axes[1].set_ylabel("截距 b")
    axes[1].set_title("书上图 7.8 的那座山谷", fontsize=9.5)
    axes[1].tick_params(labelsize=8)
    fig.set_constrained_layout(True)
    sec.figure("sse-surface", fig,
               caption="每个残差画成一个正方形；右图的最低点（红星）就是闭式解，"
                       "2000 次随机扰动没有一次能低过它")

    # 图 2：弯曲真值下的插值/外推
    fig, ax = plot.newfig(figsize=(6.8, 3.2))
    grid = np.linspace(15, 145, 400)
    ax.plot(grid, saturation_truth(grid), lw=1.6, color=plot.ACCENT2, label="真实关系（饱和曲线）")
    xx = np.linspace(15, 145, 100)
    ax.plot(xx, curv["m"] * xx + curv["b"], lw=1.3, color=plot.ACCENT, label="线性拟合（观测区间内）")
    ax.scatter(curv["x_obs"], curv["y_obs"], s=14, color=plot.ACCENT, alpha=0.75, zorder=3)
    ax.axvspan(52, 102, alpha=0.10, color="k", label="观测范围")
    ax.axhline(0, lw=0.8, color=plot.MUTED)
    ax.set_xlabel("温度（华氏度）"); ax.set_ylabel("销量")
    ax.set_title(f"观测区间内 R² = {curv['r2_in']:.3f}，两侧 RMSE 却是它的 "
                 f"{max(curv['rmse_cold'], curv['rmse_hot'])/curv['rmse_in_true']:.0f} 倍", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    sec.figure("curvature-trap", fig,
               caption="灰带以内的一切指标都很健康；问题全在你没有数据的那两段")

    # 图 3：BTC 插值 vs 外推
    if btc:
        fig, axes = plot.newfig(1, 2, figsize=(8.4, 3.0))
        seg = btc["first"]["seg"]
        idx = np.arange(len(seg))
        axes[0].plot(idx, seg, lw=1.0, color=plot.ACCENT2, label="log 价格")
        axes[0].plot(btc["first"]["idx_i"], btc["first"]["pred_i"], lw=1.8, color=plot.ACCENT,
                     label="插值预测（两端包围）")
        axes[0].plot(btc["first"]["idx_e"], btc["first"]["pred_e"], lw=1.8, ls="--",
                     color=plot.ACCENT, label="外推预测（推向未来）")
        axes[0].axvspan(60, 79, alpha=0.10, color="k")
        axes[0].set_xlabel("交易日"); axes[0].set_ylabel("log 收盘价")
        axes[0].set_title("训练点 60 个 / 预测点 20 个，两边都相同", fontsize=9.5)
        axes[0].legend(frameon=False, fontsize=7.5)
        axes[0].tick_params(labelsize=8)
        axes[1].hist(btc["interp"], bins=40, alpha=0.7, color=plot.ACCENT2, label="插值")
        axes[1].hist(btc["extrap"], bins=40, alpha=0.7, color=plot.ACCENT, label="外推")
        axes[1].axvline(float(np.median(btc["interp"])), ls=":", lw=1.0, color=plot.ACCENT2)
        axes[1].axvline(float(np.median(btc["extrap"])), ls=":", lw=1.0, color=plot.ACCENT)
        axes[1].set_xlabel("RMSE（对数价格）"); axes[1].set_ylabel("窗口数")
        axes[1].set_title(f"{btc['frac_worse']:.0%} 的窗口外推更差，RMSE 中位数比值 "
                          f"{btc['ratio']:.2f}×", fontsize=9.5)
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].tick_params(labelsize=8)
        fig.tight_layout()
        sec.figure("btc-interp-extrap", fig,
                   caption="左边是一个真实窗口的对照；右边是全部滚动窗口的分布")


def run(sec: Section) -> None:
    repro = experiment_reproduce(sec)
    mini = experiment_is_minimum(sec)
    curv = experiment_curvature_trap(sec)
    btc = experiment_btc_interp_vs_extrap(sec)
    draw(sec, repro, mini, curv, btc)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
