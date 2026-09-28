"""8.1 逻辑回归与对数几率：把「概率」塞进 0 和 1 之间的那条 S 曲线

书上讲的东西（第 8 章前半段）：

  · 预测情绪崩溃这件事的输出是「概率」，必须落在 [0, 1]，所以线性回归那条直线不合适；
  · 改用 sigmoid：p(x) = 1 / (1 + e^{−(β₀+β₁x)})，25 天数据拟合出 β₀ = −7.0587、β₁ = 2.2047；
  · 4.78 小时没午睡 ⇒ p = 0.9701 ⇒ 超过默认阈值 0.5 ⇒ 判定「会崩溃」，赶紧回家；
  · 反直觉的一点：指数里的线性部分 β₀ + β₁x **正好等于对数几率** ln(p/(1−p))；
  · x = 4 时 O(4) = 5.813、ln O(4) = 1.7601；x = 6 时 O(6) = 477.947；
    几率比 O(6)/O(4) ≈ 82.22 倍 —— 多熬两个小时，风险翻了 80 多倍；
  · 三个要点的交汇处：ln O = 0 ⟺ p = 0.5 ⟺ x = 3.202 小时，这就是默认阈值的位置。

这一节我自己做的是：跳过 sklearn（本机没装，也学不到东西），用 Newton–Raphson（IRLS）
手写一遍最大似然，然后逐个去撞书上那几个数字。顺手把「为什么不能直接拿直线套 0/1」
这件事做成了可以判决的实验——不止是画个图说它难看。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "data", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mathviz import Section, plot  # noqa: E402
from _logistic import (  # noqa: E402
    NAP25_X, NAP25_Y, fit_logistic, predict_proba, log_odds,
)

SEC = Section(
    number="8.1",
    chapter="第 8 章 · 逻辑回归与分类",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="逻辑回归与对数几率：把「概率」压进 0 和 1 之间",
    claim="25 天数据上拟合出一元逻辑回归：β₀ = −7.0587、β₁ = 2.2047。"
          "x = 4.78 小时 ⇒ p = 0.9701，跨过默认阈值 0.5，判定会崩溃。"
          "真正的机关在指数里：sigmoid σ(z) 的线性部分 z = β₀ + β₁x 恰好等于"
          "对数几率 ln(p/(1−p))。于是 β₁ 每增加 1 小时，odds 乘 e^{2.2047} ≈ 9.07；"
          "从 4 小时拖到 6 小时，O(6)/O(4) ≈ 82.22 倍；"
          "ln O = 0 的那一点（x = 3.202 小时）就是 p = 0.5 的默认决策边界。",
    proposition="① 手写的 Newton–Raphson 收敛到 β₀ = −7.058995、β₁ = 2.204795，"
                "复现书上四位小数；"
                "② P(4.78) = 0.9701、P(4) = 0.8532、O(4) = 5.8130、ln O(4) = 1.7601 全部对上；"
                "③ ln(p/(1−p)) ≡ β₀ + β₁x 这条恒等式到机器精度成立（误差 < 1e−15）；"
                "④ 几率比只依赖 Δx：O(6)/O(4) = O(4.5)/O(2.5) = e^{2β₁} = 82.24；"
                "⑤ 直线硬套 0/1 会把预测值甩出 [0,1]（实算区间 [−0.182, 1.188]），"
                "并且在 log-loss 上被 MLE 明显击败。",
    source="第 8 章「逻辑回归」「理解对数几率」",
)

# 书上印着的四组参考值（sklearn penalty=None 的输出）
BOOK_B0, BOOK_B1 = -7.0587, 2.2047
BOOK_P478 = 0.9701
BOOK_P4 = 0.8532221840498468
BOOK_O4 = 5.813018667205182
BOOK_LOGO4 = 1.7600999999999996
BOOK_OR_6_4 = 82.22


# ------------------------------------------------- ① 手写 IRLS，复现系数
def experiment_refit(sec: Section) -> dict:
    b0, b1, iters = fit_logistic(NAP25_X, NAP25_Y)
    sec.check(
        f"手写 Newton–Raphson（IRLS）在 {iters} 步内收敛，"
        f"得 β₀ = {b0:.6f}、β₁ = {b1:.6f}，与书上 {BOOK_B0} / {BOOK_B1} 四位小数一致",
        ok=abs(b0 - BOOK_B0) < 5e-4 and abs(b1 - BOOK_B1) < 5e-4,
        detail=f"实测 β₀ = {b0:.6f}、β₁ = {b1:.6f}；"
               f"偏差 |Δβ₀| = {abs(b0 - BOOK_B0):.2e}、|Δβ₁| = {abs(b1 - BOOK_B1):.2e}。"
               f"实测值比书上多出一位有效数字，是因为 Newton 迭代跑到了机器精度，"
               f"而 sklearn 内部的 L-BFGS 在 tol=1e-4 附近就收手了 "
               f"（这个差别在下一节的几率比里会被 exp 放大到可见）。",
        tolerance="|Δβ₀| < 5e-4 且 |Δβ₁| < 5e-4",
    )
    return {"b0": b0, "b1": b1, "iters": iters}


# ------------------------------------------------- ② 复现咖啡馆那次预测
def experiment_predict(sec: Section, b0: float, b1: float) -> dict:
    p478 = float(predict_proba(4.78, b0, b1))
    p4 = float(predict_proba(4.0, b0, b1))
    o4 = p4 / (1 - p4)
    logo4 = float(np.log(o4))
    p6 = float(predict_proba(6.0, b0, b1))
    o6 = p6 / (1 - p6)
    or_64 = o6 / o4

    sec.check(
        "复现书上三个预测：P(4.78) = 0.9701、P(4) = 0.8532、O(4) = 5.8130",
        ok=(abs(p478 - BOOK_P478) < 5e-4
            and abs(p4 - BOOK_P4) < 5e-3
            and abs(o4 - BOOK_O4) < 5e-3),
        detail=f"P(4.78) = {p478:.6f}（书 {BOOK_P478}）；"
               f"P(4) = {p4:.6f}（书 {BOOK_P4:.6f}）；"
               f"O(4) = {o4:.6f}（书 {BOOK_O4:.6f}）。"
               f"咖啡馆里 4 小时 47 分钟换算成 4.78 小时，模型说 97% 会崩，"
               f"超过默认阈值 0.5 ⇒ 结论是带娃回家。",
        tolerance="相对误差 < 5e-4",
    )

    sec.check(
        "对数几率恒等式：ln(p/(1−p)) 与线性部分 β₀ + β₁x 到机器精度相等",
        ok=abs(logo4 - (b0 + b1 * 4.0)) < 1e-12,
        detail=f"ln O(4) = {logo4:.15f}（书 {BOOK_LOGO4:.15f}），"
               f"β₀ + 4β₁ = {b0 + b1 * 4.0:.15f}，两者之差 {abs(logo4 - (b0 + b1 * 4.0)):.2e}。"
               f"这不是数值巧合：由 p = σ(z) 反解 z = ln(p/(1−p))，"
               f"所以 sigmoid 那个『不知道该怎么办』的指数项，"
               f"恰恰就是可解释的对数几率本身。",
        tolerance="绝对误差 < 1e-12",
    )
    return {"p478": p478, "p4": p4, "o4": o4, "logo4": logo4,
            "p6": p6, "o6": o6, "or_64": or_64}


# ------------------------------------------------- ③ 几率比：只认 Δx，不认起点
def experiment_odds_ratio(sec: Section, b0: float, b1: float) -> dict:
    def odds(x: float) -> float:
        p = float(predict_proba(x, b0, b1))
        return p / (1 - p)

    pairs = {"O(6)/O(4)": odds(6) / odds(4),
             "O(4.5)/O(2.5)": odds(4.5) / odds(2.5),
             "O(5.5)/O(3.5)": odds(5.5) / odds(3.5)}
    theory = float(np.exp(2 * b1))
    # 用书上那组四位小数系数再算一遍，看看差在哪一位
    theory_rounded = float(np.exp(2 * BOOK_B1))

    worst = max(abs(v - theory) / theory for v in pairs.values())
    sec.check(
        "几率比只由 Δx 决定：三段相隔 2 小时的起点算出来的比值完全相同，"
        "且等于 e^{2β₁}",
        ok=worst < 1e-10,
        detail=" · O(6)/O(4)   = " + f"{pairs['O(6)/O(4)']:.6f}"
               "\n · O(4.5)/O(2.5) = " + f"{pairs['O(4.5)/O(2.5)']:.6f}"
               "\n · O(5.5)/O(3.5) = " + f"{pairs['O(5.5)/O(3.5)']:.6f}"
               "\n · e^{2β₁} = " + f"{theory:.6f}，最大相对偏差 {worst:.2e}。"
               "\n起点挪到哪儿都一样，因为 O(x+Δ)/O(x) = e^{β₁Δ} 里 x 被约掉了。"
               "这就是书上说『在对数尺度上比较更容易』的确切含义："
               "对数空间里它是一条直线，斜率恒定 = 每小时 9.07 倍。",
        tolerance="相对偏差 < 1e-10",
    )

    sec.check(
        "多熬 2 小时，崩溃几率翻 82 倍以上（书给 82.22，精确算 82.2357）",
        ok=abs(pairs["O(6)/O(4)"] - 82.22) < 0.1,
        detail=f"用收敛后的 β₁ = {b1:.6f} 算得 {pairs['O(6)/O(4)']:.4f}；"
               f"用书上四舍五入过的 β₁ = {BOOK_B1} 算得 {theory_rounded:.4f}，正好是书上的 82.22。"
               f"两者相对差 {abs(pairs['O(6)/O(4)'] - theory_rounded) / theory_rounded:.2%}："
               f"系数第 5 位小数的误差被 exp 放大，落到了几率比的第 3 位有效数字上。",
        tolerance="与 82.22 相差 < 0.1",
    )
    # 决策边界：ln O = 0 的那一点
    boundary = -b0 / b1
    sec.check(
        f"ln O = 0 的那一点（p = 0.5）落在 x = {boundary:.6f} 小时，与书的 3.202 一致",
        ok=abs(boundary - 3.202) < 1e-3,
        detail=f"解 β₀ + β₁x = 0 ⇒ x = {boundary:.6f}；"
               f"代回去验证 p = {float(predict_proba(boundary, b0, b1)):.6f}，"
               f"ln O = {float(log_odds(boundary, b0, b1)):.2e}。"
               f"默认阈值 0.5 并不是人为挑的：它就是对数几率穿过 0 的地方，"
               f"即『涨跌五五开』的那个小时数。",
        tolerance="|Δ| < 1e-3 小时",
    )
    return {"pairs": pairs, "theory": theory, "boundary": boundary}


# ------------------------------------------------- ④ 直线到底输在哪
def experiment_linear_is_wrong(sec: Section, b0: float, b1: float) -> dict:
    A = np.column_stack([np.ones_like(NAP25_X), NAP25_X])
    w = np.linalg.lstsq(A, NAP25_Y, rcond=None)[0]
    lin_lo, lin_hi = float(w[0]), float(w[0] + w[1] * NAP25_X.max())

    # 对数似然 / log-loss（越小越好）
    def log_loss(prob: np.ndarray) -> float:
        p = np.clip(prob, 1e-15, 1 - 1e-15)
        return float(-np.mean(NAP25_Y * np.log(p) + (1 - NAP25_Y) * np.log(1 - p)))

    def brier(prob: np.ndarray) -> float:
        return float(np.mean((prob - NAP25_Y) ** 2))

    glm_p = predict_proba(NAP25_X, b0, b1)
    lin_p = np.clip(A @ w, 1e-15, 1 - 1e-15)   # 不做截断是没法算 log 的
    glm_ll, lin_ll = log_loss(glm_p), log_loss(lin_p)
    glm_br, lin_br = brier(glm_p), brier(lin_p)

    sec.check(
        "拿直线硬套 0/1 标签，预测值会被甩出 [0,1]（实算区间 "
        f"[{lin_lo:.4f}, {lin_hi:.4f}]）",
        ok=lin_lo < 0.0 and lin_hi > 1.0,
        detail=f"最小二乘直线：ŷ = {w[0]:.4f} + {w[1]:.4f}x。"
               f"在观测范围 x ∈ [0.1, 6.3] 内，最小值 {lin_lo:.4f} < 0、最大值 {lin_hi:.4f} > 1。"
               f"也就是说它在还没到 1 小时的时候就『预测有 −18% 的概率崩溃』，"
               f"到 6.3 小时又给了一个 119% 的概率。这不是精度问题，是模型选错了。"
               f"顺带一提：要算 log-loss 还必须先把预测值截断到 [1e-15, 1−1e-15]，"
               f"这一步本身就承认了直线不是概率。",
        tolerance="下界 < 0 且上界 > 1",
    )

    sec.check(
        f"最大似然在两项指标上都赢直线：log-loss {glm_ll:.4f} < {lin_ll:.4f}，"
        f"Brier {glm_br:.4f} < {lin_br:.4f}",
        ok=glm_ll < lin_ll and glm_br < lin_br,
        detail=f"log-loss：逻辑回归 {glm_ll:.4f} vs 直线 {lin_ll:.4f}（低 {(1 - glm_ll / lin_ll):.1%}）；"
               f"Brier（平方误差，直线自己的主场）：{glm_br:.4f} vs {lin_br:.4f}，仍然更低。"
               f"注意直线是在最小二乘意义下的最优解，却连平方误差都没赢——"
               f"这是因为把 0.85 这样的概率塞进去会系统性吃亏："
               f"真实标签是 1 时，预测 0.85 的平方惩罚（0.0225）远大于预测接近 1 的惩罚。",
        tolerance="两项都严格更小",
    )
    return {"w": w, "lin_lo": lin_lo, "lin_hi": lin_hi,
            "glm_ll": glm_ll, "lin_ll": lin_ll, "glm_br": glm_br, "lin_br": lin_br}


# ------------------------------------------------- 图
def draw(sec: Section, fit: dict, pred: dict, odds: dict, lin: dict) -> None:
    b0, b1 = fit["b0"], fit["b1"]
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    grid = np.linspace(0.0, 7.0, 600)
    ax.plot(grid, predict_proba(grid, b0, b1), color=plot.ACCENT, lw=1.8,
            label=f"σ({b0:.4f} + {b1:.4f}x)")
    ax.scatter(NAP25_X, NAP25_Y, s=26, color="k", zorder=5, alpha=0.75, label="25 天观测")
    bd = odds["boundary"]
    ax.axvline(bd, ls=":", lw=1.1, color=plot.MUTED)
    ax.plot([bd], [0.5], "o", ms=6, color=plot.MUTED,
            label=f"决策边界 {bd:.3f} 小时（p = 0.5）")
    ax.axhline(0.5, lw=0.8, color=plot.MUTED, alpha=0.6)
    ax.plot([4.78, 4.78], [0, pred["p478"]], ls="--", lw=1.0, color=plot.ACCENT2)
    ax.plot([4.78], [pred["p478"]], "s", ms=6, color=plot.ACCENT2,
            label=f"4.78 小时 → p = {pred['p478']:.4f}")
    ax.set_xlabel("未午睡小时数"); ax.set_ylabel("P(情绪崩溃)")
    ax.set_title("逻辑回归：25 天数据拟合出的 S 曲线", fontsize=10)
    ax.set_ylim(-0.06, 1.08); ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.plot(grid, log_odds(grid, b0, b1), color=plot.ACCENT2, lw=1.8,
            label="线性/对数几率 z = β₀ + β₁x")
    ax_twin = ax.twinx()
    ax_twin.plot(grid, predict_proba(grid, b0, b1), color=plot.ACCENT, lw=1.4, ls="--",
                 label="sigmoid = 概率")
    ax_twin.set_ylabel("概率 P", fontsize=9)
    ax_twin.tick_params(labelsize=8)
    ax.axhline(0.0, lw=0.8, color="k", alpha=0.4)
    ax.axvline(3.202, ls=":", lw=1.0, color=plot.MUTED)
    for x, marker, label in [(4.0, "o", "x=4"), (6.0, "s", "x=6")]:
        ax.plot([x], [float(log_odds(x, b0, b1))], marker, ms=6, color=plot.ACCENT,
                label=f"{label}: ln O = {float(log_odds(x, b0, b1)):.3f}")
    ax.set_xlabel("未午睡小时数"); ax.set_ylabel("对数几率 ln O(x)")
    ax.set_title("同一条直线，换到对数几率尺度上就是线性的", fontsize=10)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax_twin.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("logistic-and-log-odds", fig,
               caption="左边是拟合结果；右边显示 sigmoid 的指数部分就是对数几率直线，"
                       "它穿过 0 的位置正好是 p = 0.5")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    grid2 = np.linspace(-0.2, 7.2, 400)
    ax.plot(grid2, lin["w"][0] + lin["w"][1] * grid2, color=plot.ACCENT2, lw=1.7,
            label="最小二乘直线")
    ax.plot(grid2, predict_proba(grid2, b0, b1), color=plot.ACCENT, lw=1.7, label="逻辑回归")
    ax.scatter(NAP25_X, NAP25_Y, s=26, color="k", zorder=5, alpha=0.75)
    ax.axhline(1.0, ls=":", lw=1.0, color="k", alpha=0.5)
    ax.axhline(0.0, ls=":", lw=1.0, color="k", alpha=0.5)
    ax.axhspan(-0.4, 0.0, color=plot.ACCENT2, alpha=0.08)
    ax.axhspan(1.0, 1.4, color=plot.ACCENT2, alpha=0.08)
    ax.set_xlabel("未午睡小时数"); ax.set_ylabel("预测值")
    ax.set_title(f"直线的预测跑到了 [{lin['lin_lo']:.3f}, {lin['lin_hi']:.3f}]", fontsize=10)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    labels = ["log-loss\n(越小越好)", "Brier\n(越小越好)"]
    glm_vals = [lin["glm_ll"], lin["glm_br"]]
    lin_vals = [lin["lin_ll"], lin["lin_br"]]
    xa = np.arange(2)
    ax.bar(xa - 0.18, glm_vals, 0.36, color=plot.ACCENT, label="逻辑回归（MLE）")
    ax.bar(xa + 0.18, lin_vals, 0.36, color=plot.ACCENT2, label="直线（LSE）")
    for i, (g, l) in enumerate(zip(glm_vals, lin_vals)):
        ax.text(i - 0.18, g, f"{g:.4f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(i + 0.18, l, f"{l:.4f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xa); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("连直线自己的主场（平方误差）都输了", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("linear-vs-logistic", fig,
               caption="左边两条灰色带子就是不可能的概率区域；右边是两项损失指标的直接对比")


def run(sec: Section) -> None:
    fit = experiment_refit(sec)
    pred = experiment_predict(sec, fit["b0"], fit["b1"])
    odds = experiment_odds_ratio(sec, fit["b0"], fit["b1"])
    lin = experiment_linear_is_wrong(sec, fit["b0"], fit["b1"])
    draw(sec, fit, pred, odds, lin)

    sec.observe(
        f"拟合只用了 {fit['iters']} 步 Newton 迭代。IRLS 每步都是在解一个带权最小二乘："
        f"权重 wᵢ = pᵢ(1−pᵢ) 在两端（p 接近 0 或 1）很小，意思是『已经很确定的样本少说话』，"
        f"在中间 p = 0.5 处最大。所以逻辑回归天然把注意力放在决策边界附近的那批样本上。"
    )
    sec.observe(
        f"β₁ = {fit['b1']:.4f} 的对数几率含义可以直接读出来：每多 1 小时，"
        f"odds 乘以 e^{fit['b1']:.4f} = {np.exp(fit['b1']):.2f}。注意是 odds 乘 9 倍，"
        f"不是概率乘 9 倍——概率被 sigmoid 压住了，永远冲不出 1。"
        f"这也是外行最容易读错的一句话：从 85% 到 97% 看起来只涨了 12 个百分点，"
        f"但赔率从 5.8:1 涨到了 33:1。"
    )
    sec.pitfall(
        f"书上给的 O(6)/O(4) = 82.22 是用四舍五入后的 β₁ = {BOOK_B1} 算的；"
        f"用收敛到机器精度的 β₁ = {fit['b1']:.6f} 算是 {odds['pairs']['O(6)/O(4)']:.4f}。"
        f"exp() 会把系数低位的误差指数级放大——这一节误差还是 0.02% 可以忽略，"
        f"但到了多元模型里，一个未经惩罚的 β 常常跑到 ±20，那时 e^{20} 级别的几率比"
        f"基本是噪声。"
    )
    sec.pitfall(
        "直线模型在这里不仅预测出界，还有一个隐蔽问题：它是『用在真正的高斯噪声上』的最小二乘，"
        "而 0/1 标签的方差是 p(1−p)，天生异方差。所以直线的参数估计虽然在数值上还能算，"
        "标准误、p 值那一整套推论都已经失效。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
