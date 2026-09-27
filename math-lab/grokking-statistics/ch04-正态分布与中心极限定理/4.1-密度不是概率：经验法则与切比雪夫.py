"""4.1 密度不是概率：经验法则与切比雪夫定理

书上第 4 章用一个脑筋急转弯开场：金毛寻回犬体重 ~ N(64, 3²)，
「恰好 61.63 磅」的概率是多少？答案是 0 —— 0.0973 那个数是密度，不是概率。
这一节把这个区别做成可判决的实验，并把两条"多少个标准差内有多少数据"的
规则（经验法则 vs 切比雪夫定理）放到真实右偏数据上比一比松紧。
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

import grokking_data as gd  # noqa: E402
from mathviz import Section, num, plot  # noqa: E402

SEC = Section(
    number="4.1",
    chapter="第 4 章 · 正态性的宏大展览",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="密度不是概率：经验法则与切比雪夫定理",
    claim="连续分布下单个点的概率恒为 0，pdf 值只是「密度」；"
          "经验法则给出正态数据 ±1/2/3σ 内的比例 68/95/99.7%；"
          "数据不正态时改用切比雪夫定理：至少 1−1/k² 落在 ±kσ 内。",
    proposition="① 区间概率随宽度线性收缩，density×h 的近似误差收敛阶为 1（即密度是「单位宽度的概率」）；"
                "② 经验法则的精确值是 68.27/95.45/99.73%，且与 μ、σ 的取值无关；"
                "③ 切比雪夫下界对任意分布都成立（含真实右偏的龙卷风数据），但松到几乎没有决策价值；"
                "④ 书上咖啡数据的 mean/std 可复现为 27.6 / 3.2 克。",
    source="第 4 章「正态分布 / 经验法则 / 切比雪夫定理」（金毛犬 61.63 磅、咖啡师 78 人调查）",
)

MU, SIGMA = 64.0, 3.0          # 金毛寻回犬体重（书例）
RNG = np.random.default_rng(20260927)


# ------------------------------------------------------- ① 密度 vs 概率


def experiment_density_is_not_probability(sec: Section) -> tuple[np.ndarray, np.ndarray]:
    x0 = 61.63
    pdf_val = float(stats.norm.pdf(x0, MU, SIGMA))
    sec.info(
        "书里那个 0.0973 是什么",
        f"N({MU:g}, {SIGMA:g}²) 在 x={x0} 处的 pdf = {pdf_val:.4f}；"
        f"而 P(X = {x0}) = 0（连续分布单点概率恒为 0）",
    )

    # 把区间 [x0, x0+h] 的概率除以 h，看它是否收敛到 pdf(x0)
    hs = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    probs = np.array([float(stats.norm.cdf(x0 + h, MU, SIGMA) - stats.norm.cdf(x0, MU, SIGMA))
                      for h in hs])
    ratios = probs / hs
    rate = num.convergence_rate(hs, np.abs(ratios - pdf_val))
    sec.check(
        "概率/宽度 → 密度：误差随 h 一阶收敛",
        bool(rate > 0.9 and rate < 1.1) and abs(ratios[-1] / pdf_val - 1) < 1e-3,
        f"h 从 1 缩到 0.005：P/h 由 {ratios[0]:.5f} 收敛到 {ratios[-1]:.5f}"
        f"（pdf = {pdf_val:.5f}，末项相对差 {abs(ratios[-1]/pdf_val-1):.2e}）；"
        f"对 h 的收敛阶 = {rate:.3f}",
        tolerance="收敛阶 ∈ (0.9, 1.1) 且末项相对差 < 1e-3",
    )

    # 一个区间的概率才有意义：书里 61~63 磅 = 21.08%
    area = float(stats.norm.cdf(63, MU, SIGMA) - stats.norm.cdf(61, MU, SIGMA))
    sec.check(
        "区间概率可复现书上的 21.08%",
        abs(area - 0.2108) < 5e-4,
        f"P(61 ≤ X ≤ 63) = {area:.4f}（书上 .2108）",
        tolerance="与 0.2108 差 < 5e-4",
    )
    return hs, probs


# ------------------------------------------------------- ② 经验法则


def experiment_empirical_rule(sec: Section) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact = np.array([float(stats.norm.cdf(MU + k * SIGMA, MU, SIGMA)
                            - stats.norm.cdf(MU - k * SIGMA, MU, SIGMA)) for k in (1, 2, 3)])
    rough = np.array([0.68, 0.95, 0.997])
    sec.check(
        "经验法则的精确值是 68.27 / 95.45 / 99.73%",
        bool(np.allclose(exact, [0.68268949, 0.95449974, 0.99730020], atol=1e-6)),
        "±1σ = {:.4f}%，±2σ = {:.4f}%，±3σ = {:.4f}%；书上口头版 "
        "{:.0%}/{:.1%}/{:.1%} 是四舍五入后的说法".format(
            exact[0], exact[1], exact[2], rough[0], rough[1], rough[2]),
        tolerance="与解析值差 < 1e-6",
    )

    # 换一组 μ、σ，比例不变 —— 这是"经验法则"能当经验用的前提
    other = np.array([float(stats.norm.cdf(1000 + k * 137, 1000, 137)
                            - stats.norm.cdf(1000 - k * 137, 1000, 137)) for k in (1, 2, 3)])
    sec.check(
        "经验法则与 μ、σ 的具体取值无关",
        float(np.max(np.abs(other - exact))) < 1e-12,
        f"换成 N(1000, 137²)：最大差 {np.max(np.abs(other - exact)):.2e}",
        tolerance="< 1e-12",
    )
    return exact, rough, other


# ------------------------------------------------------- ③ 切比雪夫


def experiment_chebyshev(sec: Section, exact: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = np.array([1.5, 2.0, 3.0, 4.0])
    lower_bounds = 1.0 - 1.0 / ks**2

    # (a) 对任意分布必须成立：拿真实右偏的龙卷风数据验
    x = gd.tornado_width()
    mu, sd = float(x.mean()), float(x.std(ddof=1))
    inside = np.array([float(np.mean(np.abs(x - mu) <= k * sd)) for k in ks])
    sec.check(
        "切比雪夫下界在真实右偏数据上确实成立（说「至少」没吹牛）",
        bool(np.all(inside >= lower_bounds)),
        f"龙卷风宽度（n={len(x)}，偏度 {float(stats.skew(x)):.2f}）："
        + "；".join(f"±{k:g}σ 实测 {p:.2%} ≥ 下界 {b:.2%}" for k, p, b in zip(ks, inside, lower_bounds)),
        tolerance="每个 k 上实测比例 ≥ 1−1/k²",
    )

    # (b) 但松到什么程度？真正要命的是尾部：下界允许的风险 vs 正态下的真实风险
    normal_tail = np.array([2 * float(stats.norm.sf(k)) for k in ks])
    bound_tail = 1.0 / ks**2                      # 切比雪夫：落在 ±kσ 外的概率 ≤ 1/k²
    inflation = bound_tail / normal_tail
    tail_mask = ks >= 3
    sec.check(
        "下界松到不能用：k≥3 时它允许的尾部风险比正态真实值大 10 倍以上，且 k 越大越离谱",
        bool(np.all(inflation[tail_mask] > 10)) and bool(np.all(np.diff(inflation) > 0)),
        "；".join(f"±{k:g}σ 之外：切比雪夫允许 ≤ {b:.3%}，正态实际 {n:.3%}（放大 {r:.0f} 倍）"
                 for k, b, n, r in zip(ks, bound_tail, normal_tail, inflation)),
        tolerance="k≥3 时放大 > 10 倍，且放大倍数随 k 单调递增",
    )
    return ks, lower_bounds, inside


# ------------------------------------------------------- ④ 书上咖啡数据


def experiment_barista(sec: Section) -> None:
    x = gd.barista()
    mean, median, sd = float(x.mean()), float(np.median(x)), float(x.std(ddof=1))
    sec.check(
        "书上咖啡师数据可复现：mean ≈ 27.6，median ≈ 28，std ≈ 3.2",
        abs(mean - 27.6) < 0.05 and abs(median - 28.0) < 0.01 and abs(sd - 3.2) < 0.05,
        f"n={len(x)}：mean {mean:.2f}（书 27.6），median {median:.1f}（书 28），"
        f"std(ddof=1) {sd:.3f}（书 3.2）",
        tolerance="mean 差 < 0.05，std 差 < 0.05",
    )
    # 经验法则套到咖啡数据上
    lo68, hi68 = mean - sd, mean + sd
    frac68 = float(np.mean((x >= lo68) & (x <= hi68)))
    sec.info(
        "把经验法则套回咖啡数据",
        f"±1σ = [{lo68:.1f}, {hi68:.1f}] 克（书上是 24.4~30.8），"
        f"实际落进去 {frac68:.1%} —— 与 68% 的偏差是样本量只有 78 的噪声",
    )


def run(sec: Section) -> None:
    hs, probs = experiment_density_is_not_probability(sec)
    exact, rough, _ = experiment_empirical_rule(sec)
    ks, bounds, inside = experiment_chebyshev(sec, exact)
    experiment_barista(sec)

    # ---- 图 1：pdf 与 cdf，密度 vs 面积
    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.0))
    xs = np.linspace(MU - 4 * SIGMA, MU + 4 * SIGMA, 400)
    axes[0].plot(xs, stats.norm.pdf(xs, MU, SIGMA), color=plot.ACCENT, lw=1.3)
    axes[0].fill_between(xs[(xs >= 61) & (xs <= 63)],
                         stats.norm.pdf(xs[(xs >= 61) & (xs <= 63)], MU, SIGMA),
                         alpha=0.35, color=plot.ACCENT, label="P(61≤X≤63)=21.08%")
    axes[0].plot([61.63], [stats.norm.pdf(61.63, MU, SIGMA)], "o", color=plot.ACCENT2, ms=5,
                 label="单点 pdf=0.0973，不是概率")
    axes[0].set_title("PDF：高度是密度，面积才是概率", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(xs, stats.norm.cdf(xs, MU, SIGMA), color=plot.ACCENT, lw=1.3)
    axes[1].axvline(63, ls="--", lw=1, color=plot.ACCENT2)
    axes[1].set_title("CDF：直接读出「截至 x 的概率」", fontsize=10)
    for ax in axes:
        ax.set_xlabel("体重（磅）")
        ax.tick_params(labelsize=8)
    fig.suptitle(f"金毛寻回犬体重 N({MU:g}, {SIGMA:g}²)", fontsize=11)
    sec.figure("pdf-cdf", fig, caption="同一个数 0.0973 是密度，21.08% 才是概率")

    # ---- 图 2：经验法则 vs 切比雪夫（区间内比例 / 尾部概率）
    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.0))
    w = 0.38
    idx = np.arange(len(ks))
    axes[0].bar(idx - w / 2, inside * 100, w, color=plot.ACCENT, label="龙卷风实测")
    axes[0].bar(idx + w / 2, bounds * 100, w, color=plot.MUTED, label="切比雪夫下界")
    normal_inside = np.array([float(stats.norm.cdf(k) - stats.norm.cdf(-k)) for k in ks]) * 100
    axes[0].plot(idx, normal_inside, "o--", color=plot.ACCENT2, lw=1.2, ms=4, label="正态实际")
    axes[0].set_xticks(idx, [f"±{k:g}σ" for k in ks])
    axes[0].set_ylim(0, 138)                     # 给图例留空，避免压住柱子
    axes[0].set_title("区间内比例：下界确实成立", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8, loc="upper center", ncols=3,
                   handlelength=1.2, columnspacing=1.0)

    axes[1].plot(ks, 1.0 / ks**2, "o-", color=plot.MUTED, lw=1.2, ms=4, label="切比雪夫允许的尾部")
    axes[1].plot(ks, np.array([2 * float(stats.norm.sf(k)) for k in ks]), "s-",
                 color=plot.ACCENT2, lw=1.2, ms=4, label="正态真实尾部")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("k（标准差倍数）")
    axes[1].set_title("尾部概率：下界放大几十到上千倍", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.suptitle("切比雪夫定理：保底声明 vs 可用估计", fontsize=11)
    sec.figure("chebyshev", fig, caption="左图看它成立，右图看它为什么不能当估计值用")

    sec.observe(
        f"书上那句「恰好 61.63 磅的概率是 0」不是修辞，实测就是这么回事："
        f"把区间宽度 h 从 1 缩到 0.005，P/h 稳定收敛到 pdf 值 "
        f"{float(stats.norm.pdf(61.63, MU, SIGMA)):.4f}，收敛阶 {num.convergence_rate(hs, np.abs(probs/hs - stats.norm.pdf(61.63, MU, SIGMA))):.2f}"
        f" —— 密度就是「单位宽度的概率」，所以单点（宽度 0）的概率必然是 0。"
    )
    sec.observe(
        "切比雪夫定理实测确实成立（真实右偏的龙卷风数据在每个 k 上都在下界之上）。"
        "但我原本想说「它只是保守了 20 个百分点」，实测否掉了这个说法——"
        "绝对差在 k 大时会缩小（±3σ 只差 10.8 个百分点），真正离谱的是尾部："
        "±3σ 之外切比雪夫允许 11.11%，正态真实只有 0.27%，**放大 41 倍**；"
        "±4σ 之外更夸张，允许 6.25% vs 实际 0.0063%，放大近 1000 倍。"
        "所以它是「最坏情况声明」，不是「估计值」——"
        "拿 1/k² 去计提风险，等于把尾部风险放大几十到上千倍。"
    )
    sec.pitfall(
        "用 scipy 时别把 norm.pdf(x) 当概率写进判断逻辑（if pdf > 0.05: 触发风控）——"
        "pdf 值会随 σ 变小而无限变大，量纲是 1/单位，压根不能和 0.05 比。"
        "要概率就用 cdf 相减，或者 sf（生存函数）算尾部。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
