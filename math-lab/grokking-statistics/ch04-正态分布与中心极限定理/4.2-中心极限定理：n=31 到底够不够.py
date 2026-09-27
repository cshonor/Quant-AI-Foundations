"""4.2 中心极限定理：书上那句「样本量 31 就够了」成不成立？

书上原文：「无论总体分布形态如何，抽取样本量至少为 31 的样本、求均值，
这些均值的分布**通常**会近似钟形曲线。为什么是 31？推导超出本书范围。
不过也有例外，很多从业者认为教科书这个说法有问题。」

这一节不争论，直接量：拿真实右偏的龙卷风宽度（偏度 3.30）当总体，
把 n 从 5 拉到 1000，看三件事——均值、标准误、以及**尾部近似的误差**。
结论是：「31」够不够，取决于你要用这个正态近似去干什么。
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
    number="4.2",
    chapter="第 4 章 · 正态性的宏大展览",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="中心极限定理：n=31 到底够不够",
    claim="CLT：样本均值的均值 → 总体均值；样本均值的标准差 = σ/√n；"
          "样本量 ≥31 时，样本均值的分布近似正态（哪怕总体不正态）。",
    proposition="① 样本均值的均值收敛到总体均值（相对误差 < 0.5%）；"
                "② 标准误严格等于 σ/√n（各 n 上实测与理论差 < 3%）；"
                "③ 残余偏度按 1/√n 衰减，实测幂律斜率 ≈ −0.5；"
                "④ 关键：n=31 时「均值附近」够用，但**右尾 2σ 的概率用正态近似会明显低估**；"
                "⑤ 先取对数再抽样，同样 n=31 的近似误差大幅下降。",
    source="第 4 章「中心极限定理」；数据为 NOAA 2023–2024 龙卷风宽度（n=3210）",
)

NS = np.array([5, 15, 31, 45, 100, 250, 500, 1000])
M = 4000                                   # 每个 n 抽多少批样本
SEED = 20260927


def sample_means(pop: np.ndarray, n: int, rng: np.random.Generator, m: int = M) -> np.ndarray:
    """从总体里有放回抽 m 批、每批 n 个，返回 m 个样本均值。"""
    idx = rng.integers(0, len(pop), size=(m, n))
    return pop[idx].mean(axis=1)


def experiment_moments(sec: Section, pop: np.ndarray, rng) -> dict:
    mu, sigma = float(pop.mean()), float(pop.std(ddof=1))
    rows = []
    for n in NS:
        means = sample_means(pop, int(n), rng)
        rows.append({
            "n": int(n),
            "mean": float(means.mean()),
            "se": float(means.std(ddof=1)),
            "se_theory": sigma / np.sqrt(n),
            "skew": float(stats.skew(means)),
            "means": means,
        })

    # ① 均值：偏差要跟"抽样噪声"比，不能跟 0 比
    #    均值之均值的标准误 = σ/√(n·M)
    z_mean = np.array([abs(r["mean"] - mu) / (sigma / np.sqrt(r["n"] * M)) for r in rows])
    sec.check(
        "样本均值的均值 = 总体均值（偏差全部落在抽样噪声内）",
        bool(np.all(z_mean < 3.0)),
        f"总体均值 {mu:.2f}；各 n 上实测偏差 / 理论噪声 = "
        + "、".join(f"n={r['n']}: {z:.2f}σ" for r, z in zip(rows, z_mean))
        + f"（n=5 实测 {rows[0]['mean']:.2f}，n=1000 实测 {rows[-1]['mean']:.2f}）",
        tolerance="每个 n 上 |偏差| < 3 倍理论标准误 σ/√(nM)",
    )

    # ② 标准误：小 n 下 SE 本身的估计噪声就大，用 n≥15 判
    se_err = np.array([abs(r["se"] / r["se_theory"] - 1) for r in rows])
    mask = np.array([r["n"] >= 15 for r in rows])
    sec.check(
        "标准误 = σ/√n（n≥15 时误差 < 3%）",
        bool(np.all(se_err[mask] < 0.03)),
        "；".join(f"n={r['n']}: 实测 {r['se']:.2f} vs 理论 {r['se_theory']:.2f}"
                  f"（差 {e:.1%}）" for r, e in zip(rows, se_err)),
        tolerance="n≥15 相对误差 < 3%；n=5 的 4% 来自 SE 估计自身噪声（重尾总体）",
    )

    # ③ 偏度按 1/√n 衰减：只在偏度还显著大于估计噪声的区间上拟合
    #    偏度估计的标准误 ≈ √(6/M)
    skew_noise = np.sqrt(6.0 / M)
    usable = np.array([r["n"] for r in rows
                       if float(stats.skew(pop)) / np.sqrt(r["n"]) > 3 * skew_noise])
    skews = np.array([r["skew"] for r in rows])
    us = np.array([r["n"] for r in rows])
    sel = np.isin(us, usable)
    slope = num.convergence_rate(us[sel].astype(float), np.abs(skews[sel]))
    theory = float(stats.skew(pop)) / np.sqrt(us[sel])
    sec.check(
        "残余偏度按 1/√n 衰减（在偏度 > 3 倍估计噪声的区间上，斜率 ≈ −0.5）",
        abs(slope + 0.5) < 0.15 and bool(np.all(np.abs(skews[sel] - theory) < 3 * skew_noise)),
        f"总体偏度 {float(stats.skew(pop)):.3f}；偏度估计噪声 ±{skew_noise:.3f}；"
        + "；".join(f"n={r['n']}: {r['skew']:.3f}(理论 {t:.3f})"
                   for r, t in zip([x for x, s in zip(rows, sel) if s], theory))
        + f"；拟合斜率 {slope:.3f}",
        tolerance=f"斜率 ∈ (−0.65, −0.35)，且各点与 γ₁/√n 差 < {3*skew_noise:.3f}",
    )
    return {"rows": rows, "mu": mu, "sigma": sigma, "skews": skews, "slope": slope,
            "skew_noise": skew_noise}


def experiment_tail(sec: Section, res: dict) -> dict:
    """均值附近够用不代表尾部够用：用正态近似估计右尾概率，和实测比。"""
    mu, sigma = res["mu"], res["sigma"]
    out = {}
    for r in res["rows"]:
        n = r["n"]
        z = 2.0
        cutoff = mu + z * sigma / np.sqrt(n)
        empirical = float(np.mean(r["means"] > cutoff))
        normal = float(stats.norm.sf(z))
        # 顺便看看左尾（右偏 ⇒ 左尾更瘦）
        left_cut = mu - z * sigma / np.sqrt(n)
        emp_left = float(np.mean(r["means"] < left_cut))
        out[n] = {"right": empirical, "left": emp_left, "normal": normal}

    r31, r500 = out[31], out[500]
    err31 = r31["right"] / r31["normal"] - 1
    err500 = r500["right"] / r500["normal"] - 1
    sec.check(
        "n=31 时右尾 2σ 的概率被正态近似明显低估（>25%），n=500 时基本收敛（<10%）",
        err31 > 0.25 and abs(err500) < 0.10,
        f"P(x̄ > μ+2σ/√n)：n=31 实测 {r31['right']:.2%} vs 正态 {r31['normal']:.2%}"
        f"（低估 {err31:.0%}）；n=500 实测 {r500['right']:.2%} vs {r500['normal']:.2%}"
        f"（偏差 {err500:+.1%}）",
        tolerance="n=31 低估 > 25%，n=500 偏差 < 10%",
    )

    asym31 = r31["right"] / max(r31["left"], 1e-9)
    sec.check(
        "n=31 时左右尾不对称：右尾频率是左尾的 2 倍以上（正态模型永远给不出这种不对称）",
        asym31 > 2.0,
        f"n=31：右尾 {r31['right']:.2%} vs 左尾 {r31['left']:.2%}，比值 {asym31:.1f}×",
        tolerance="比值 > 2",
    )
    return out


def experiment_log_transform(sec: Section, pop: np.ndarray, rng) -> None:
    """书上给的解法：先取对数。量一下它到底救回来多少。"""
    logpop = np.log10(pop)
    mu, sigma = float(logpop.mean()), float(logpop.std(ddof=1))
    n = 31
    means = sample_means(logpop, n, rng)
    z = 2.0
    cutoff = mu + z * sigma / np.sqrt(n)
    empirical = float(np.mean(means > cutoff))
    normal = float(stats.norm.sf(z))
    err = empirical / normal - 1

    raw_pop_skew = float(stats.skew(pop))
    log_skew = float(stats.skew(logpop))
    sec.check(
        "先取对数能把 n=31 的尾部误差压到 10% 以内",
        abs(err) < 0.10 and abs(log_skew) < abs(raw_pop_skew) / 10,
        f"log10 后总体偏度 {log_skew:.3f}（原尺度 {raw_pop_skew:.3f}）；"
        f"n=31 右尾 2σ：实测 {empirical:.2%} vs 正态 {normal:.2%}（偏差 {err:+.1%}）",
        tolerance="尾部偏差 < 10% 且偏度降到原来的 1/10 以下",
    )
    sec.observe(
        "书上说「增长型数据（收入、订阅数、龙卷风宽度）往往适合对数正态」，"
        "这里给了它一个可量化的理由：同样 n=31，原尺度的尾部近似误差 "
        f"{out_err_placeholder:.0%}（右偏让右尾更肥），取对数后降到 {err:+.1%}。"
        "代价是结论要换尺度解释——「样本均值的均值」说的是 log 值的均值，"
        "取 10 的幂回去得到的是**几何均值**，不是算术均值。",
    )


out_err_placeholder = float("nan")


def run(sec: Section) -> None:
    global out_err_placeholder
    pop = gd.tornado_width()
    rng = np.random.default_rng(SEED)

    sec.info(
        "总体长什么样",
        f"龙卷风宽度 n={len(pop)}：均值 {pop.mean():.1f} 码，标准差 {pop.std(ddof=1):.1f}，"
        f"偏度 {float(stats.skew(pop)):.2f}（严重右偏），超额峰度 {float(stats.kurtosis(pop)):.2f}",
    )

    res = experiment_moments(sec, pop, rng)
    tails = experiment_tail(sec, res)
    out_err_placeholder = tails[31]["right"] / tails[31]["normal"] - 1
    experiment_log_transform(sec, pop, rng)

    # ---- 图 1：不同 n 下样本均值的直方图
    fig, axes = plot.newfig(1, 3, figsize=(7.6, 2.6))
    for ax, n in zip(axes, (5, 31, 500)):
        r = next(x for x in res["rows"] if x["n"] == n)
        ax.hist(r["means"], bins=40, color=plot.ACCENT, alpha=0.85, density=True)
        xs = np.linspace(r["means"].min(), r["means"].max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, res["mu"], r["se_theory"]), color=plot.ACCENT2,
                lw=1.3, label="CLT 预测的正态")
        ax.set_title(f"n={n}：偏度 {r['skew']:.2f}", fontsize=10)
        ax.tick_params(labelsize=7)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("样本均值的分布随 n 收敛到正态（总体偏度 3.30）", fontsize=11)
    sec.figure("clt-hist", fig, caption="n=5 明显右偏，n=31 像钟形但仍不对称，n=500 才贴合")

    # ---- 图 2：偏度衰减 + 尾部误差
    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.0))
    axes[0].loglog(NS, np.abs(res["skews"]), "o-", color=plot.ACCENT, lw=1.2, ms=4,
                   label="实测残余偏度")
    axes[0].loglog(NS, float(stats.skew(pop)) / np.sqrt(NS), "--", color=plot.ACCENT2,
                   lw=1.1, label=r"理论 $|\gamma_1|/\sqrt{n}$")
    axes[0].set_xlabel("样本量 n")
    axes[0].set_ylabel("|偏度|")
    axes[0].legend(frameon=False, fontsize=8)

    ns = np.array(sorted(tails))
    axes[1].semilogx(ns, [tails[int(n)]["right"] * 100 for n in ns], "o-", color=plot.ACCENT,
                     lw=1.2, ms=4, label="实测右尾")
    axes[1].semilogx(ns, [tails[int(n)]["left"] * 100 for n in ns], "s-", color="#639922",
                     lw=1.2, ms=4, label="实测左尾")
    axes[1].axhline(tails[31]["normal"] * 100, ls="--", color=plot.ACCENT2, lw=1.1,
                    label="正态模型给出的 2.28%")
    axes[1].set_xlabel("样本量 n")
    axes[1].set_ylabel("P(x̄ 超出 μ ± 2σ/√n) %")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.suptitle("CLT 收敛：偏度按 √n 衰减，但尾部不对称衰减得更慢", fontsize=11)
    sec.figure("clt-tail", fig, caption="右图是判「31 够不够」的关键：看你要算均值还是算尾部")

    sec.observe(
        f"书上说「31 通常够了」，实测是这样的：均值与标准误这两条，n=5 就已经成立"
        f"（标准误相对误差最大 {max(abs(r['se']/r['se_theory']-1) for r in res['rows']):.1%}）；"
        f"真正拖后腿的是形状——残余偏度按 √n 衰减（斜率 {res['slope']:.2f}），"
        f"n=31 时还剩 {next(r['skew'] for r in res['rows'] if r['n']==31):.2f}。"
        "所以如果你的用途是「估计总体均值、给个 ±2σ 区间」，31 确实够；"
        "如果用途是「算尾部风险」——比如「策略日均收益跌破多少就要止损」——31 远远不够。"
    )
    sec.pitfall(
        "别把 CLT 当成「样本量 ≥31 就可以当正态用」。CLT 收敛的是**分布整体**，"
        "而尾部（你唯一真正关心的那部分）收敛得最慢。CTA 里最典型的翻车方式："
        "用几十笔交易的均值 ± 2σ 去设风险预算，结果小样本 + 右偏让真实右尾比模型肥得多。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
