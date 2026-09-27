"""5.1 应用 CLT：样本量到底买回多少确定性

书上第 5 章开头给了道题：健康成年人体温 μ=98.2°F、σ=0.73°F，抽 31 人，
问平均体温低于 98°F 的概率。答案是 6.36%。作者的本意是演示「样本均值的分布
比总体窄」，顺带埋了一句经验法则：样本量要大于 30。

这一节把这件事推到底：
  · 窄多少？σ/√n —— 幂律斜率应该精确等于 −0.5；
  · 想再窄一半要花多少样本？不是 2 倍，是 4 倍（这是个常被忽略的成本）；
  · 「n>30 就够」这条规则，在 σ 未知、总体右偏的时候还成立吗？（→5.2）
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
    number="5.1",
    chapter="第 5 章 · 样本量超能力",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="应用 CLT：样本量到底买回多少确定性",
    claim="体温题：μ=98.2°F、σ=0.73°F、n=31 ⇒ P(样本均值 < 98°F) = 6.36%。"
          "作者的结论是「样本量越大，钟形曲线越窄、置信度越高」，"
          "并给出经验法则：样本量应大于 30。",
    proposition="① 那条曲线变窄的速度严格是 σ/√n（幂律斜率 −0.5）；"
                "② 想把误差范围压到一半，样本量要翻 4 倍而不是 2 倍；"
                "③ 真实右偏数据（龙卷风 n=3210）上重抽样，样本均值的散布同样服从 σ/√n；"
                "④ 书上题目假设 σ 已知，实测把 σ 换成样本估计 s 之后结论会漂 —— 差额正好通向 5.2。",
    source="第 5 章「应用中心极限定理」（图 5.1/5.2，体温例题）；"
           "数据：NOAA 龙卷风宽度 3210 条",
)

BODY_MU = 98.2
BODY_SIGMA = 0.73
BODY_N = 31
BODY_TARGET = 98.0


# ------------------------------------------------- ① 复现书上的体温题


def experiment_body_temp(sec: Section) -> tuple[np.ndarray, np.ndarray, float]:
    ns = np.array([5, 10, 20, 31, 60, 100, 250, 1000], dtype=float)
    probs = np.array([
        float(stats.norm.cdf(BODY_TARGET, BODY_MU, BODY_SIGMA / np.sqrt(n)))
        for n in ns
    ])
    p31 = probs[list(ns).index(BODY_N)]
    sec.check(
        "复现书上例题：n=31 时 P(样本均值 < 98°F) = 6.36%",
        abs(p31 - 0.0636) < 5e-4,
        f"实测 {p31:.4%}（书上 0.0636，差 {abs(p31-0.0636)*1e4:.2f} 个万分点）",
        tolerance="与 0.0636 的绝对差 < 5e-4",
    )
    return ns, probs, p31


def experiment_sqrt_law(sec: Section, ns: np.ndarray) -> float:
    """样本均值的标准差 = σ/√n —— 而且是很干净的一条幂律。"""
    se_theory = BODY_SIGMA / np.sqrt(ns)
    rng = np.random.default_rng(20260927)
    M = 40_000
    se_emp, slopes = [], []
    for n in ns:
        n = int(n)
        # 每次独立抽样 n 个个体，重复 M 次；分布宽度就是样本均值的标准差
        samples = rng.normal(BODY_MU, BODY_SIGMA, size=(M, n))
        se_emp.append(float(samples.mean(axis=1).std(ddof=1)))
    se_emp = np.array(se_emp)

    rel = np.abs(se_emp / se_theory - 1)
    slope = num.convergence_rate(ns, se_emp)
    sec.check(
        "样本均值的散布 = σ/√n（不是 σ/n，也不是 σ）",
        bool(np.all(rel < 0.02)) and abs(slope + 0.5) < 0.02,
        f"M={M:,} 次重抽样；SE 对 n 的幂律斜率 {slope:.4f}（理论 −0.5）；"
        f"与 σ/√n 的最大相对偏差 {rel.max():.2%}；"
        + "；".join(f"n={int(n)}: {e:.4f} vs {t:.4f}"
                   for n, e, t in list(zip(ns, se_emp, se_theory))[:4]),
        tolerance="相对误差 <2% 且斜率 ∈ (−0.52, −0.48)",
    )
    return slope


# ------------------------------------------------- ② 想更准要付 4 倍价钱


def experiment_cost(sec: Section) -> None:
    """√n 律最要紧的推论：精度是平方根买的，不是线性买的。"""
    base = 30
    rows = []
    for mult, target in ((2, 1 / 2), (4, 1 / 2), (10, 1 / 10), (100, 1 / 10)):
        rows.append((mult, BODY_SIGMA / np.sqrt(base * mult), target))
    half_n = BODY_SIGMA / np.sqrt(base * 4) / (BODY_SIGMA / np.sqrt(base))
    tenth_n = BODY_SIGMA / np.sqrt(base * 100) / (BODY_SIGMA / np.sqrt(base))

    sec.check(
        "精度是平方根买的：误差减半需 4 倍样本，降到 1/10 需 100 倍",
        abs(half_n - 0.5) < 1e-12 and abs(tenth_n - 0.1) < 1e-12,
        f"以 n=30 为基准：n×4 → 误差 ×{half_n:.3f}；n×100 → 误差 ×{tenth_n:.3f}；"
        f"（对照：只给 2 倍样本，误差仅降到 ×{1/np.sqrt(2):.3f}）",
        tolerance="精确到 1e-12",
    )
    sec.observe(
        "这条换算很重要却被书上轻轻带过：误差范围 ∝ 1/√n，所以它是**平方根**在起作用。"
        "想把策略收益估计的不确定性压一半，样本要翻 4 倍（回测窗口长度 ×4）；"
        "想压到 1/10，要翻 100 倍。反过来读就是：从 n=30 加到 n=100，"
        f"花了 3.3 倍的数据，误差只从 ±{BODY_SIGMA/np.sqrt(30)*1.96:.3f} 降到 "
        f"±{BODY_SIGMA/np.sqrt(100)*1.96:.3f}。"
    )


# ------------------------------------------------- ③ 真实右偏数据上还成立吗


def experiment_real_data(sec: Section) -> dict:
    pop = gd.tornado_width()
    mu, sigma = float(pop.mean()), float(pop.std(ddof=1))
    rng = np.random.default_rng(7)
    M = 20_000
    ns = np.array([10, 31, 100, 300], dtype=float)
    sd_emp, err_mean = [], []
    for n in ns:
        n = int(n)
        idx = rng.integers(0, len(pop), size=(M, n))
        means = pop[idx].mean(axis=1)
        sd_emp.append(float(means.std(ddof=1)))
        err_mean.append(float(np.abs(means.mean() - mu)))
    sd_emp = np.array(sd_emp)
    err_mean = np.array(err_mean)
    se_theory = sigma / np.sqrt(ns)
    slope = num.convergence_rate(ns, sd_emp)

    sec.check(
        "真实右偏数据（偏度 3.30）上，重抽样 OK：样本均值散布仍服从 σ/√n",
        bool(np.all(np.abs(sd_emp / se_theory - 1) < 0.05)) and abs(slope + 0.5) < 0.06,
        f"总体 μ={mu:.2f}、σ={sigma:.2f}；M={M:,} 次重抽样；斜率 {slope:.3f}；"
        + "；".join(f"n={int(n)}: 实测 {e:.2f} vs σ/√n={t:.2f}"
                   for n, e, t in zip(ns, sd_emp, se_theory)),
        tolerance="相对误差 <5% 且斜率 ∈ (−0.56, −0.44)",
    )
    sec.observe(
        "这是 CLT 真正好用的地方：总体本身偏得一塌糊涂（龙卷风宽度偏度 3.30），"
        "但只要抽的是**均值**，散布照样按 σ/√n 走。"
        "注意区分两件事——散布（方差）归 CLT 管，形状（尾部）归第 4 章管；"
        "4.2 已经量过：同样是 n=31，中心部分很准，尾部还是被正态低估 70%。"
    )
    return {"pop": pop, "mu": mu, "sigma": sigma, "ns": ns, "sd_emp": sd_emp}


# ------------------------------------------------- ④ σ 换成 s，会发生什么


def experiment_sigma_unknown(sec: Section, real: dict) -> None:
    """书上例题默认知道 σ。现实中不知道，必须用样本标准差 s 顶替 —— 这一步有代价。"""
    pop = real["pop"]
    mu = real["mu"]
    rng = np.random.default_rng(11)
    M = 20_000
    z = float(stats.norm.ppf(0.975))
    rows = []
    for n in (10, 20, 31, 100):
        idx = rng.integers(0, len(pop), size=(M, n))
        draws = pop[idx]
        means = draws.mean(axis=1)
        s = draws.std(ddof=1, axis=1)
        half_z = z * s / np.sqrt(n)
        t_crit = float(stats.t.ppf(0.975, n - 1))
        half_t = t_crit * s / np.sqrt(n)
        cov_z = float(np.mean(np.abs(means - mu) <= half_z))
        cov_t = float(np.mean(np.abs(means - mu) <= half_t))
        rows.append((n, cov_z, cov_t))
    names = np.array([r[0] for r in rows])
    cz = np.array([r[1] for r in rows])
    ct = np.array([r[2] for r in rows])

    sec.check(
        "σ 未知、改用样本标准差 s 后，95% 区间的实际覆盖率就不再是 95%（指向 t 分布）",
        bool(np.all(cz < 0.95)) and cz[0] < 0.93,
        "；".join(f"n={int(n)}: z 区间覆盖 {z_:.2%}（t 区间 {t_:.2%}）"
                  for n, z_, t_ in zip(names, cz, ct)),
        tolerance="所有 n 上 z 区间覆盖 < 95%，且 n=10 时 < 93%",
    )
    sec.pitfall(
        "书上的体温题把 σ=0.73 当已知量写进公式，这在真实问题里几乎不可能——"
        "知道 σ 基本等于知道了整个总体。一旦换成样本估计 s，"
        "同一句「95% 置信」就开始打折扣：小样本上尤其明显。"
        "这正是本章后半要用 t 分布的原因，也是 5.2 的主題。"
    )
    sec.observe(
        f"这条结果把我自己的预期打掉了。我原本以为：n=10 这么小，"
        f"换 t 分布应该能把覆盖率从 {cz[0]:.1%} 拉回 95% 附近——书上一整节都在讲「小样本用 t 分布」，"
        f"读起来像是 t 能兜住这件事。实测只从 {cz[0]:.1%} 抬到 {ct[0]:.1%}，"
        f"差 {(ct[0]-cz[0])*100:.1f} 个百分点，离 95% 还差 {95-ct[0]*100:.1f} 个百分点。"
        "原因是分工不同：t 分布修的是**不知道 σ**，修不了**抽样分布本身不正态**。"
        "这里总体偏度 3.30，n=10 时 x̄ 的分布还是斜的，任何靠 ±临界值×标准误拼出来的对称区间都吃亏。"
        "⇒ 「小样本就用 t」这句话只在总体近似正态时成立，5.2 会把它钉死。"
    )
    return {"rows": rows}


# ------------------------------------------------- 图


def draw(sec: Section, ns: np.ndarray, probs: np.ndarray, real: dict) -> None:
    # 图 1：复现图 5.2（三个面板）
    fig, axes = plot.newfig(1, 3, figsize=(9.0, 2.9))
    for ax, n in zip(axes, (31, 60, 250)):
        se = BODY_SIGMA / np.sqrt(n)
        x = np.linspace(BODY_MU - 4 * se, BODY_MU + 4 * se, 400)
        y = stats.norm.pdf(x, BODY_MU, se)
        ax.plot(x, y, lw=1.2, color=plot.ACCENT)
        xs = np.linspace(BODY_MU - 4 * se, BODY_TARGET, 100)
        ax.fill_between(xs, stats.norm.pdf(xs, BODY_MU, se),
                        color=plot.ACCENT2, alpha=0.35)
        ax.set_title(f"n = {n}｜P < {BODY_TARGET}°F = "
                     f"{float(stats.norm.cdf(BODY_TARGET, BODY_MU, se)):.4f}",
                     fontsize=9)
        ax.set_xlabel("样本均值", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("密度", fontsize=9)
    fig.suptitle("复现图 5.2：n 增大，样本均值的分布整体变窄", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    sec.figure("clt-narrowing", fig,
               caption="同一个总体抽不同的 n：曲线变窄，尾部面积被赶到中心")

    # 图 2：√(n) 律 + 争取精度的代价
    fig, axes = plot.newfig(1, 2, figsize=(7.6, 3.0))
    n_grid = np.logspace(0.7, 4, 200)
    axes[0].loglog(n_grid, BODY_SIGMA / np.sqrt(n_grid), lw=1.4, color=plot.ACCENT,
                   label="标准误 σ/√n")
    axes[0].loglog(n_grid, BODY_SIGMA / n_grid, "--", lw=1.0, color=plot.MUTED,
                   label="（错误直觉）σ/n")
    axes[0].set_xlabel("样本量 n"); axes[0].set_ylabel("样本均值的标准差")
    axes[0].legend(frameon=False, fontsize=8); axes[0].set_title("越抽越准，但只是 √n", fontsize=10)

    mults = np.array([1, 2, 4, 10, 100])
    axes[1].plot(mults, 1 / np.sqrt(mults), "o-", color=plot.ACCENT2, lw=1.3, ms=4)
    axes[1].axhline(1.0, lw=0.8, color=plot.MUTED)
    for m in (4, 100):
        axes[1].annotate(f"n×{m}\n→ 误差 ×{1/np.sqrt(m):.2f}",
                         xy=(m, 1 / np.sqrt(m)), xytext=(m * 1.15, 1 / np.sqrt(m) + 0.12),
                         fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7, color=plot.MUTED))
    axes[1].set_xscale("log"); axes[1].set_xlabel("样本量翻多少倍")
    axes[1].set_ylabel("误差范围还剩几成")
    axes[1].set_ylim(0, 1.25)
    axes[1].set_title("√n 的账单：一半精度要 4 倍样本", fontsize=10)
    fig.tight_layout()
    sec.figure("sqrt-cost", fig,
               caption="左边核对 √n 律，右边是它对成本的直接含义")

    # 图 3：真实数据上的覆盖率塌陷
    rows = real["coverage"]
    xs = np.array([r[0] for r in rows])
    cz = np.array([r[1] for r in rows]) * 100
    ct = np.array([r[2] for r in rows]) * 100
    fig, ax = plot.newfig(figsize=(6.2, 3.2))
    ax.plot(xs, cz, "o-", color=plot.ACCENT, lw=1.3, ms=5, label="用 z=1.96 + 样本 s")
    ax.plot(xs, ct, "s-", color="#639922", lw=1.1, ms=4, label="用 t 临界值")
    ax.axhline(95, ls="--", lw=1.0, color=plot.MUTED, label="标称 95%")
    ax.set_xscale("log"); ax.set_xlabel("样本量 n"); ax.set_ylabel("真实覆盖率 %")
    ax.set_ylim(80, 100); ax.legend(frameon=False, fontsize=8)
    sec.figure("coverage-preview", fig,
               caption="把 σ 换成估计值 s 之后，同一句「95% 置信」在小样本上就不兑现了（→5.2）")


def run(sec: Section) -> None:
    ns, probs, _ = experiment_body_temp(sec)
    experiment_sqrt_law(sec, ns)
    experiment_cost(sec)
    real = experiment_real_data(sec)
    real["coverage"] = experiment_sigma_unknown(sec, real)["rows"]
    draw(sec, ns, probs, real)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
