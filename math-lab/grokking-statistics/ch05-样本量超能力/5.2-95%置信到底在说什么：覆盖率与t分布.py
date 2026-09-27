"""5.2 「95% 置信」到底在说什么：覆盖率、t 分布，以及 n>30 那条规则的账单

书上讲得很用心：「95% 的网能捞到鱼，不是说某一张网有 95% 的概率捞到鱼」。
这句话绝大多数人读完就忘，因为它没有被翻成可执行的东西。

本节把它翻成可执行：**覆盖率是可以被数值判决的**。
把区间公式当一台机器，反复喂新样本，数一数真值被包住的比例。
一旦这么看，本章后半那些「用 t 分布」「样本量超过 30 就行」的说法，
全都变成了可结算的账单。
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
from mathviz import Section, plot  # noqa: E402

SEC = Section(
    number="5.2",
    chapter="第 5 章 · 样本量超能力",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="「95% 置信」到底在说什么：覆盖率、t 分布与 n>30 的账单",
    claim="作者反复强调：95% 置信区间的意思是「重复做很多次实验，其中 95% 的区间会包含真值」，"
          "而不是「这个区间有 95% 的概率包含真值」——把 CI 想成捞鱼的网。"
          "另外两条：样本量 ≤30 时用 t 分布（自由度 n−1）；"
          "样本量超过 30 时 t 分布与正态分布没有显著差异，可以用正态。",
    proposition="① σ 已知时 z 区间的实际覆盖率确实是 95%（这句话可以被实验判决）；"
                "② σ 未知时改用样本标准差 s，z 区间覆盖率会系统性低于 95%，且小样本上塌陷明显；"
                "③ t 区间在正态总体下把覆盖率精确修回 95%；"
                "④ 书上「n>30 就可以用正态代替 t」有代价：n=31 上要亏约 1 个百分点的覆盖率；"
                "⑤ t 分布只修「σ 未知」，修不了「总体不正态」——真实右偏数据上它同样崩。",
    source="第 5 章「基于均值的置信区间 / 小样本与 t 分布」（网与鱼的类比、蜡烛例题图 5.8）；"
           "数据：NOAA 龙卷风宽度 3210 条、BTCUSDT 日收益",
)

CONF = 0.95
Z_CRIT = float(stats.norm.ppf(0.975))
M = 30_000
RNG_SEED = 5


# ------------------------------------------------- ① σ 已知：95% 就是 95%


def experiment_coverage_sigma_known(sec: Section) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    mu, sigma = 100.0, 15.0
    out = {}
    for n in (10, 31, 100):
        means = rng.normal(mu, sigma, size=(M, n)).mean(axis=1)
        half = Z_CRIT * sigma / np.sqrt(n)          # σ 已知 ⇒ 只有抽样误差
        cov = float(np.mean(np.abs(means - mu) <= half))
        out[n] = cov
    worst = max(abs(v - CONF) for v in out.values())
    mc_se = np.sqrt(CONF * (1 - CONF) / M)          # 蒙特卡洛自身的标准误
    sec.check(
        "σ 已知时，95% 区间的实际覆盖率 = 95%（频率学定义成立）",
        worst < 3 * mc_se,
        f"M={M:,} 次重抽样；" + "；".join(f"n={n}: {v:.2%}" for n, v in out.items())
        + f"；蒙特卡洛标准误 ±{mc_se:.2%}，最大偏离 {worst:.2%}（{worst/mc_se:.1f} 倍标准误）",
        tolerance="偏离 < 3 倍蒙特卡洛标准误",
    )
    sec.observe(
        f"「95% 的网能捞到鱼」这句话是可以查账的：真的抽 {M:,} 次样、算 {M:,} 个区间，"
        f"覆盖真值的比例就是 {out[31]:.2%}。反过来，**单个区间没有概率可谈**——"
        "某一张网要么捞到要么没捞到，真值要么在里面要么不在。"
        "这一句之所以重要，是因为回测报告里那句「策略年化 18% ± 3%（95% CI）」"
        "被误读成「真收益有 95% 概率落在 15%~21%」几乎成了行业惯例。"
    )
    return out


# ------------------------------------------------- ② σ 未知：z 区间开始漏水


def z_theoretical_coverage(n: int) -> float:
    """正态总体、σ 未知用 z=1.96 时的精确覆盖率 = P(|T_{n−1}| < 1.96)。"""
    return float(2 * stats.t.cdf(Z_CRIT, n - 1) - 1)


def experiment_sigma_unknown(sec: Section) -> dict:
    rng = np.random.default_rng(RNG_SEED + 1)
    mu, sigma = 100.0, 15.0
    ns = np.array([5, 10, 20, 31, 60, 100, 200])
    cov_sim, cov_t = [], []
    for n in ns:
        draws = rng.normal(mu, sigma, size=(M, n))
        means = draws.mean(axis=1)
        s = draws.std(ddof=1, axis=1)
        t_crit = float(stats.t.ppf(0.975, n - 1))
        cov_sim.append(float(np.mean(np.abs(means - mu) <= Z_CRIT * s / np.sqrt(n))))
        cov_t.append(float(np.mean(np.abs(means - mu) <= t_crit * s / np.sqrt(n))))
    cov_sim, cov_t = np.array(cov_sim), np.array(cov_t)
    cov_theory = np.array([z_theoretical_coverage(int(n)) for n in ns])
    mc_se = np.sqrt(CONF * (1 - CONF) / M)

    gap_theory = np.abs(cov_sim - cov_theory)
    # 单调性交给解析值（它严格单调）；模拟值只负责证实解析值——
    # 相邻 n 之间真实差距已被蒙特卡洛噪声埋掉，用 diff>0 判会误伤
    sec.check(
        "σ 未知仍用 z=1.96 ⇒ 覆盖率系统性低于 95%，而且越低 n 漏得越多",
        bool(np.all(cov_theory < CONF))
        and bool(np.all(np.diff(cov_theory) > 0))
        and bool(np.all(gap_theory < 3 * mc_se))
        and cov_sim[0] < CONF - 20 * mc_se,
        f"M={M:,} 次；" + "；".join(
            f"n={int(n)}: 模拟 {cs:.2%} / 解析 {ct:.2%}"
            for n, cs, ct in zip(ns, cov_sim, cov_theory))
        + f"；与解析值 P(|T_{{n−1}}|<1.96) 的最大差 {gap_theory.max():.2%}"
        + f"（σ_MC=±{mc_se:.2%}，故「系统性低于 + 单调」用解析值判，"
          f"小样本 n=5 的亏损 {(CONF-cov_sim[0])*100:.1f} 个百分点 = "
          f"{(CONF-cov_sim[0])/mc_se:.0f} 倍噪声）",
        tolerance="解析值全部 < 95%、随 n 单调、模拟与解析差 < 3σ_MC、n=5 亏损 > 20σ_MC",
    )

    sec.check(
        "换成 t 区间，覆盖率精确回到 95%（这才是 t 分布存在的理由）",
        bool(np.all(np.abs(cov_t - CONF) < 3 * mc_se))
        and cov_sim[0] < CONF - 5 * mc_se,
        "；".join(f"n={int(n)}: t 区间 {ct:.2%}（z 区间 {cs:.2%}）"
                  for n, ct, cs in zip(ns, cov_t, cov_sim)),
        tolerance="各 n 上都落在 95% ± 3σ_MC 内",
    )
    sec.observe(
        "t 分布不是「近似正态的另一条曲线」，它是**给 σ 的不确定性补差价**的量。"
        f"代价是区间变宽：n=5 时临界值从 1.96 涨到 {stats.t.ppf(0.975, 4):.3f}（宽 41.7%），"
        f"n=31 时 2.042（宽 4.2%），n=100 时 1.984（宽 1.2%）。"
        "不付这笔钱，换来的就是覆盖率从 95% 掉到 87.8%。"
    )
    return {"ns": ns, "cov_z": cov_sim, "cov_t": cov_t, "cov_theory": cov_theory,
            "mc_se": mc_se}


# ------------------------------------------------- ③ 「n>30 就用正态」的账单


def experiment_n30_rule(sec: Section) -> dict:
    """书上：样本量超过 30，t 与正态没有显著差异。这句话要付费。"""
    rows = []
    for n in (10, 20, 31, 50, 100, 200, 500, 1000, 5000):
        rows.append((n, z_theoretical_coverage(n), float(stats.t.ppf(0.975, n - 1))))
    ns = np.array([r[0] for r in rows], dtype=float)
    cov = np.array([r[1] for r in rows])
    tcrit = np.array([r[2] for r in rows])
    loss = CONF - cov
    i31 = list(ns).index(31.0)
    # 亏 < 0.1 个百分点要多少样本
    ok = np.where(loss < 0.001)[0]
    n_needed = int(ns[ok[0]]) if len(ok) else -1

    sec.check(
        "「n>30 就可以用正态代替 t」要付出约 1 个百分点的覆盖率",
        loss[i31] > 0.008 and loss[i31] < 0.012,
        f"n=31 时 z 区间覆盖率 {cov[i31]:.2%}（亏 {loss[i31]*100:.2f} 个百分点）；"
        f"n=100 亏 {loss[list(ns).index(100.0)]*100:.2f} 个百分点；"
        f"要亏到 0.1 个百分点以下需要 n ≥ {n_needed}",
        tolerance="n=31 上的亏损 ∈ (0.8, 1.2) 个百分点",
    )
    sec.check(
        "这笔账单调递减但收敛极慢：亏 1 个百分点 → 0.1 个百分点要再放大 16 倍样本",
        bool(np.all(np.diff(loss) < 0)) and n_needed >= 16 * 31,
        "；".join(f"n={int(n)}: 亏 {l*100:.2f} 个百分点（t 临界值 {c:.3f}）"
                  for n, l, c in list(zip(ns, loss, tcrit))[::2]),
        tolerance="单调递减且所需 n ≥ 496",
    )
    sec.pitfall(
        f"「n>30 就不用 t 了」这句话在绝大多数教科书里都有，包括这本。"
        f"实测它不是免费：n=31 上覆盖率只有 {cov[i31]:.2%}。"
        "要不要付这 1 个百分点，取决于你在干什么——算着玩的报告无所谓，"
        "但如果这条区间要去触发风控阈值（比如「亏损超过区间下界就降杠杆」），"
        "1 个百分点的漏工作 Systematically bias 到同一个方向，长期是会被放大的。"
        f"更重要的是：现代计算里查 t 临界值和查 z 一样廉价，所以这里**没有省事的理由**。"
    )
    return {"ns": ns, "cov_z": cov, "loss": loss, "tcrit": tcrit, "n_needed": n_needed}


# ------------------------------------------------- ④ 复现蜡烛例题


def experiment_candle(sec: Section) -> None:
    mean, s, n = 42.24, 3.62, 10
    conf = 0.99
    t_crit = float(stats.t.ppf(1 - (1 - conf) / 2, n - 1))
    margin = t_crit * s / np.sqrt(n)
    lo, hi = mean - margin, mean + margin
    z_margin = float(stats.norm.ppf(1 - (1 - conf) / 2)) * s / np.sqrt(n)
    sec.check(
        "复现书上蜡烛例题：99% CI = (38.520, 45.960)",
        abs(lo - 38.520) < 5e-3 and abs(hi - 45.960) < 5e-3,
        f"t 临界值（dof=9）= {t_crit:.5f}；误差范围 {margin:.4f}；"
        f"区间 ({lo:.3f}, {hi:.3f})；"
        f"若误用 z={float(stats.norm.ppf(0.995)):.5f}，区间会窄成 "
        f"({mean - z_margin:.3f}, {mean + z_margin:.3f})（宽度少 {1 - z_margin/margin:.1%}）",
        tolerance="两端与书上一致到 5e-3",
    )
    sec.observe(
        "这道题是全书最容易踩坑的一处：它同时要求 99% 置信**和**小样本，"
        "于是两个因素叠加——置信越高区间越宽、样本越小惩罚越重，"
        f"最后临界值高达 {t_crit:.3f}（对比常用的 1.96）。"
        "如果这里顺手写成 1.96 或 2.58，宣传出去的燃烧时长承诺就是偏乐观的。"
    )


# ------------------------------------------------- ⑤ t 修不了「总体不正态」


def experiment_nonnormal(sec: Section) -> dict:
    """把标准教材习题里的「总体正态」前提拿掉，看还剩下什么。"""
    pop = gd.tornado_width()
    mu = float(pop.mean())
    rng = np.random.default_rng(RNG_SEED + 2)
    ns = np.array([10, 31, 100, 300], dtype=float)
    cov_z, cov_t, cov_boot = [], [], []
    B = 1500                                        # bootstrap 次数
    for n in ns:
        n = int(n)
        idx = rng.integers(0, len(pop), size=(M // 5, n))
        draws = pop[idx]
        M2 = draws.shape[0]
        means = draws.mean(axis=1)
        s = draws.std(ddof=1, axis=1)
        t_crit = float(stats.t.ppf(0.975, n - 1))
        cov_z.append(float(np.mean(np.abs(means - mu) <= Z_CRIT * s / np.sqrt(n))))
        cov_t.append(float(np.mean(np.abs(means - mu) <= t_crit * s / np.sqrt(n))))
        # bootstrap 百分位区间：从样本自身重抽样，不假设任何分布形状。
        # 注意覆盖的判据是「真值 μ 是否落在区间里」，不是「样本均值是否落在区间里」
        # ——后者必然接近 100%，因为 bootstrap 分布本来就围着 x̄ 转。
        bi = rng.integers(0, n, size=(B, n))
        res = draws[:, bi].mean(axis=2)
        lo = np.quantile(res, 0.025, axis=1)
        hi = np.quantile(res, 0.975, axis=1)
        cov_boot.append(float(np.mean((mu >= lo) & (mu <= hi))))
    cov_z, cov_t, cov_boot = map(np.array, (cov_z, cov_t, cov_boot))

    sec.check(
        "总体严重右偏时，z 和 t 都救不回来；bootstrap 也不比 t 好（我的预期被实测打掉）",
        bool(np.all(cov_t[:3] < 0.93)) and cov_t[1] < 0.90
        and bool(np.all(cov_boot[:3] <= cov_t[:3] + 0.005))
        and cov_t[-1] < CONF,
        f"龙卷风宽度（偏度 3.30），每组 {M//5:,} 次抽样、bootstrap B={B}；" + "；".join(
            f"n={int(n)}: z {a:.1%} / t {b:.1%} / bootstrap {c:.1%}"
            for n, a, b, c in zip(ns, cov_z, cov_t, cov_boot)),
        tolerance="n≤100 上 t < 93%、n=31 上 < 90%；bootstrap 不优于 t；n=300 仍未达标",
    )
    sec.observe(
        "这条是本节我自己被打脸的地方。我原本的预期是：bootstrap 不依赖分布假设，"
        "在这种严重右偏的数据上总该比 t 强一点。实测**没有**——"
        f"n=10 时 bootstrap {cov_boot[0]:.1%} 反而略低于 t 的 {cov_t[0]:.1%}，"
        f"n=31 时 {cov_boot[1]:.1%} vs {cov_t[1]:.1%}，全程打平。"
        "原因不难懂：百分位 bootstrap 的区间宽度直接取自「这个样本自己的离散程度」，"
        "n=10 的小样本既偏又窄，bootstrap 分布照抄了这两点。"
        "⇒ 换再花哨的方法也没用。**t 分布只修 σ 未知，bootstrap 也不修分布形状**；"
        f"真想把覆盖率推回 95%，唯一有效的旋钮是把 n 拉上去（n=300 才 {cov_t[-1]:.1%}）。"
        "这句话的值钱之处在于：当你面对的是重尾数据（收益序列、延迟分布、龙卷风宽度），"
        "提高样本量是唯一确定的解法，任何区间公式的巧思都只是边际改良。"
    )
    return {"ns": ns, "cov_z": cov_z, "cov_t": cov_t, "cov_boot": cov_boot}


# ------------------------------------------------- 图


def draw(sec: Section, sigma_unknown: dict, n30: dict, nonnormal: dict) -> None:
    mc_se = sigma_unknown["mc_se"]

    # 图 1：100 张网（σ 已知，n=31）
    rng = np.random.default_rng(99)
    mu, sigma, n = 100.0, 15.0, 31
    means = rng.normal(mu, sigma, size=(100, n)).mean(axis=1)
    half = Z_CRIT * sigma / np.sqrt(n)
    miss = np.abs(means - mu) > half
    fig, ax = plot.newfig(figsize=(7.2, 3.6))
    for i, m in enumerate(means):
        ax.plot([i, i], [m - half, m + half],
                lw=1.1, color=plot.ACCENT2 if miss[i] else plot.ACCENT, alpha=0.85)
        ax.plot(i, m, "|", ms=6, color=plot.ACCENT2 if miss[i] else plot.ACCENT)
    ax.axhline(mu, lw=1.0, color="k", ls="--")
    ax.set_xlabel("第几次实验")
    ax.set_ylabel("样本均值 ± 误差范围")
    ax.set_title(f"100 个 95% 置信区间：{int(miss.sum())} 张网没捞到鱼（σ 已知）",
                 fontsize=11)
    sec.figure("nets", fig,
               caption="「95%」说的是这类网的整体命中率，不是某一位农民的单次运气")

    # 图 2：σ 未知后 z 漏水、t 补价
    ns = sigma_unknown["ns"]
    fig, axes = plot.newfig(1, 2, figsize=(7.8, 3.1))
    axes[0].semilogx(ns, sigma_unknown["cov_z"] * 100, "o-", color=plot.ACCENT,
                     lw=1.3, ms=4, label="z 区间（误用 σ≈s）")
    axes[0].semilogx(ns, sigma_unknown["cov_t"] * 100, "s-", color="#639922",
                     lw=1.1, ms=4, label="t 区间")
    axes[0].axhline(95, ls="--", lw=1.0, color=plot.MUTED)
    axes[0].fill_between([ns[0], ns[-1]], (95 - 3 * mc_se * 100), (95 + 3 * mc_se * 100),
                         color=plot.MUTED, alpha=0.15, label="蒙特卡洛噪声带")
    axes[0].set_xlabel("样本量 n"); axes[0].set_ylabel("真实覆盖率 %")
    axes[0].set_ylim(85, 100); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_title("正态总体：t 把覆盖率修回 95%", fontsize=10)

    axes[1].semilogx(n30["ns"], n30["loss"] * 100, "o-", color=plot.ACCENT2, lw=1.3, ms=4)
    axes[1].axhline(0.1, ls=":", lw=1.0, color=plot.MUTED)
    axes[1].annotate(f"亏 0.1 个百分点\n要 n ≥ {n30['n_needed']}",
                     xy=(n30["n_needed"], 0.1), xytext=(60, 0.35), fontsize=8,
                     arrowprops=dict(arrowstyle="->", lw=0.7, color=plot.MUTED))
    axes[1].set_xlabel("样本量 n"); axes[1].set_ylabel("丢掉的覆盖率（百分点）")
    axes[1].set_title("用正态代替 t 的账单（收敛很慢）", fontsize=10)
    fig.tight_layout()
    sec.figure("z-vs-t", fig,
               caption="左边解释 t 为什么存在，右边给「n>30 就用正态」标价")

    # 图 3：前提被破坏时
    fig, ax = plot.newfig(figsize=(6.4, 3.2))
    ax.semilogx(nonnormal["ns"], nonnormal["cov_z"] * 100, "o-", color=plot.ACCENT,
                lw=1.2, ms=4, label="z 区间")
    ax.semilogx(nonnormal["ns"], nonnormal["cov_t"] * 100, "s-", color="#639922",
                lw=1.1, ms=4, label="t 区间")
    ax.semilogx(nonnormal["ns"], nonnormal["cov_boot"] * 100, "^--", color=plot.ACCENT2,
                lw=1.1, ms=4, label="bootstrap 百分位")
    ax.axhline(95, ls="--", lw=1.0, color=plot.MUTED, label="标称 95%")
    ax.set_xlabel("样本量 n"); ax.set_ylabel("真实覆盖率 %")
    ax.set_ylim(60, 100); ax.legend(frameon=False, fontsize=8, loc="lower right")
    sec.figure("nonnormal", fig,
               caption="总体右偏（偏度 3.30）：t 与 bootstrap 都补不上分布形状这笔账")


def run(sec: Section) -> None:
    experiment_coverage_sigma_known(sec)
    sigma_unknown = experiment_sigma_unknown(sec)
    n30 = experiment_n30_rule(sec)
    experiment_candle(sec)
    nonnormal = experiment_nonnormal(sec)
    draw(sec, sigma_unknown, n30, nonnormal)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
