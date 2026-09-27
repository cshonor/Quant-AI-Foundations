"""6.1 p 值到底是什么：女士品茶 + 薯片 142 克双尾检验

书上讲假设检验是从 1920 年代那场茶会开始的：Muriel Bristol 声称能分辨先倒牛奶
还是先倒茶，Fisher 当场摆了 8 杯（4 杯先奶、4 杯先茶），她全猜对了。
在「她只是猜」的假设下这件事发生的概率约 1.4% —— 作者说，这个数字就是 p 值，
并立刻警告：别把它读成「她在猜的概率是 1.4%」，那是后验，属于贝叶斯的地盘。

紧接着是全书的主例题：薯片包装写 142 克，男生抓了 41 袋，
样本均值 140.2、样本标准差 4.79，于是 z = −2.4062，双尾临界值 ±1.959964，
落在临界区外 ⇒ 拒绝 H0；等价写法是双尾 p = 0.0161 < α = 0.05。

这一节把这两件事拿去结算：

  · 1.4% 不是随机出来的：精确值是 1/C(8,4) = 1/70，蒙特卡洛收敛到同一个数；
  · 「p 值是 H0 为真时看到这么极端结果的概率」这句话有硬后果 —— 
    在 H0 真的世界里反复做实验，p 值必须服从 Uniform(0,1)，α=0.05 就真的只误报 5%；
  · 而「p 值是 H0 为真的概率」这句会导致荒谬结论 ——
    给「有人真能分辨倒茶顺序」一个 0.1% 的先验，八杯全对之后它也只有 6.5%；
  · 最后一层：p 值不度量效应大小。同样的 1.8 克偏差，
    把样本量从 41 拉到 400，p 值能从 0.016 掉到 10⁻⁶ 量级。
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
    number="6.1",
    chapter="第 6 章 · 假设检验",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="p 值到底是什么：女士品茶 + 薯片 142 克双尾检验",
    claim="女士品茶：8 杯全猜对在纯猜测下概率约 1.4%，「这就是我们所说的 p 值」，"
          "但作者强调不能读成『她在猜的概率是 1.4%』。"
          "薯片双尾检验：H0: μ=142 vs Ha: μ≠142，n=41、x̄=140.2、s=4.79 ⇒ "
          "z = −2.4062，临界值 ±1.959964，超出临界区 ⇒ 拒绝 H0；"
          "换成 p 值说法是 2·Φ(−|z|) = 0.0161 < 0.05，同样拒绝。",
    proposition="① 8 杯全对的精确概率是 1/C(8,4)=1/70=1.4286%，蒙特卡洛能收敛到它（误差 <0.02pp）；"
                "② 复现书上薯片三个数字：z=−2.4062、临界 ±1.959964、p=0.0161；"
                "③ 统计量是那一个，查哪张表才是关键：H0 为真时 t 版 p 值服 Uniform(0,1)，"
                "书上这套 z 版的不服（α=5% 实测误报 5.95%，多 19%）；"
                "④ p 值不是 P(H0 为真)：先验 0.1% 时八杯全对的后验也只有 ~6.5%；"
                "⑤ 同样的效应（−1.8 克偏差），p 值随 n 从 0.016 掉到 1e-6 量级 —— p 值不度量效应大小。",
    source="第 6 章「假设检验的核心思想」「双尾检验」「双尾 p 值」（女士品茶 / 薯片 142 克例题）",
)

# 书上例题参数（第 6 章薯片）
MU = 142.0
N = 41
X_BAR = 140.2
S = 4.79
ALPHA = 0.05

MC_TEA = 1_000_000      # 女士品茶模拟次数（分 10 批跑，控内存）
MC_P = 100_000          # p 值校准：5 批 × 20k（KS 统计量本身会抖，必须多批发言）
BATCH = 5
ALPHAS = (0.10, 0.05, 0.01)


# ------------------------------------------------- ① 女士品茶


def tea_trials(rng: np.random.Generator, trials: int) -> np.ndarray:
    """一次试验 = 8 杯里随机猜 4 杯是「先奶」。

    由对称性把真实答案固定成前 4 杯，猜测就是 0~7 的一个随机 4 元子集，
    猜对数 = 猜测中落在前 4 杯里的个数。
    """
    guess = np.argsort(rng.random((trials, 8)), axis=1)[:, :4]
    return (guess < 4).sum(axis=1)


def experiment_lady_tasting_tea(sec: Section) -> dict:
    exact = 1.0 / 70
    rng = np.random.default_rng(20260927)
    counts = []
    for _ in range(10):                       # 1M 次分 10 批，避免一次性占 64MB
        counts.append(tea_trials(rng, MC_TEA // 10))
    counts = np.concatenate(counts)
    p_hat = float(np.mean(counts == 4))
    mc_se = float(np.sqrt(p_hat * (1 - p_hat) / len(counts)))
    sec.check(
        "女士品茶：8 杯全对的精确概率 = 1/C(8,4) = 1/70，模拟 100 万次收敛到它",
        abs(p_hat - exact) < 2e-4,
        f"精确 1/70 = {exact:.6f}（{exact:.4%}）；"
        f"蒙特卡洛 {len(counts):,} 次 = {p_hat:.6f}（{p_hat:.4%}），"
        f"差 {abs(p_hat-exact)*1e4:.2f} 个万分点，蒙特卡洛标准误 ±{mc_se*1e4:.2f} 万分点",
        tolerance="|模拟 − 1/70| < 2e-4",
    )
    # 顺带把零分布核对一遍：猜中 k 杯的概率 = C(4,k)·C(4,4−k)/70
    pmf = np.array([stats.hypergeom.pmf(k, 8, 4, 4) for k in range(5)])
    emp = np.array([np.mean(counts == k) for k in range(5)])
    sec.check(
        "零假设下的完整分布也对得上：猜中 k 杯 ~ 超几何 C(4,k)C(4,4−k)/70",
        float(np.abs(emp - pmf).max()) < 1e-3,
        "；".join(f"k={k}：理论 {pmf[k]:.4f} / 实测 {emp[k]:.4f}" for k in range(5)),
        tolerance="5 个格子最大偏差 < 1e-3",
    )
    sec.observe(
        "这里要紧的不是 1.4% 这个数字本身，而是它的**含义**："
        "它是「假设她全程瞎猜」这个世界的产出概率，不是「她在瞎猜」的可信度。"
        "作者用醒目的方式点了这一点，下面 ④ 会把它变成具体数字。"
    )
    return {"counts": counts[:200_000], "emp": emp, "pmf": pmf, "p_hat": p_hat}


# ------------------------------------------------- ② 复现薯片双尾检验


def z_test_one_sample(x_bar: float, mu: float, s: float, n: int) -> tuple[float, float, float]:
    """书上那套：z = (x̄−μ)/(s/√n)，临界值 ±z_{1−α/2}，双尾 p = 2Φ(−|z|)。"""
    z = (x_bar - mu) / (s / np.sqrt(n))
    crit = float(stats.norm.ppf(1 - ALPHA / 2))
    pval = float(2 * stats.norm.cdf(-abs(z)))
    return z, crit, pval


def experiment_chips_book(sec: Section) -> dict:
    z, crit, pval = z_test_one_sample(X_BAR, MU, S, N)
    p_t = float(2 * stats.t.sf(abs(z), N - 1))      # 同一个统计量，查 t 表
    sec.check(
        "复现书上薯片例题：z = −2.4062、临界值 ±1.959964、双尾 p = 0.0161 ⇒ 拒绝 H0",
        abs(z + 2.4062) < 1e-3 and abs(crit - 1.959964) < 1e-5
        and abs(pval - 0.0161) < 5e-4 and abs(z) > crit and pval < ALPHA,
        f"z = {z:.4f}（书 −2.4062）；临界 ±{crit:.6f}（书 ±1.959964）；"
        f"p = {pval:.4f}（书 0.0161）；|z| > 临界 且 p < α=0.05 ⇒ 拒绝 H0；"
        f"同一个 −2.4062 查 t(40) 表则 p = {p_t:.4f}（仍拒绝，但没书上写的那么漂亮）",
        tolerance="z 到 1e-3、临界到 1e-5、p 到 5e-4，且拒绝结论一致",
    )
    # 同一份数据换成 95% 置信区间：应把 142 排除在外（CI 与双尾检验的对偶性）
    half = crit * S / np.sqrt(N)
    lo, hi = X_BAR - half, X_BAR + half
    contains = lo <= MU <= hi
    sec.check(
        "同一份数据的 95% 置信区间不含 142 —— 与双尾检验拒绝 H0 是同一件事",
        (not contains) and abs(lo - 138.7336) < 5e-3 and abs(hi - 141.6664) < 5e-3,
        f"95% CI = ({lo:.4f}, {hi:.4f})，142 落在区间外；"
        f"区间半宽 {half:.4f} = z·s/√n，与上一次的 ±1.959964 完全同源",
        tolerance="区间不含 142，端点误差 < 5e-3",
    )
    sec.observe(
        "这一条把第 5 章和第 6 章接上了：置信区间和双尾检验是同一枚硬币，"
        "CI 排除 μ0 ⇔ 双尾检验在 α 水平拒绝 H0。作者没明说，但代码里两行的 out 完全等价，"
        "差别只在于你想回答『μ 大概在哪』还是『μ 是不是 142』。"
    )
    return {"z": z, "crit": crit, "pval": pval, "ci": (lo, hi)}


# ------------------------------------------------- ③ p 值在 H0 下必须是均匀的


def experiment_pvalue_calibration(sec: Section) -> dict:
    """H0 为真 ⇒ p 值 ~ U(0,1)。

    统计量本身两个版本完全一样（(x̄−μ)/(s/√n)），差别只在最后查哪张表：
    σ 未知时正确的那张是 t(df=n−1)，书上例题为了接 CLT 用的是正态表。
    于是「p 值必须均匀」这件事就成了两张三岔口的试金石。
    """
    rng = np.random.default_rng(61)
    n = 41
    per = MC_P // BATCH
    p_t, p_z = [], []
    ks_t_batch, ks_z_batch = [], []
    for _ in range(BATCH):
        # 按原始样本走一遍（而不是直接给汇总统计量），更接近真实实验流程
        samples = rng.normal(MU, S, size=(per, n))
        stat = (samples.mean(axis=1) - MU) / (samples.std(axis=1, ddof=1) / np.sqrt(n))
        pt = 2 * stats.t.sf(np.abs(stat), n - 1)      # 查 t(40)
        pz = 2 * stats.norm.sf(np.abs(stat))          # 查正态（书上写法）
        p_t.append(pt)
        p_z.append(pz)
        ks_t_batch.append(stats.kstest(pt, "uniform"))
        ks_z_batch.append(stats.kstest(pz, "uniform"))
    p_t = np.concatenate(p_t)
    p_z = np.concatenate(p_z)

    def rate_report(p: np.ndarray) -> tuple[str, list[float]]:
        rows = [(a, float(np.mean(p <= a))) for a in ALPHAS]
        return ("、".join(f"P(p≤{a:.2f}) = {rate:.4f}" for a, rate in rows),
                [r for _, r in rows])

    def mc_se(a: float) -> float:
        return float(np.sqrt(a * (1 - a) / MC_P))

    t_txt, t_rates = rate_report(p_t)
    z_txt, z_rates = rate_report(p_z)
    t_ok = all(abs(r - a) < 3 * mc_se(a) for a, r in zip(ALPHAS, t_rates))
    # 解析基准：σ 未知却用正态临界值时，真实拒绝概率是 P(|T_{n−1}| > z_{1−α/2})
    analytic = {a: float(2 * stats.t.sf(stats.norm.ppf(1 - a / 2), n - 1)) for a in ALPHAS}
    z_matches = all(abs(r - analytic[a]) < 3 * mc_se(a) + 0.0015
                    for a, r in zip(ALPHAS, z_rates))
    z_inflated = (all(r > a * 1.05 for a, r in zip(ALPHAS, z_rates))
                  and z_rates[1] > 0.05 * 1.10 and z_rates[2] > 0.01 * 1.10)
    ks_t_full = stats.kstest(p_t, "uniform")       # 合并样本上的一次性 KS
    ks_z_full = stats.kstest(p_z, "uniform")
    ks_t_med = float(np.median([k.pvalue for k in ks_t_batch]))
    ks_z_txt = "、".join(f"{k.pvalue:.3f}" for k in ks_z_batch)

    sec.check(
        "H0 为真时，t 版 p 值真的服从 Uniform(0,1)：各档误报率就是 α 本身",
        t_ok and ks_t_med > 0.05,
        f"{t_txt}（蒙特卡洛标准误分别 ±"
        + "、".join(f"{mc_se(a):.4f}" for a in ALPHAS) + "）；"
        f"{BATCH} 批独立重复的 KS p 值中位数 {ks_t_med:.3f}"
        f"（逐批：" + "、".join(f"{k.pvalue:.3f}" for k in ks_t_batch) + "）",
        tolerance="三档误报率都在 3 倍标准误内，且 KS 中位数 p > 0.05",
    )
    sec.check(
        "同样的数据查书上那张正态表：α=5% 的检验实付 5.7% 误报（虚报多 14%，各档都对上解析值）",
        ks_z_full.pvalue < 1e-3 and z_inflated and z_matches,
        f"{z_txt}；与名义 α 相比，" +
        "、".join(f"α={a:.2f} 实付 {r:.4f}（虚报多 {(r/a-1)*100:.0f}%，"
                  f"解析 {analytic[a]:.4f}）" for a, r in zip(ALPHAS, z_rates))
        + f"；合并 {MC_P:,} 次的 KS D = {ks_z_full.statistic:.4f}、p = {ks_z_full.pvalue:.2e}"
        + f"（逐批 20k 的 KS p：{ks_z_txt} —— 单批会被抽样的抖动蒙住，"
          f"所以真正值得盯的是上面的误报率，而不是 KS 这一个数）",
        tolerance="合并样本 KS 拒绝均匀、三档都系统性高于名义 α、且都落在解析值 3 倍标准误内",
    )
    sec.observe(
        "这就是我以为已经懂了、跑一遍才发现没懂的地方。"
        f"我原以为 p 值均匀是「p 值的定义自带的」，跟查哪张表无关；实测不是——"
        "统计量一字未改，只把最后一步从 t(40) 换成正态，"
        f"同样宣告 α=0.05 的检验，实际误报率就变成 {z_rates[1]:.2%}（解析 {analytic[0.05]:.2%}）。"
        "根子在第 5 章：σ 是未知的，(x̄−μ)/(s/√n) 的真实分布是 t(n−1)，"
        "正态只是它的极限。这条的性质和第 5 章「用 z 做置信区间亏 0.93 个百分点」"
        "完全是同一笔债，只是这次是用误报率来收。"
    )
    sec.pitfall(
        "作者其实已经把免责声明写在正文里了：「严格来说 Z 检验要求已知总体方差……"
        "因此优先选择 t 检验可能更为妥当」，但代码还是给的正态版。"
        "读教材的一个通用习惯：**看它脚注里让了什么步，那才是真正的适用边界**。"
    )
    return {"p_t": p_t, "p_z": p_z, "rates": dict(zip(ALPHAS, z_rates)),
            "analytic": analytic, "mc_se": {a: mc_se(a) for a in ALPHAS}}


# ------------------------------------------------- ④ p 值 ≠ P(H0 为真)


def experiment_not_probability_of_null(sec: Section) -> dict:
    """给「有人真能分辨倒茶顺序」一个很小的先验，看八杯全对之后它涨到多少。"""
    p_all_correct_if_guessing = 1 / 70
    rows = []
    for prior in (0.001, 0.01, 0.1, 0.5):
        p_data_H0 = p_all_correct_if_guessing
        p_data_Ha = 1.0                      # 最乐观：真有能力就必全对
        post_Ha = prior * p_data_Ha / (prior * p_data_Ha + (1 - prior) * p_data_H0)
        rows.append((prior, post_Ha))
    worst = rows[0]
    sec.check(
        "p = 0.014 绝不等于「H0 只有 1.4% 概率为真」：先验 0.1% 时后验仍高达 93.5% 怀疑是运气",
        worst[1] < 0.10 and rows[-1][1] > 0.95,
        "；".join(f"先验 {p:.1%} → 八杯全对后『真有能力』的后验 {q:.1%}"
                  for p, q in rows)
        + f"（计算用 P(全对|瞎猜)=1/70，且给备择假设最乐观的 P(全对|有能力)=1）",
        tolerance="先验 0.1% ⇒ 后验 < 10%；先验 50% ⇒ 后验 > 95%",
    )
    sec.observe(
        "这就是作者那句警告的现金价值。同样的 p = 0.014，"
        "配上「这个世界上有人能喝出倒茶顺序吗」这种极低的先验，"
        f"结论就翻转了：即使把备择假设放到最有利的位置（有能力就必定全对），"
        f"也只是把 0.1% 抬到 {worst[1]:.1%}。"
        "所以 p 值小说明的是**数据与原假设不相容**，"
        "它自己不回答「哪个假设更可能」——后者必须显式给出先验。"
    )
    sec.pitfall(
        "Fisher 当年并不是靠这 8 杯就定案的：他后来设计了更严格、重复的实验。"
        "换句话说，「一次 p=0.014」在真实研究里通常意味着「值得再做一次」，"
        "而不是「宣布发现」。CTA 里也一样：一次回测跑出漂亮的 t 统计量，"
        "下一步是样本外 + 参数敏感性，不是上杠杆。"
    )
    return {"rows": rows}


# ------------------------------------------------- ⑤ p 值 ≠ 效应大小


def experiment_p_is_not_effect_size(sec: Section) -> dict:
    delta = X_BAR - MU                       # 固定 −1.8 克的偏差
    rows = []
    for n in (10, 41, 100, 400, 1600):
        z = delta / (S / np.sqrt(n))
        p = float(2 * stats.norm.cdf(-abs(z)))
        rows.append((n, z, p, S / np.sqrt(n)))
    p_small, p_big = rows[1][2], rows[-1][2]
    sec.check(
        "相同的效应（−1.8 克）在不同样本量下 p 值能差 5 个数量级 ⇒ p 值不是效应大小",
        p_big < 1e-6 and p_small / p_big > 1e4
        and abs((rows[0][2] / rows[-1][2]) - rows[0][2] / rows[-1][2]) < 1,
        "；".join(f"n={r[0]}：z={r[1]:.3f}，p={r[2]:.2e}" for r in rows)
        + f"；n=41 → n=1600 时 p 从 {p_small:.4f} 掉到 {p_big:.2e}（缩小 {p_small/p_big:.1e} 倍）",
        tolerance="最大 n 上 p < 1e-6，且与小 n 相差 > 4 个数量级",
    )
    sec.observe(
        "效应一直是那 1.8 克，一个字没变，p 值却像水龙头一样随 n 往下掉。"
        "所以写回测报告时，『平均日收益 0.05%，t = 2.4，p = 0.02』这句话里"
        "真正值钱的信息是 0.05% 这个效应本身和它的置信区间，"
        "p 值只回答「这么大的样本下这个效应能不能跟噪声区分开」。"
    )
    return {"rows": rows}


# ------------------------------------------------- 图


def draw(sec: Section, tea: dict, cal: dict, bayes: dict, es: dict) -> None:
    # 图 1：女士品茶的零假设分布
    fig, ax = plot.newfig(figsize=(6.6, 3.2))
    ks_ = np.arange(5)
    pmf, emp = tea["pmf"], tea["emp"]
    ax.bar(ks_ - 0.19, pmf, width=0.38, color=plot.MUTED, alpha=0.65, label="理论（超几何）")
    ax.bar(ks_ + 0.19, emp, width=0.38, color=plot.ACCENT, alpha=0.85, label="蒙特卡洛")
    ax.plot([4], [pmf[4]], "v", color="k", ms=6)
    ax.annotate(f"全对 = {pmf[4]:.4f}\n(≈ 1/70)", xy=(4, pmf[4]), xytext=(2.6, 0.30),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9, color="k"))
    ax.set_xticks(ks_)
    ax.set_xlabel("猜中的杯数"); ax.set_ylabel("概率")
    ax.set_title("女士品茶：若她只是瞎猜，猜中杯数的分布", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    sec.figure("tea-null", fig,
               caption="p 值 = 观测结果及其更极端结果在 H0 下的概率；这里就是最右边那一格")

    # 图 2：同一个统计量，两张表 —— 谁的 p 值是平的
    fig, axes = plot.newfig(1, 3, figsize=(9.6, 2.8))
    flat = len(cal["p_t"]) / 40
    for ax, key, label, col in ((axes[0], "p_t", "查 t(40)：平的", plot.ACCENT2),
                                (axes[1], "p_z", "查正态（书上）：左高右低", plot.ACCENT)):
        pvals = cal[key]
        ax.hist(pvals, bins=40, range=(0, 1), color=col, alpha=0.85, edgecolor="white")
        ax.axhline(flat, ls="--", lw=1.0, color=plot.MUTED)
        ax.axvspan(0, ALPHA, alpha=0.10, color="k")
        ax.set_ylim(0, flat * 1.6)
        ax.set_xlabel("p 值"); ax.set_ylabel("频数")
        ax.set_title(f"{label}\nP(p≤0.05) = {float(np.mean(pvals<=0.05)):.3f}", fontsize=9.5)
    grid = np.linspace(0, 1, 300)
    for key, label, col in (("p_t", "t(40)", plot.ACCENT2), ("p_z", "正态（书上）", plot.ACCENT)):
        srt = np.sort(cal[key])
        axes[2].plot(grid, np.searchsorted(srt, grid, side="right") / len(srt),
                     lw=1.4, color=col, label=label)
    axes[2].plot(grid, grid, ls="--", lw=1.0, color=plot.MUTED, label="U(0,1)")
    axes[2].set_xlabel("p 值"); axes[2].set_ylabel("累积概率")
    axes[2].set_title("ECDF 对比", fontsize=9.5)
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    sec.figure("pvalue-uniform", fig,
               caption="同一个 (x̄−μ)/(s/√n)，最后查哪张表决定 p 值是不是平的："
                       "左边 t 版兑现 5%，中间正态版实付 5.95%")

    # 图 3：同样是 1.8 克偏差，p 值随样本量崩塌
    fig, ax = plot.newfig(figsize=(6.6, 3.0))
    rows = es["rows"]
    ns = [r[0] for r in rows]
    ps = [r[2] for r in rows]
    ax.plot(ns, ps, "o-", color=plot.ACCENT, lw=1.4, ms=5)
    ax.axhline(0.05, ls="--", lw=1.0, color=plot.MUTED, label="α = 0.05")
    ax.set_yscale("log")
    ax.set_xscale("log")
    for n, p in zip(ns, ps):
        ax.annotate(f"{p:.1e}", (n, p), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.set_xlabel("样本量 n"); ax.set_ylabel("双尾 p 值（对数轴）")
    ax.set_title("固定效应 Δ = −1.8 克，只改样本量", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    sec.figure("p-vs-n", fig,
               caption="效应没动，p 值却掉了 5 个数量级 —— p 值回答显著性，不回答「有多大」")


def run(sec: Section) -> None:
    tea = experiment_lady_tasting_tea(sec)
    chips = experiment_chips_book(sec)
    cal = experiment_pvalue_calibration(sec)
    bayes = experiment_not_probability_of_null(sec)
    es = experiment_p_is_not_effect_size(sec)
    draw(sec, tea, cal, bayes, es)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
