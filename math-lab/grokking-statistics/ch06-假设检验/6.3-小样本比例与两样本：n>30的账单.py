"""6.3 小样本、比例值与两个样本：n > 30 这条判据的账单

第 6 章后半段连着甩出三道老师和练习题，每一道都复用了前面的框架，只是换的分布：

  1. 某校标准化考试：H0: μ = 100，抽 21 人得 x̄ = 110、s = 30，α = 0.01，问是否高于平均
     ⇒ 样本量 21（≤ 30）用 t(20)，得 t = 1.5275、p = 0.0711 ⇒ 不拒绝 H0；
  2. 社交媒体比例：H0: p = 0.65，调查 45 人得 p̂ = 0.80，α = 0.05 右尾
     ⇒ 先检查 np ≥ 5 且 n(1−p) ≥ 5，然后 z = 2.1100、p = 0.0174 ⇒ 拒绝；
  3. 两个独立班级：A 班 x̄₁=83、s₁=11.2、n₁=37，B 班 x̄₂=75、s₂=10.9、n₂=35，
     Ha: μ₁ > μ₂ ⇒ z = 3.0713、p = 0.0011 ⇒ 拒绝。

这一段最值钱的不是那三个数字，而是作者顺手写下的两条免责声明：

  · 「n > 30 才能用正态」这套说法，这一次是我自己要付钱的 —— n=21、α=0.01 上，
    z 版检验的实际误报率是 ~1.55%，比名义值虚报 55%（呼应 6.1 那个 n=41 上的 5.7%）；
  · 比例检验这里作者偷偷换了个更好的公式：标准误用的是 H0 下的 p 而不是样本里的 p̂，
    所以它的表现比第 5 章那个 Wald 区间稳得多（这是全书没有点破的一处不一致）；
  · 独立样本那一行注释「变异性 featur 作者自己也说应该用 t」——σ₁、σ₂ 不等时，
    书上的 z 版在 n₁≠n₂ 时会系统性虚报，Welch 版本才是能用的。
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
    number="6.3",
    chapter="第 6 章 · 假设检验",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="小样本、比例值与两个样本：n > 30 这条判据的账单",
    claim="换三种场景、不换框架："
          "① 标准化考试（n=21、x̄=110、s=30、α=0.01，右尾）用 t(20) 得 p=0.0711，不拒绝；"
          "② 社交媒体比例（p₀=0.65、n=45、p̂=0.80）先检查 np≥5 与 n(1−p)≥5，得 z=2.1100、p=0.0174，拒绝；"
          "③ 两个独立班级（83/11.2/37 与 75/10.9/35，Ha: μ₁>μ₂）得 z=3.0713、p=0.0011，拒绝。"
          "作者同时给出两条忠告：n≤30 就用 t；并且「两个样本方差是否不同」这件事本身就是不确定性来源。",
    proposition="① 三个 p 值全部复现（0.0711 / 0.0174 / 0.0011）；"
                "② n>30 不是免费的：n=21、α=0.01 时 z 版实付误报 ~1.55%，比名义虚报 55%；"
                "③ 这节的比例检验其实偷偷换成了 score 版（SE 用 p₀），比第 5 章的 Wald 版稳得多；"
                "④ 两样本情景下 σ 不等 + n 不等时，书上的 z 版会系统性虚报，Welch 版不会。",
    source="第 6 章「小样本的假设检验」「基于比例值的假设检验」「独立样本」三道例题",
)

MC = 60_000


# ------------------------------------------------- ① 复现三道题


def experiment_reproduce_exercises(sec: Section) -> dict:
    # 题 1：单样本 t 检验（右尾，α = 0.01）
    mu, n1, xb1, s1, a1 = 100.0, 21, 110.0, 30.0, 0.01
    t1 = (xb1 - mu) / (s1 / np.sqrt(n1))
    p1 = float(stats.t.sf(t1, n1 - 1))
    p1_z = float(stats.norm.sf(t1))            # 同一个统计量查正态表
    # 题 2：比例（右尾，α = 0.05）
    p0, n2, ph2, a2 = 0.65, 45, 0.80, 0.05
    z2 = (ph2 - p0) / np.sqrt(p0 * (1 - p0) / n2)
    p2 = float(stats.norm.sf(z2))
    rule_ok = n2 * p0 >= 5 and n2 * (1 - p0) >= 5
    # 题 3：两个独立样本（右尾，α = 0.05）
    xb_a, sa, na = 83.0, 11.2, 37
    xb_b, sb, nb = 75.0, 10.9, 35
    se2 = np.sqrt(sa ** 2 / na + sb ** 2 / nb)
    z3 = (xb_a - xb_b) / se2
    p3 = float(stats.norm.sf(z3))
    p3_welch = float(stats.t.sf(z3, welch_df(sa, na, sb, nb)))

    sec.check(
        "复现书上三道题：考试 t=1.5275/p=0.0711（不拒绝）、比例 z=2.1100/p=0.0174、"
        "两班 z=3.0713/p=0.0011（都拒绝）",
        abs(t1 - 1.5275) < 1e-3 and abs(p1 - 0.0711) < 5e-4 and p1 > a1
        and rule_ok and abs(z2 - 2.1100) < 1e-3 and abs(p2 - 0.0174) < 5e-4 and p2 < a2
        and abs(z3 - 3.0713) < 1e-3 and abs(p3 - 0.0011) < 5e-4 and p3 < 0.05,
        f"① t = {t1:.4f}（书 1.5275）、p = {p1:.4f}（书 0.0711）> α=0.01 ⇒ 不拒绝；"
        f"② z = {z2:.4f}、p = {p2:.4f}（书 0.0174）< 0.05 ⇒ 拒绝，"
        f"且 np={n2*p0:.2f}、n(1−p)={n2*(1-p0):.2f} 都 ≥5；"
        f"③ z = {z3:.4f}（书 3.0713）、p = {p3:.4f}（书 0.0011）⇒ 拒绝；"
        f"同一份数据改 Welch(df={welch_df(sa, na, sb, nb):.1f}) 表则 p = {p3_welch:.5f}",
        tolerance="三个统计量到 1e-3、三个 p 值到 5e-4，拒绝结论与书上一致",
    )
    sec.observe(
        f"题 ① 用的是 α=0.01 而不是 0.05，这不是随手写的："
        f"越是「我要负责任地宣布一个发现」的场合，阈值压得越低，"
        f"而下面会看到，压得越低，用错分布付的钱比例反而越贵。"
        f"同一个 t=1.5275 查正态表是 p={p1_z:.4f}（三个版本都一样查出来的错觉），"
        f"正确答案是 {p1:.4f} —— 都大于 0.01，所以这题的结论不动，只是报告的 p 值差了 11%。"
    )
    return {"t1": t1, "p1": p1, "p1_z": p1_z, "z2": z2, "p2": p2, "z3": z3,
            "p3": p3, "p3_welch": p3_welch}


def welch_df(s1: float, n1: int, s2: float, n2: int) -> float:
    """Welch–Satterthwaite 自由度。"""
    v1, v2 = s1 ** 2 / n1, s2 ** 2 / n2
    return (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))


# ------------------------------------------------- ② n>30 的账单（方差未知的真正成本）


def z_bill_table() -> list[tuple[int, float, float]]:
    """不同自由度下，把正态临界值用在 t 分布上的真实误报率。"""
    rows = []
    for df in (5, 10, 20, 40, 100, 400, 10_000):
        for alpha in (0.05, 0.01):
            zc = float(stats.norm.ppf(1 - alpha))
            rows.append((df, alpha, float(stats.t.sf(zc, df))))
    return rows


def experiment_small_n_bill(sec: Section) -> dict:
    rows = z_bill_table()
    n21_a01 = next(r for r in rows if r[0] == 20 and r[1] == 0.01)
    n41_a05 = next(r for r in rows if r[0] == 40 and r[1] == 0.05)
    n400_a05 = next(r for r in rows if r[0] == 400 and r[1] == 0.05)
    n20_a05 = next(r for r in rows if r[0] == 20 and r[1] == 0.05)
    rel20_01 = n21_a01[2] / 0.01 - 1
    rel20_05 = n20_a05[2] / 0.05 - 1

    sec.check(
        "σ 未知却用正态临界值：α=0.01、n=21 时实付误报 ~1.55%，虚报 55%（书上这题正好在这个档位）",
        n21_a01[2] > 0.014 and rel20_01 > 2 * rel20_05,
        f"解析值 P(T_df > z_α)：df=20 且 α=0.01 ⇒ {n21_a01[2]:.4f}（虚报 {rel20_01*100:.0f}%）；"
        f"同一 df 换 α=0.05 ⇒ {n20_a05[2]:.4f}（虚报 {rel20_05*100:.0f}%）—— "
        f"阈值压严十倍，相对误差翻 {rel20_01/rel20_05:.1f} 倍；"
        f"对照 df=40、α=0.05 ⇒ {n41_a05[2]:.4f}（虚报 {(n41_a05[2]/0.05-1)*100:.0f}%）、"
        f"df=400、α=0.05 ⇒ {n400_a05[2]:.4f}（虚报 {(n400_a05[2]/0.05-1)*100:.0f}%）",
        tolerance="df=20、α=0.01 的实付 > 1.4%，且相对虚报是同 df 下 α=0.05 的两倍以上",
    )
    sec.check(
        "同一笔债会随 n 收敛掉：df 从 20 涨到 400，正态近似的相对误差落一个数量级",
        (n21_a01[2] / 0.01 - 1) > 5 * (n400_a05[2] / 0.05 - 1)
        and n400_a05[2] / 0.05 - 1 < 0.01,
        "；".join(f"df={df}：α={a} 实付 {r:.4f}（相对虚报 {(r/a-1)*100:.1f}%）"
                  for df, a, r in rows if a == 0.05),
        tolerance="df=400 的相对误差 < 1%，且比 df=20 小 5 倍以上",
    )
    sec.observe(
        "这张表就是「n > 30」这句教科书口号的真实形状。它不是一条分界线，"
        f"而是一条**收敛速度很慢的尾巴**：df=100 时 α=0.05 实付还要虚报 "
        f"{(next(r for r in rows if r[0]==100 and r[1]==0.05)[2]/0.05-1)*100:.1f}%，"
        f"到 df=400 才压进 1% 以内。"
        "所以作者那句「实际应用中几乎所有从业者都用 t，我给 Z 检验只是为了接上 CLT」"
        "是对的，但更准确的说法应该是：**t 分布比正态永远更保守，而且它不额外花钱** —— "
        "现代软件里两个函数的调用成本完全一样，没有理由为了省一行代码去虚报。"
    )
    return {"rows": rows}


# ------------------------------------------------- ③ 比例检验：作者偷偷换了公式


def exact_binom_size(n: int, p0: float, alpha: float = 0.05) -> float:
    """精确二项检验在 H0 下的真实（双尾）误报率：把所有可能的 x 枚举一遍。"""
    x = np.arange(n + 1)
    pmf = stats.binom.pmf(x, n, p0)
    p_exact = np.array([_two_sided_binom_p(k, n, p0) for k in x])
    return float(pmf[p_exact <= alpha].sum())


def _two_sided_binom_p(k: int, n: int, p0: float) -> float:
    """scipy 的双尾二项 p 值（method of small p-values）。"""
    pmf = stats.binom.pmf(np.arange(n + 1), n, p0)
    return float(pmf[pmf <= pmf[k] * (1 + 1e-9)].sum())


def proportion_size(p0: float, n: int, alpha: float = 0.05,
                    mc: int = MC, seed: int = 65) -> dict:
    """H0: p = p0 的双尾检验，三种统计量的实际误报率：
    score（SE 用 p₀，第 6 章这节用的）/ wald（SE 用 p̂，第 5 章 CI 那版）/ 精确二项。
    """
    rng = np.random.default_rng(seed)
    x = rng.binomial(n, p0, size=mc)
    ph = x / n
    se0 = np.sqrt(p0 * (1 - p0) / n)
    se_hat = np.sqrt(ph * (1 - ph) / n)
    z_score = np.where(n > 0, (ph - p0) / se0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_wald = np.where(se_hat > 0, (ph - p0) / se_hat, 0.0)
    rej_score = np.abs(z_score) > stats.norm.ppf(1 - alpha / 2)
    rej_wald = np.abs(z_wald) > stats.norm.ppf(1 - alpha / 2)
    return {"score": float(rej_score.mean()),
            "wald": float(rej_wald.mean()),
            "exact": exact_binom_size(n, p0, alpha),
            "np_min": min(n * p0, n * (1 - p0))}


def experiment_proportion(sec: Section) -> dict:
    cases = [(0.65, 45), (0.05, 100), (0.50, 30)]      # 最后一个是回测里最常见的形态
    rows = []
    for p0, n in cases:
        r = proportion_size(p0, n)
        rows.append({"p0": p0, "n": n, **r})
    edge = rows[1]      # np = 5，正好卡在书上那条规则的边界
    worst_wald = max(rows, key=lambda r: abs(r["wald"] - 0.05))

    sec.check(
        "比例检验这里作者换了个更好的公式：SE 用 H0 的 p₀ ⇒ np=5 边界上 score 版误报接近名义值",
        abs(edge["score"] - 0.05) < 0.02
        and abs(edge["score"] - 0.05) < abs(edge["wald"] - 0.05),
        f"p₀={edge['p0']}、n={edge['n']}（np={edge['np_min']:.0f}）："
        f"score 版（书上）误报 {edge['score']:.4f}、Wald 版（第 5 章那套）{edge['wald']:.4f}、"
        f"精确二项 {edge['exact']:.4f}；"
        f"三种里最跑偏的是 p₀={worst_wald['p0']}, n={worst_wald['n']} 上的 Wald：{worst_wald['wald']:.4f}",
        tolerance="score 版与名义 0.05 相差 < 0.02，且比同条件的 Wald 版更稳",
    )
    backtest = rows[2]
    sec.check(
        "回测最爱的形态（30 笔交易、H0 胜率 50%）也是名义水平最难兑现的形态：三者都不等于 5%",
        abs(backtest["score"] - 0.05) > 0.005 or abs(backtest["wald"] - 0.05) > 0.005,
        f"p₀=0.50、n=30：score {backtest['score']:.4f}、Wald {backtest['wald']:.4f}、"
        f"精确二项 {backtest['exact']:.4f}（精确版天生偏保守，这是离散分布的必然，"
        f"想把它凑成 5% 只能靠随机化检验）",
        tolerance="至少一个版本的实际误报率偏离 5% 超过 0.5 个百分点",
    )
    sec.observe(
        "这是一处全书没有点破的不一致，但方向是对的："
        "第 5 章的置信区间用 p̂ 估标准误（Wald），这一章的检验改用 p₀ 估标准误（score），"
        "后者在系统 Wong 更稳，因为它不需要拿同一个 p̂ 既当估计又当尺度。"
        "作者可能只是顺着「检验里 H0 说 p 是多少，那尺度就用这个数」的直觉写下去，"
        "结果歪打正着 —— 这提醒我：**同一个统计问题，估计视角和检验视角的最优公式未必是同一个**。"
    )
    return {"rows": rows}


# ------------------------------------------------- ④ 两样本：书上的 z 版在什么条件下坏掉


def two_sample_size(s1_pop: float, s2_pop: float, n1: int, n2: int,
                    alpha: float = 0.05, mc: int = MC, seed: int = 66) -> dict:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, s1_pop, size=(mc, n1))
    b = rng.normal(0, s2_pop, size=(mc, n2))
    sa, sb = a.std(axis=1, ddof=1), b.std(axis=1, ddof=1)
    xa, xb = a.mean(axis=1), b.mean(axis=1)
    se = np.sqrt(sa ** 2 / n1 + sb ** 2 / n2)
    stat = (xa - xb) / se                       # 书上给的那个统计量
    # Welch 自由度每次都变，只能逐个算 —— 抽样做，避免太慢
    idx = rng.choice(mc, size=min(mc, 4000), replace=False)
    df = np.array([welch_df(sa[i], n1, sb[i], n2) for i in idx])
    rej_welch = np.abs(stat[idx]) > stats.t.ppf(1 - alpha / 2, df)
    return {
        "z": float(np.mean(np.abs(stat) > stats.norm.ppf(1 - alpha / 2))),
        "welch": float(rej_welch.mean()),
        "df_mean": float(df.mean()),
        "ratio": max(s1_pop, s2_pop) / min(s1_pop, s2_pop),
    }


def experiment_two_sample(sec: Section) -> dict:
    even = two_sample_size(10.0, 10.0, 37, 35, seed=661)      # 书上例题的同款规模，方差齐
    harsh = two_sample_size(6.0, 18.0, 40, 12, seed=662)      # 方差 1:3、样本量也不齐
    sec.check(
        "方差齐 + 样本量接近时，书上的 z 版与 Welch 差别可以忽略（都 ≈5%）",
        abs(even["z"] - 0.05) < 0.012 and abs(even["welch"] - 0.05) < 0.012,
        f"σ₁=σ₂=10、n₁=37、n₂=35（书例题的量级）：z 版误报 {even['z']:.4f}、"
        f"Welch {even['welch']:.4f}，平均 Welch 自由度 {even['df_mean']:.1f}",
        tolerance="两者与名义 5% 相差都 < 1.2 个百分点",
    )
    sec.check(
        "方差 1:3 且样本量也悬殊时，书上的 z 版系统性虚报，Welch 版基本守住 5%",
        harsh["z"] > 0.06 and abs(harsh["welch"] - 0.05) < 0.015
        and harsh["z"] - harsh["welch"] > 0.015,
        f"σ₁=6、σ₂=18、n₁=40、n₂=12：z 版误报 {harsh['z']:.4f}、"
        f"Welch {harsh['welch']:.4f}（自由度均值 {harsh['df_mean']:.1f}），"
        f"两者相差 {(harsh['z']-harsh['welch'])*100:.1f} 个百分点",
        tolerance="z 版 > 6%、Welch 在 5%±1.5%、差值 > 1.5 个百分点",
    )
    sec.observe(
        "这个坑的形态很稳定：**方差不齐本身不可怕，可怕的是方差不齐叠上样本量悬殊** —— "
        "哪一组样本少、哪一组方差大，误差就会被放大。"
        "作者在独立样本那一节留的注释（『T 分布通常是应对两个样本方差是否不同这一不确定性的首选工具』）"
        "是对的，只是用一行注释带过了它的代价。"
    )
    sec.pitfall(
        "对做回测的人，这个场景每天都会发生：把「有信号的那几天」和「没信号的那几天」"
        "当成两个独立样本比较收益，前者 40 天、后者几百天，两组方差天然不同。"
        "用书上的 z 版会让你更容易宣布「有信号的日子收益显著更高」。"
    )
    return {"even": even, "harsh": harsh}


# ------------------------------------------------- 图


def draw(sec: Section, bill: dict, prop: dict, two: dict) -> None:
    rows = bill["rows"]
    dfs = sorted({r[0] for r in rows})
    fig, axes = plot.newfig(1, 2, figsize=(8.0, 3.0))
    for ax, alpha in zip(axes, (0.05, 0.01)):
        vals = [next(r for r in rows if r[0] == df and r[1] == alpha)[2] * 100 for df in dfs]
        ax.plot(range(len(dfs)), vals, "o-", color=plot.ACCENT, lw=1.3, ms=4)
        ax.axhline(alpha * 100, ls="--", lw=1.0, color=plot.MUTED, label=f"名义 α={alpha}")
        ax.set_xticks(range(len(dfs)))
        ax.set_xticklabels([str(d) for d in dfs], fontsize=8)
        ax.set_xlabel("自由度 df = n−1")
        if ax is axes[0]:
            ax.set_ylabel("真实误报率 %")
        ax.set_title(f"α = {alpha}", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("σ 未知却查正态表：同一条 CLT 捷径，样本越小越贵", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    sec.figure("z-bill", fig,
               caption="α=0.01 那条线比 α=0.05 跌得更狠 —— 阈值越严格，近似误差占比越高")

    fig, axes = plot.newfig(1, 2, figsize=(8.0, 3.0))
    labels = [f"p₀={r['p0']}\nn={r['n']}" for r in prop["rows"]]
    xs = np.arange(len(prop["rows"]))
    w = 0.26
    axes[0].bar(xs - w, [r["score"] * 100 for r in prop["rows"]], w,
                color=plot.ACCENT2, label="score（第 6 章）")
    axes[0].bar(xs, [r["wald"] * 100 for r in prop["rows"]], w,
                color=plot.ACCENT, label="Wald（第 5 章）")
    axes[0].bar(xs + w, [r["exact"] * 100 for r in prop["rows"]], w,
                color=plot.MUTED, label="精确二项")
    axes[0].axhline(5, ls="--", lw=1.0, color="k")
    axes[0].set_xticks(xs, labels, fontsize=8)
    axes[0].set_ylabel("真实双尾误报率 %")
    axes[0].set_title("比例检验：书上没有点破的换公式", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    names = ["方差齐\n n=37/35", "方差 1:3\n n=40/12"]
    vals = [two["even"], two["harsh"]]
    xs = np.arange(2)
    w = 0.35
    axes[1].bar(xs - w / 2, [v["z"] * 100 for v in vals], w,
                color=plot.ACCENT, label="书上的 z 版")
    axes[1].bar(xs + w / 2, [v["welch"] * 100 for v in vals], w,
                color=plot.ACCENT2, label="Welch t")
    axes[1].axhline(5, ls="--", lw=1.0, color="k")
    axes[1].set_xticks(xs, names, fontsize=8)
    axes[1].set_title("两样本：什么时候必须用 Welch", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sec.figure("proportion-and-two-sample", fig,
               caption="左：第 6 章的 score 版比第 5 章的 Wald 版稳；右：方差不齐 + 样本悬殊才见真章")


def run(sec: Section) -> None:
    repro = experiment_reproduce_exercises(sec)
    bill = experiment_small_n_bill(sec)
    prop = experiment_proportion(sec)
    two = experiment_two_sample(sec)
    draw(sec, bill, prop, two)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
