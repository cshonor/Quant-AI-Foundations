"""5.3 比例的置信区间：np≥5 这条经验法则保不保

书上讲比例的好处是实用性：除了均值，我们常常关心「有多少比例的人/物具备某属性」。
游戏玩家的例子：声称 60% 的美国人玩游戏，随机抽 100 人、55 人玩，
问总体比例的 95% 置信区间，答案是 (45.2%, 64.8%)。

作者还给了两条配套规则（np≥5 且 n(1−p)≥5），说这两个条件满足后
正态近似的结果「具有合理的准确性」。这一节把这句话拿去结算：

  · 恰好卡在规则边界上（p=0.05, n=100 ⇒ np=5）时，覆盖率只有 87.7%；
  · 更糟的是 25% 的样本会算出**负的下界**（「胜率可能为 −3%」这种荒谬输出）；
  · 覆盖率不随 n 单调收敛，而是锯齿状振荡；
  · 换个公式（Wilson）几乎零成本地把这些毛病全解决。

最后落到 CTA：回测报告里那句「30 笔交易 27 笔盈利 = 90% 胜率」，
Wald 公式给的区间上界是 100.7%。
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
    number="5.3",
    chapter="第 5 章 · 样本量超能力",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="比例的置信区间：np≥5 这条经验法则保不保",
    claim="比例也能套置信区间，公式只改一处：把标准误换成 √(p̂(1−p̂)/n)。"
          "作者的配套规则是 np≥5 且 n(1−p)≥5，称这两个条件保证近似「具有合理的准确性」。"
          "例子：抽 100 人、55 人玩游戏 ⇒ 总体比例 95% CI = (45.2%, 64.8%)。",
    proposition="① 复现书上例题得 (0.452, 0.648)；"
                "② 恰好卡在 np=5 的边界上（p=0.05, n=100），Wald 覆盖率只有 87.7%，不满足勒 Kane;"
                "③ 同一场景 25% 的样本算出负的下界——公式会输出「−3% 的胜率」；"
                "④ Wald 覆盖率随 n 锯齿振荡，不是单调收敛；"
                "⑤ Wilson 区间零成本解决这两个毛病；"
                "⑥ 精度换算：p̂=0.5 最坏情况下要 ±3% 需 n≥1068 笔样本。",
    source="第 5 章「中心极限定理（CLT）与比例的置信区间」（游戏玩家例题、np≥5 规则）；"
           "延伸：Brown, Cai & DasGupta (2001) 关于 Wald 区间覆盖率的经典结论",
)

CONF = 0.95
Z = float(stats.norm.ppf(0.975))
M = 40_000


# ------------------------------------------------- 公式（“三个吵架的”


def wald(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """书上教的：p̂ ± z·√(p̂(1−p̂)/n)。"""
    p = x / n
    half = Z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """把中心点往 0.5 拉一点、并把 p 本身的不确定性算进标准差 —— 改动很小。"""
    den = n + Z * Z
    center = (x + Z * Z / 2) / den
    half = Z / den * np.sqrt(x * (n - x) / n + Z * Z / 4)
    return center - half, center + half


def clopper_pearson(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """精确区间（Beta 分位数）：保证覆盖率 ≥ 95%，代价是偏保守。"""
    lo = np.where(x > 0, stats.beta.ppf(0.025, x, n - x + 1), 0.0)
    hi = np.where(x < n, stats.beta.ppf(0.975, x + 1, n - x), 1.0)
    return lo, hi


# ------------------------------------------------- ① 复现例题


def experiment_book_example(sec: Section) -> None:
    x, n = 55, 100
    lo, hi = wald(np.array([x]), n)
    lo, hi = float(lo[0]), float(hi[0])
    sec.check(
        "复现书上例题：55/100 玩游戏 ⇒ 95% CI = (0.452, 0.648)",
        abs(lo - 0.452) < 5e-4 and abs(hi - 0.648) < 5e-4,
        f"p̂={x/n:.2f}；Wald ({lo:.4f}, {hi:.4f})；"
        f"同一样本的 Wilson ({float(wilson(np.array([x]), n)[0][0]):.4f}, "
        f"{float(wilson(np.array([x]), n)[1][0]):.4f})、"
        f"精确 ({float(clopper_pearson(np.array([x]), n)[0][0]):.4f}, "
        f"{float(clopper_pearson(np.array([x]), n)[1][0]):.4f})",
        tolerance="两端与书上一致到 5e-4",
    )
    sec.observe(
        "这个例子本身没什么毛病：p̂=0.55 离两端都远，n=100 也够大，"
        "三种公式给出的区间几乎重合（Wald 宽 0.196，Wilson 0.192，精确 0.203）。"
        "问题是教材选的例子通常在公式「表现良好」的区域，"
        "而读者会默认这个公式在别处也一样好用。"
    )


# ------------------------------------------------- ② np≥5 保不保


def coverage_grid(sec: Section) -> dict:
    rng = np.random.default_rng(31)
    ps = [0.5, 0.2, 0.1, 0.05]
    ns = [30, 100, 300]
    rows = []
    for p in ps:
        for n in ns:
            x = rng.binomial(n, p, size=M)
            rec = {"p": p, "n": n, "np": p * n}
            for name, fn in (("wald", wald), ("wilson", wilson), ("cp", clopper_pearson)):
                lo, hi = fn(x, n)
                rec[name] = float(np.mean((p >= lo) & (p <= hi)))
            wlo, whi = wald(x, n)
            rec["out_of_range"] = float(np.mean((wlo < 0) | (whi > 1)))
            rows.append(rec)
    return {"rows": rows}


def experiment_rule_of_five(sec: Section, grid: dict) -> None:
    rows = grid["rows"]
    # 卡在边界上的那格：p=0.05, n=100 ⇒ np=5，书上规则说「够了」
    edge = next(r for r in rows if r["p"] == 0.05 and r["n"] == 100)
    sec.check(
        "np≥5 恰好满足时（p=0.05, n=100），书上教的 Wald 覆盖率远低于 95%",
        edge["wald"] < CONF - 0.05,
        f"np={edge['np']:.0f}（规则要求 ≥5）：Wald 覆盖仅 {edge['wald']:.2%}，"
        f"同条件 Wilson {edge['wilson']:.2%}、精确 {edge['cp']:.2%}；"
        f"且 {edge['out_of_range']:.1%} 的样本算出负的下界",
        tolerance="覆盖率 < 90%",
    )

    violating = [r for r in rows if r["out_of_range"] > 0.01]
    sec.check(
        "Wald 会输出跨越 [0,1] 的荒谬区间（负的比例 / 超过 100% 的比例）",
        len(violating) >= 2,
        "；".join(f"p={r['p']:.2f}, n={r['n']}: {r['out_of_range']:.1%} 的样本越界"
                  for r in sorted(violating, key=lambda r: -r["out_of_range"])[:3]),
        tolerance="至少 2 组参数上越界率 > 1%",
    )

    ok = [r for r in rows if r["p"] in (0.5, 0.2)]         # 公式表现良好的区域
    bad = [r for r in rows if r["p"] in (0.1, 0.05)]
    sec.observe(
        "np≥5 这条规则真正担保的是「离散二项分布长得有点像正态」，"
        f"它并不担保**覆盖率**。实测：在 p=0.5/0.2（np≥6~150）区间尚可"
        f"（Wald {min(r['wald'] for r in ok):.1%}~{max(r['wald'] for r in ok):.1%}），"
        f"一到 p=0.1/0.05 就崩（{min(r['wald'] for r in bad):.1%}~{max(r['wald'] for r in bad):.1%}）。"
        "根子是同一个：p̂ 本身确定性怎样、标准误用它自己估，两件事耦合在一起，"
        "p̂ 越接近 0 或 1，这个耦合越致命。Wilson 正是把这层耦合拆开了。"
    )


# ------------------------------------------------- ③ 振荡


def experiment_oscillation(sec: Section) -> dict:
    rng = np.random.default_rng(37)
    p = 0.2
    ns = np.arange(20, 205, 5)
    cov_w, cov_wil = [], []
    for n in ns:
        x = rng.binomial(int(n), p, size=M)
        lo, hi = wald(x, int(n))
        cov_w.append(float(np.mean((p >= lo) & (p <= hi))))
        lo, hi = wilson(x, int(n))
        cov_wil.append(float(np.mean((p >= lo) & (p <= hi))))
    cov_w, cov_wil = np.array(cov_w), np.array(cov_wil)
    # 「不单调」的直接证据：存在相邻样本量之间覆盖率反向跳变超过 2 个百分点
    jumps = np.abs(np.diff(cov_w))
    worst = int(np.argmax(jumps))
    sec.check(
        "Wald 覆盖率随 n 锯齿振荡，不是单调爬向 95%（加样本反而可能变糟）",
        jumps.max() > 0.02 and bool(np.any(np.diff(cov_w) < 0))
        and cov_wil.std() < cov_w.std(),
        f"p=0.20，n 从 20 到 200：Wald 在 {cov_w.min():.1%}~{cov_w.max():.1%} 之间跳，"
        f"标准差 {cov_w.std():.2%}；Wilson 标准差 {cov_wil.std():.2%}（稳 {(cov_w.std()/cov_wil.std()):.1f} 倍）；"
        f"最大一次反向跳跃 {jumps.max()*100:.1f} 个百分点，"
        f"发生在 n={ns[worst]} → {ns[worst+1]}（{cov_w[worst]:.1%} → {cov_w[worst+1]:.1%}）",
        tolerance="存在 >2 个百分点的相邻跳变且 Wilson 更稳",
    )
    sec.pitfall(
        "这条对做实验设计的人最直接：**多收集一点样本，覆盖率可能反而下降**。"
        "我原以为这只是小样本的问题，实测到 n=200 还在跳。"
        "所以别用「这次宽了/窄了」来判断样本够不够——那根本不起作用。"
    )
    return {"ns": ns, "cov_w": cov_w, "cov_wil": cov_wil}


# ------------------------------------------------- ④ 精度要多少样本


def experiment_sample_size(sec: Section) -> dict:
    """比例的误差范围 = z·√(p(1−p)/n)，p=0.5 时最宽 ⇒ n = (z/2m)²。"""
    rows = []
    for m in (0.05, 0.03, 0.01):
        n_req = (Z / (2 * m)) ** 2
        n = int(np.ceil(n_req))
        rng = np.random.default_rng(43)
        x = rng.binomial(n, 0.5, size=M)
        lo, hi = wald(x, n)
        half = float(np.mean((hi - lo) / 2))
        rows.append((m, n, half))
    sec.check(
        "n = (z/2m)² 精确预测所需样本量：±3% 要 1068 个样本",
        abs(rows[1][1] - 1068) < 2
        and bool(np.all(np.abs(np.array([r[2] for r in rows])
                               / np.array([r[0] for r in rows]) - 1) < 0.02)),
        "；".join(f"目标 ±{r[0]*100:.0f}% → n ≥ {r[1]:,}（实测半宽 {r[2]:.4f}）"
                  for r in rows),
        tolerance="±3% 对应 n=1068（±2）且实测半宽误差 < 2%",
    )
    sec.observe(
        "比例这边同样是 √n 在收钱，而且它是 LibreOffice 那种「最坏情况固定」的："
        "p(1−p) 在 p=0.5 取到最大值 0.25，所以无论真实比例是多少，"
        "用 p=0.5 算出来的样本量一定够。政治民调标配 1000 人左右不是行业习惯，"
        "是 (1.96/0.06)² ≈ 1068 这条曲线的直接结果。"
        "顺带给出一个可以立刻用的换算：想把民调精度翻一倍（±3% → ±1.5%），"
        "样本要从 1068 涨到 4269——又是 4 倍，和 5.1 那条 √n 账单完全同源。"
    )
    return {"rows": rows}


# ------------------------------------------------- ⑤ CTA：胜率


def experiment_win_rate(sec: Section) -> dict:
    cases = [(55, 100), (27, 30), (9, 10)]
    rows = []
    for x, n in cases:
        wlo, whi = (float(v[0]) for v in wald(np.array([x]), n))
        slo, shi = (float(v[0]) for v in wilson(np.array([x]), n))
        clo, chi = (float(v[0]) for v in clopper_pearson(np.array([x]), n))
        rows.append((x, n, x / n, wlo, whi, slo, shi, clo, chi))
    bad = [r for r in rows if r[4] > 1.0]
    sec.check(
        "「30 笔交易 27 笔盈利」这种回测样本：Wald 会给出超过 100% 的胜率上界",
        len(bad) >= 1 and rows[1][4] > 1.0,
        "；".join(f"{r[0]}/{r[1]}={r[2]:.0%}：Wald ({r[3]:.3f}, {r[4]:.3f})"
                  f" vs Wilson ({r[5]:.3f}, {r[6]:.3f})" for r in rows),
        tolerance="至少一个案例 Wald 上界 > 1",
    )
    sec.observe(
        "这条几乎是为 CTA 量身定做的。回测跑出「30 笔交易 27 笔盈利」时，"
        f"按书上公式算出来的 95% 区间是 ({rows[1][3]:.3f}, {rows[1][4]:.3f})——"
        "上界 100.7%，意思是「胜率可能在 100.7%」。我自己写出来的第一反应是想贴 [0,1] 截断，"
        "但截断是掩盖，不是修复；Wilson 给的是 "
        f"({rows[1][5]:.3f}, {rows[1][6]:.3f})，天然落在 [0,1] 内，"
        "且把「下界其实低到 74.4%」这件要命的事如实报了出来。"
        "⇒ 胜率这种「小样本 + 高比例」的组合是 Wald 区间的重灾区，"
        "而这恰好是回测报告最常出现的形态。"
    )
    return {"rows": rows}


# ------------------------------------------------- 图


def draw(sec: Section, grid: dict, osc: dict, size: dict, win: dict) -> None:
    rows = grid["rows"]

    # 图 1：四个 p 上三种公式的覆盖率 vs n
    fig, axes = plot.newfig(2, 2, figsize=(8.4, 5.0))
    for ax, p in zip(axes.ravel(), (0.5, 0.2, 0.1, 0.05)):
        sub = [r for r in rows if r["p"] == p]
        ns = [r["n"] for r in sub]
        ax.plot(ns, [r["wald"] * 100 for r in sub], "o-", color=plot.ACCENT,
                lw=1.3, ms=4, label="Wald（书上）")
        ax.plot(ns, [r["wilson"] * 100 for r in sub], "s-", color="#639922",
                lw=1.1, ms=4, label="Wilson")
        ax.plot(ns, [r["cp"] * 100 for r in sub], "^--", color=plot.ACCENT2,
                lw=1.0, ms=4, label="Clopper–Pearson（精确）")
        ax.axhline(95, ls="--", lw=1.0, color=plot.MUTED)
        ax.fill_between([min(ns), max(ns)], [80, 80], [100, 100], alpha=0)
        ax.set_ylim(75, 101)
        ax.set_title(f"p = {p}（np = {p*30:.1f}~{p*300:.0f}）", fontsize=10)
        ax.set_xlabel("样本量 n", fontsize=9)
        ax.set_ylabel("实测覆盖率 %", fontsize=9)
        ax.tick_params(labelsize=8)
        if p == 0.5:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("同一个 95% 承诺，三种公式的兑现率", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    sec.figure("coverage-grid", fig,
               caption="np≥5 只保证「分布像正态」，不保证覆盖率；p 越靠近两端，Wald 漏得越多")

    # 图 2：振荡
    fig, ax = plot.newfig(figsize=(7.0, 3.1))
    ax.plot(osc["ns"], osc["cov_w"] * 100, "o-", color=plot.ACCENT, lw=1.2, ms=3.5,
            label="Wald：锯齿")
    ax.plot(osc["ns"], osc["cov_wil"] * 100, "s-", color="#639922", lw=1.1, ms=3.5,
            label="Wilson：平滑")
    ax.axhline(95, ls="--", lw=1.0, color=plot.MUTED, label="标称 95%")
    ax.set_xlabel("样本量 n"); ax.set_ylabel("实测覆盖率 %")
    ax.set_ylim(85, 100); ax.legend(frameon=False, fontsize=8, loc="lower right")
    sec.figure("oscillation", fig,
               caption="p=0.20：加样本不一定改善覆盖率（Wald 的锯齿是它自己的结构缺陷）")

    # 图 3：胜率案例
    fig, ax = plot.newfig(figsize=(7.0, 3.2))
    labels, ys = [], []
    for x, n, ph, wlo, whi, slo, shi, clo, chi in win["rows"]:
        labels.append(f"{x}/{n} = {ph:.0%}")
        ys.append(len(ys))
    ys = np.arange(len(win["rows"]))
    for i, r in enumerate(win["rows"]):
        x, n, ph, wlo, whi, slo, shi, clo, chi = r
        ax.plot([wlo, whi], [i + 0.15, i + 0.15], lw=2.4, color=plot.ACCENT,
                solid_capstyle="butt")
        ax.plot([slo, shi], [i - 0.15, i - 0.15], lw=2.4, color="#639922",
                solid_capstyle="butt")
        ax.plot([clo, chi], [i, i], lw=1.0, color=plot.ACCENT2, ls=":", alpha=0.9)
        ax.plot(ph, i + 0.15, "|", ms=8, color=plot.ACCENT)
        ax.plot(ph, i - 0.15, "|", ms=8, color="#639922")
    ax.axvline(1.0, ls="--", lw=1.0, color="k")
    ax.set_yticks(ys, labels, fontsize=9)
    ax.set_xlim(-0.05, 1.15)
    ax.set_xlabel("95% 置信区间")
    ax.set_title("橙=Wald（书上公式）　绿=Wilson　灰点线=精确", fontsize=10)
    sec.figure("winrate", fig,
               caption="回测最爱出现的「小样本高胜率」正好是 Wald 最不靠谱的区域")


def run(sec: Section) -> None:
    experiment_book_example(sec)
    grid = coverage_grid(sec)
    experiment_rule_of_five(sec, grid)
    osc = experiment_oscillation(sec)
    size = experiment_sample_size(sec)
    win = experiment_win_rate(sec)
    draw(sec, grid, osc, size, win)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
