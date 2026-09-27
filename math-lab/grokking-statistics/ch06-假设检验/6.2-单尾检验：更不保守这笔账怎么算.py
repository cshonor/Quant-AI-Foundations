"""6.2 单尾检验：作者说它「更不保守」，这笔账到底怎么算

书上先把大鼠的例子改成方向性主张：

    H0: μ ≥ 142        Ha: μ < 142

把 α 全压到一侧、不再平分两个尾部 ⇒ 临界值从 ±1.959964 变成单侧的 −1.644854，
同一份薯片数据（z = −2.4062）的双尾 p = 0.0161 也变成单尾 p = 0.0081，
正好一半。作者的结论是：单尾「通常会给出双尾一半的 p 值」，
因此在同样的 α 下更容易拒绝 H0，也就是更不保守；
并且提醒「只有当你关心单一方向的变化时才用单尾」。

这节把这句话拆成三笔可结算的账：

  · 「更容易拒绝」这句话只在**效应方向押对了**的时候成立：
    在真实的 H0 世界里，三种检验的误报率都老老实实是 5%；
  · 真正的代价在反面：方向押反了，左尾检验对这个效应的灵敏度几乎归零（实测 0.03%），
    而同一份数据双尾还有 ~62% —— 单尾是拿「另一侧的视力」换「这一侧的门槛」；
  · 最贵的一种用法是**看完数据再决定用哪个尾巴**：双尾不显著就按观测方向改单尾，
    名义 5% 的检验当场变成 ~10%，而且有整整约 5% 的样本落在「换个说法就显著」的区间里。
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
    number="6.2",
    chapter="第 6 章 · 假设检验",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="单尾检验：「更不保守」这笔账怎么算",
    claim="把薯片的主张改成方向性的 Ha: μ < 142，显著性区域就集中到一个尾部："
          "左侧临界 Z = −1.644854（不再有 ±），同一份数据 z = −2.4062 的 p 值从双尾 0.0161 "
          "变成单尾 0.0081 —— 正好一半。作者据此说单尾检验「更容易拒绝 H0」，"
          "因而不如双尾保守，并建议只在确实只关心一个方向时才用它。",
    proposition="① 复现数字：单尾临界 −1.644854、p = 0.0081，且恰好是双尾的一半；"
                "② 但「更容易拒绝」不是免费的：在真实 H0 下三种检验的误报率都是 5%；"
                "③ 单尾的代价是另一侧的视力 —— 效应押反方向时左尾灵敏度仅 ~0.03%，同一份数据双尾仍有 ~62%；"
                "④ 看完数据再挑尾巴：名义 5% 的检验当场变成 ~10%；"
                "⑤ 可被「换尾巴」翻盘的样本占真 H0 世界的约 5%，与 α 本身等宽。",
    source="第 6 章「单尾检验」「单尾检验的显著性水平」「单尾 p 值」（薯片 142 克左侧版本）",
)

ALPHA = 0.05
MU, N, X_BAR, S = 142.0, 41, 140.2, 4.79
MC = 60_000


# ------------------------------------------------- ① 复现单尾数字


def experiment_one_tail_book(sec: Section) -> dict:
    z = (X_BAR - MU) / (S / np.sqrt(N))
    crit_left = float(stats.norm.ppf(ALPHA))              # 左尾：−1.644854
    crit_two = float(stats.norm.ppf(1 - ALPHA / 2))       # 双尾：±1.959964
    p_left = float(stats.norm.cdf(z))
    p_two = float(2 * stats.norm.cdf(-abs(z)))
    sec.check(
        "复现书上单尾例子：临界 −1.644854、p = 0.0081，且正好是双尾 0.0161 的一半",
        abs(crit_left + 1.644854) < 1e-5 and abs(p_left - 0.0081) < 5e-4
        and abs(p_left * 2 - p_two) < 1e-9,
        f"z = {z:.4f}；左尾临界 {crit_left:.6f}（书 −1.644854）、"
        f"双尾临界 ±{crit_two:.6f}；左尾 p = {p_left:.4f}（书 0.0081）、"
        f"双尾 p = {p_two:.4f} —— 比值 {p_two/p_left:.6f}",
        tolerance="临界到 1e-5、p 到 5e-4、且 p_single ×2 == p_two",
    )
    sec.observe(
        "p 值减半不是什么技巧，是几何必然：双尾把同一份极值对称地算了两遍。"
        "所以真正要讨论的不是「p 变小了」，而是「我有没有资格只数一边」。"
    )
    return {"z": z, "crit_left": crit_left, "crit_two": crit_two,
            "p_left": p_left, "p_two": p_two}


# ------------------------------------------------- ② H0 下三者同样守规矩


def simulate(alpha: float = ALPHA, n: int = N, sigma: float = S,
             mu_true: float = MU, mc: int = MC, seed: int = 62) -> dict:
    """在给定真实均值下，跑三种尾部检验，返回各自的拒绝率。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(mu_true, sigma, size=(mc, n))
    stat = (x.mean(axis=1) - MU) / (x.std(axis=1, ddof=1) / np.sqrt(n))
    df = n - 1
    return {
        # 用各自分布的正确临界值（t 版），这样三者都处在被校准的状态
        "left": float(np.mean(stat < stats.t.ppf(alpha, df))),
        "right": float(np.mean(stat > stats.t.ppf(1 - alpha, df))),
        "two": float(np.mean(np.abs(stat) > stats.t.ppf(1 - alpha / 2, df))),
        # 事后挑尾：双尾显著就罢，不显著就按观测方向改单尾
        "posthoc": float(np.mean(
            (np.abs(stat) > stats.t.ppf(1 - alpha / 2, df))
            | ((stat > 0) & (stat > stats.t.ppf(1 - alpha, df)))
            | ((stat < 0) & (stat < stats.t.ppf(alpha, df)))
        )),
        "stat": stat,
    }


def experiment_same_size(sec: Section) -> dict:
    res = simulate(mc=MC)
    se = float(np.sqrt(ALPHA * (1 - ALPHA) / MC))
    ok = all(abs(res[k] - ALPHA) < 4 * se for k in ("left", "right", "two"))
    sec.check(
        "在真实 H0 世界里，左尾 / 右尾 / 双尾的误报率都是 5% —— 单尾并没有额外送 rejects",
        ok,
        f"α = 0.05，n = 41，t 临界值（σ 未知的正确版本），"
        f"{MC:,} 次重复：左尾 {res['left']:.4f}、右尾 {res['right']:.4f}、双尾 {res['two']:.4f}"
        f"（蒙特卡洛标准误 ±{se:.4f}）",
        tolerance="三者都落在 4 倍标准误内",
    )
    sec.observe(
        "我原本期待这里能抓到作者的措辞漏洞 —— 结果没有："
        f"三者实付都是 5% 左右。书上那句「单尾更容易拒绝 H0」在 H0 真的世界里**不成立**，"
        "单尾改变的不是误报率，而是**拒绝区的形状** —— 它把 5% 从「两个尾部各一半」挪到「一个尾部全押」，"
        "这正是作者给的那个「靶子更大」的比方，只是靶子大只有在真有东西可打的时候才吃香。"
    )
    return res


# ------------------------------------------------- ③ 代价：另一侧看不见了


def power_curve(deltas: np.ndarray, n: int = N, sigma: float = S,
                alpha: float = ALPHA) -> dict:
    """解析功效：备择假设 μ = 142 + Δ 时三种检验抓住它的概率。"""
    df = n - 1
    se = sigma / np.sqrt(n)
    cl = float(stats.t.ppf(alpha, df))
    cr = float(stats.t.ppf(1 - alpha, df))
    c2 = float(stats.t.ppf(1 - alpha / 2, df))
    left, right, two = [], [], []
    for d in deltas:
        nc = d / se                     # 非中心参数
        left.append(float(stats.nct.cdf(cl, df, nc)))
        right.append(float(stats.nct.sf(cr, df, nc)))
        two.append(float(stats.nct.cdf(-c2, df, nc) + stats.nct.sf(c2, df, nc)))
    return {"deltas": deltas, "left": np.array(left),
            "right": np.array(right), "two": np.array(two)}


def experiment_blind_side(sec: Section) -> dict:
    """薯片实验中那个被观察的方向是『偏轻』。假如真相其实是『偏重』呢？"""
    d = +1.8                       # 与书上观察方向相反、但绝对值相同的效应
    sim = simulate(mu_true=MU + d, mc=40_000, seed=63)
    cur = power_curve(np.array([d]))
    sec.check(
        "方向押反时左尾检验几乎瞎掉：同样样本量的灵敏度从双尾 ~62% 掉到 0.03%",
        cur["left"][0] < 0.005 and cur["two"][0] > 0.5
        and cur["two"][0] - cur["left"][0] > 0.5,
        f"真实效应 +{d} 克（偏重，与 Ha: μ<142 相反）：左尾 power = {cur['left'][0]:.4%}、"
        f"双尾 power = {cur['two'][0]:.2%}、右尾 power = {cur['right'][0]:.2%}；"
        f"模拟复核（{40000:,} 次）左尾拒绝率 {sim['left']:.4f}、双尾 {sim['two']:.4f}",
        tolerance="左尾 power < 0.5% 且双尾 > 50%，两者相差 > 50 个百分点",
    )
    # 「押对方向」时单尾确实高出一截 —— 这才兑现作者那句话
    d_ok = -1.8
    cur2 = power_curve(np.array([d_ok]))
    sec.check(
        "方向押对时单尾确实更高：−1.8 克下左尾 power 比双尾高约 11 个百分点",
        cur2["left"][0] > cur2["two"][0]
        and 0.08 < cur2["left"][0] - cur2["two"][0] < 0.15,
        f"真实效应 −{abs(d_ok)} 克（偏轻）：左尾 power = {cur2['left'][0]:.2%}、"
        f"双尾 = {cur2['two'][0]:.2%}，差 {(cur2['left'][0]-cur2['two'][0])*100:.1f} 个百分点；"
        f"右尾（完全错的方向）只有 {cur2['right'][0]:.4%}",
        tolerance="差值落在 8~15 个百分点",
    )
    sec.observe(
        "所以「单尾更容易拒绝」这句话的完整版本应该是："
        "**在效应真的朝着你押的那个方向时**，它比双尾更容易拒绝；"
        "代价是如果事实朝反方向走，你几乎永远不会发现。"
        "把它翻译成工程语言：单尾检验是一个不可逆的过滤器，"
        "你放弃了另一半信息，换来这一半的灵敏度。"
    )
    sec.pitfall(
        "放到 CTA 上：如果你声明「我只关心这个新止损会不会让回撤变大」并用左尾，"
        "那么当它其实让回撤变小、但同时把收益打没了的时候，这个检验是看不见的。"
        "更常见也更隐蔽的错误，是先看到回测结果不错、再回头宣布「我本来就只关心正向」——"
        "那就是下一笔账。"
    )
    return {"cur_ok": cur2, "cur_wrong": cur, "sim": sim}


# ------------------------------------------------- ④ 看完数据再挑尾巴


def experiment_post_hoc_tail(sec: Section) -> dict:
    res = simulate(mc=MC, seed=64)
    se10 = float(np.sqrt(0.10 * 0.90 / MC))
    # 两种口径：用正确的 t 临界（与上面的模拟一致）/ 用书上的正态临界
    analytic_t = float(2 * stats.t.sf(stats.t.ppf(1 - ALPHA, N - 1), N - 1))
    analytic_z = float(2 * stats.t.sf(stats.norm.ppf(1 - ALPHA), N - 1))
    sec.check(
        "「先跑双尾、不显著就按观测方向改单尾」：名义 5% 的检验实测误报率翻倍到 10%",
        res["posthoc"] > 0.095 and abs(res["posthoc"] - analytic_t) < 4 * se10,
        f"诚实双尾 {res['two']:.4f}；事后挑尾 {res['posthoc']:.4f}"
        f"（名义 α = 0.05，虚报多 {(res['posthoc']/ALPHA-1)*100:.0f}%）；"
        f"解析值 P(|T₄₀| > t₀.₉₅) = {analytic_t:.4f} —— 恰好是两倍 α；"
        f"若沿书上的正态临界则 '{analytic_z:.4f}'",
        tolerance="实测 > 9.5% 且与解析值 0.10 相差小于 4 倍标准误",
    )
    # 「可被翻盘」的样本有多大一块：双尾 p ∈ (α, 2α)
    stat = res["stat"]
    df = N - 1
    p_two = 2 * stats.t.sf(np.abs(stat), df)
    swing = float(np.mean((ALPHA < p_two) & (p_two < 2 * ALPHA)))
    sec.check(
        "可被『换个说法』翻盘的样本占真 H0 世界约 5% —— 与 α 本身等宽",
        abs(swing - ALPHA) < 0.006,
        f"双尾 p 落在 (0.05, 0.10) 的样本占 {swing:.4f}；"
        f"这些样本只要改成同方向单尾，p 就落到 (0.025, 0.05) 全部显著 —— "
        f"这么一块区域正好是 α 的宽度，不是巧合",
        tolerance="与 0.05 相差 < 0.006",
    )
    sec.observe(
        "把这两条合起来就是 p-hacking 最小的工作原理："
        "不需要伪造任何数据，只要在看到结果之后，从一堆同样合法的检验里"
        "挑那个最顺眼的，就能把 5% 的门槛买到 10% 以上。"
        f"而且它有整整 {swing:.1%} 的样本储备可用 —— 也就是说，"
        "在一个「什么都没发生」的世界里，每 20 次尝试就约有 1 次能被这套话术包装成发现。"
        "作者在这节只留了一句「实验开始前就先定好 α」，第 9 章会展开。"
    )
    sec.pitfall(
        "写回测报告的人最容易踩的形态：网格搜了 30 组参数，"
        "拿最好的那组说「它的收益显著为正（单尾 p = 0.03）」。"
        "这里的自由度不是一次检验，而是 30 次挑一次 —— 第 9 章会用多重比较把这笔账算清楚。"
    )
    return {"res": res, "analytic": analytic_t, "swing": swing}


# ------------------------------------------------- 图


def draw(sec: Section, book: dict, size: dict, wrong: dict, post: dict) -> None:
    # 图 1：三种检验在 α=0.05 下到底把 5% 放在哪里
    fig, axes = plot.newfig(1, 3, figsize=(9.6, 2.7))
    x = np.linspace(-4, 4, 500)
    pdf = stats.t.pdf(x, N - 1)
    crits = {
        "左尾（书上这题）": (stats.t.ppf(ALPHA, N - 1), None),
        "右尾": (None, stats.t.ppf(1 - ALPHA, N - 1)),
        "双尾": (stats.t.ppf(-ALPHA / 2, N - 1), stats.t.ppf(1 - ALPHA / 2, N - 1)),
    }
    for ax, (name, (lo, hi)) in zip(axes, crits.items()):
        ax.plot(x, pdf, lw=1.2, color=plot.ACCENT2)
        if lo is not None:
            m = x <= lo
            ax.fill_between(x[m], 0, pdf[m], color=plot.ACCENT, alpha=0.75)
        if hi is not None:
            m = x >= hi
            ax.fill_between(x[m], 0, pdf[m], color=plot.ACCENT, alpha=0.75)
        ax.set_title(name, fontsize=9.5)
        ax.set_yticks([])
        ax.set_xlabel("t")
    fig.suptitle("同样是 5% 的拒绝区：放在哪一边，决定了你能看见什么", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    sec.figure("rejection-regions", fig,
               caption="左 1 = 书上单尾版本；右 1 = 反方向；中间那块的面积总和仍然是 5%")

    # 图 2：功效曲线 —— 选错方向的代价
    deltas = np.linspace(-3, 3, 241)
    pc = power_curve(deltas)
    fig, ax = plot.newfig(figsize=(6.8, 3.1))
    ax.plot(deltas, pc["left"] * 100, lw=1.4, color=plot.ACCENT, label="左尾 Ha: μ<142")
    ax.plot(deltas, pc["right"] * 100, lw=1.4, color="#639922", label="右尾 Ha: μ>142")
    ax.plot(deltas, pc["two"] * 100, lw=1.4, ls="--", color=plot.ACCENT2, label="双尾 Ha: μ≠142")
    ax.axvline(0, lw=0.8, color=plot.MUTED)
    ax.set_xlabel("真实的均值偏移 Δ（克）"); ax.set_ylabel("抓住效应的概率 %")
    ax.set_title("n=41、σ=4.79：单尾把另一侧的视力换成了这一侧的灵敏度", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    sec.figure("power-curve", fig,
               caption="Δ=−1.8 时左尾确实更高；但 Δ=+1.8 时它几乎为零 —— 书上没有画的那一半")

    # 图 3：诚实 vs 事后挑尾
    res, analytic, swing = post["res"], post["analytic"], post["swing"]
    fig, axes = plot.newfig(1, 2, figsize=(7.6, 2.9))
    names = ["双尾（事前声明）", "事后挑尾"]
    vals = [res["two"], res["posthoc"]]
    axes[0].bar(names, np.array(vals) * 100, color=[plot.ACCENT2, plot.ACCENT], width=0.55)
    axes[0].axhline(ALPHA * 100, ls="--", lw=1.0, color="k", label="名义 α = 5%")
    for i, v in enumerate(vals):
        axes[0].annotate(f"{v*100:.2f}%", (i, v * 100), ha="center",
                         xytext=(0, 4), textcoords="offset points", fontsize=9)
    axes[0].set_ylim(0, max(vals) * 100 * 1.25)
    axes[0].set_ylabel("真实误报率 %")
    axes[0].set_title("看数据之后再选尾巴的代价", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    stat = res["stat"]
    p_two = 2 * stats.t.sf(np.abs(stat), N - 1)
    axes[1].hist(p_two, bins=50, range=(0, 1), color=plot.MUTED, alpha=0.7)
    axes[1].hist(p_two[(ALPHA < p_two) & (p_two < 2 * ALPHA)], bins=50, range=(0, 1),
                 color=plot.ACCENT, alpha=0.9,
                 label=f"可被换尾翻盘：{swing:.1%}")
    axes[1].axvline(ALPHA, ls="--", lw=1.0, color="k")
    axes[1].set_xlabel("诚实的双尾 p 值"); axes[1].set_ylabel("频数")
    axes[1].set_title("什么都没发生的世界里，照样有储备能量", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sec.figure("posthoc", fig,
               caption="左边：逻辑上仍然合规、道德上已经变质的那一步；右边：它有多少子弹可用")


def run(sec: Section) -> None:
    book = experiment_one_tail_book(sec)
    size = experiment_same_size(sec)
    wrong = experiment_blind_side(sec)
    post = experiment_post_hoc_tail(sec)
    draw(sec, book, size, wrong, post)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
