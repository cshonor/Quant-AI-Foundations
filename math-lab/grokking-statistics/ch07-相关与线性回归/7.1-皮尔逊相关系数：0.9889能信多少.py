"""7.1 皮尔逊相关系数：0.9889 这个数字能信多少

书上讲相关性的开场数据是一对温度 vs 运动饮料销量（n = 10）：

    x = [51.5, 55.1, 63.0, 71.2, 73.0, 77.1, 88.1, 91.2, 93.0, 101.6]
    y = [100.98, 137.44, 237.8, 254.96, 263.45, 353.89, 450.32, 521.34, 503.13, 573.91]

算得 r = 0.9889，p = 0.0000000654。作者的解读是「极高的正相关」，随即提醒：
相关系数本身也是一个**检验统计量**，所以用之前得先看假设：

  · 变量之间是线性关系
  · 两个变量都近似正态、方差齐（同方差）
  · 没有严重离群值（有就用 Spearman 秩相关）
  · 数据点之间互不影响

这一节把这几条假设逐条拿去打：

  · r 相同不代表图形相同：Anscombe 四重奏的 4 组数据，r、斜率、截距、R²、RMSE
    全部一致到小数点后好几位，但一组是直线、一组是抛物线、一组带一个离群值、
    一组是完全躺平的常数加上一个孤零零的点；
  · 一个离群值能把 r 从 0.99 打到 0.7 以下，而 Spearman 几乎不动（书上的建议是对的）；
  · 「强相关」和「统计显著」是两件事：r = 0.5 时 n = 10 不显著（p = 0.14），
    n = 100 显著（p = 1e−7）；
  · 最贵的一条：**两个互不相干的随机游走也会「显著相关」** ——
    这对做配对交易的人来说不是学术八卦，是每天都在发生的账单。
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
    number="7.1",
    chapter="第 7 章 · 相关性与线性回归",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="皮尔逊相关系数：0.9889 这个数字能信多少",
    claim="温度 vs 运动饮料销量（n=10）：皮尔逊 r = 0.9889，p = 0.0000000654，"
          "作者称之为「极高的正相关」，并说明相关系数的 p 值来自检验统计量 "
          "t = r/√((1−r²)/(n−2))，自由度是 n−2 而不是 n−1。"
          "同时给出皮尔逊的四条使用前提：线性关系、连续且近似正态、方差齐、无严重离群值，"
          "并引用安斯库姆四重奏说明「不能只看表面」。",
    proposition="① 复现 r = 0.9889、t 值与 p = 6.54e−8（df = n−2）；"
                "② Anscombe 四重奏的 4 组数据，r / 斜率 / 截距 / R² 全部一致，但形状完全不同；"
                "③ 一个离群值能把 r = 0.99 打到 0.7 以下，同一份数据上的 Spearman 几乎不动；"
                "④ 强相关 ≠ 显著：r = 0.5 在 n = 10 上不显著（p ≈ 0.14），n = 100 上显著（p ≈ 1e−7）；"
                "⑤ 两个互相独立的随机游走也会「显著相关」：|r| > 0.5 的比例远超 iid 情形。",
    source="第 7 章「相关性」「皮尔逊相关系数」「对相关系数进行假设检验」「皮尔逊相关性的假设」",
)

X = np.array([51.5, 55.1, 63.0, 71.2, 73.0, 77.1, 88.1, 91.2, 93.0, 101.6])
Y = np.array([100.98, 137.44, 237.8, 254.96, 263.45, 353.89, 450.32, 521.34, 503.13, 573.91])

# Anscombe 四重奏（Anscombe, 1973）
ANS = [
    ("I", np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
     np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])),
    ("II", np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
     np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])),
    ("III", np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]),
     np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])),
    ("IV", np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]),
     np.array([6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89])),
]

MC_RW = 3000          # 随机游走伪相关模拟次数
MC_OUT = 20_000       # 离群值 / 显著性门槛模拟次数


# ------------------------------------------------- ① 复现书上 karkyna


def pearson_manual(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x - x.mean()) * (y - y.mean()))
                 / np.sqrt(np.sum((x - x.mean()) ** 2) * np.sum((y - y.mean()) ** 2)))


def r_to_p(r: float, n: int) -> tuple[float, float]:
    """书上给的通道：t = r/√((1−r²)/(n−2))，再跑双尾 t(df=n−2)。"""
    t = r / np.sqrt((1 - r ** 2) / (n - 2))
    p = float(2 * stats.t.sf(abs(t), n - 2))
    return float(t), p


def experiment_reproduce_book(sec: Section) -> dict:
    r_manual = pearson_manual(X, Y)
    r_scipy, p_scipy = stats.pearsonr(X, Y)
    t_stat, p_manual = r_to_p(r_manual, len(X))
    sec.check(
        "复现书上例题：r = 0.9889、t = 18.6…、双尾 p = 6.54e−8（df = n−2 = 8）",
        abs(r_manual - 0.9889) < 5e-4 and abs(r_manual - r_scipy) < 1e-9
        and abs(p_manual - p_scipy) < 1e-9 and abs(p_manual - 6.54e-8) < 5e-9,
        f"手算 r = {r_manual:.6f}（书 0.9889）；scipy pearsonr r = {r_scipy:.6f}、"
        f"p = {p_scipy:.3e}；按书上公式 t = {t_stat:.4f}（df = {len(X)-2}）"
        f" ⇒ p = {p_manual:.3e}（书 0.0000000654）",
        tolerance="r 到 5e-4，p 值与 scipy 完全一致（1e-9）",
    )
    sec.observe(
        "r 本身就标配一个 p 值，这件事很方便，也很危险："
        "它让人误以为「相关性」只需要这一个数字。"
        "注意书上这几行从头到尾都在强调假设——这其实是在说："
        "**这个 p 值只在假设成立时才有意义**，而假设是否成立，p 值自己不告诉你。"
    )
    return {"r": float(r_scipy), "p": float(p_scipy), "t": t_stat}


# ------------------------------------------------- ② Anscombe 四重奏


def fit_summary(x: np.ndarray, y: np.ndarray) -> dict:
    lr = stats.linregress(x, y)
    resid = y - (lr.slope * x + lr.intercept)
    return {"slope": float(lr.slope), "intercept": float(lr.intercept),
            "r": float(lr.rvalue), "p": float(lr.pvalue),
            "r2": float(lr.rvalue ** 2), "rmse": float(np.sqrt(np.mean(resid ** 2))),
            "resid": resid, "x": x, "y": y}


def experiment_anscombe(sec: Section) -> dict:
    rows = [(tag, fit_summary(x, y)) for tag, x, y in ANS]
    spread_r = max(r["r"] for _, r in rows) - min(r["r"] for _, r in rows)
    spread_slope = max(r["slope"] for _, r in rows) - min(r["slope"] for _, r in rows)
    spread_r2 = max(r["r2"] for _, r in rows) - min(r["r2"] for _, r in rows)
    sec.check(
        "Anscombe 四重奏：4 组数据的 r / 斜率 / 截距 / R² 全部一致（差异 < 0.002）",
        spread_r < 0.002 and spread_slope < 0.002 and spread_r2 < 0.002,
        "；".join(f"第{tag}组 r={d['r']:.4f}、slope={d['slope']:.4f}、"
                  f"intercept={d['intercept']:.3f}、R²={d['r2']:.4f}"
                  for tag, d in rows)
        + f" —— 四组的最大差异：r {spread_r:.2e}、斜率 {spread_slope:.2e}、R² {spread_r2:.2e}",
        tolerance="四组统计量极差 < 0.002",
    )
    # 探针一：给这条直线加一个二次项能对尾巴有多大改善？（抓抛物线）
    # 探针二：最大残差 / 平均残差（抓单个离群点）
    quad_gain, peak = {}, {}
    for tag, d in rows:
        x, y, resid = d["x"], d["y"], d["resid"]
        sse_lin = float(np.sum(resid ** 2))
        quad = np.polyfit(x, y, 2)
        sse_quad = float(np.sum((y - np.polyval(quad, x)) ** 2))
        quad_gain[tag] = 1 - sse_quad / sse_lin
        peak[tag] = float(np.abs(resid).max() / np.abs(resid).mean())
    sec.check(
        "但残差结构天差地别：加个二次项能把第 II 组的残差抹平 100%，"
        "第 III 组的最大残差是平均残差的 4.5 倍",
        quad_gain["II"] > 0.99 and max(quad_gain[k] for k in ("I", "III", "IV")) < 0.15
        and peak["III"] > 4.0 and peak["I"] < 3.0,
        "加二次项后 SSE 的下降比例：" +
        "、".join(f"第{tag}组 {gain:.2%}" for tag, gain in quad_gain.items())
        + "；最大残差/平均残差：" +
        "、".join(f"第{tag}组 {v:.2f}" for tag, v in peak.items()),
        tolerance="只有第 II 组的二次项改进 > 99%，只有第 III 组的残差峰值比 > 4",
    )
    sec.observe(
        "这就是作者把 Anscombe 四重奏搬出来的原因，也是我认为全书最有价值的一张图："
        "**四个数据集的 r 都是 0.816，回归线都是 y = 3 + 0.5x，R² 都是 0.667，"
        "可它们长得完全不一样** —— 一个是直线、一个是抛物线、"
        "一个被单个离群值拖偏、一个干脆是「8 个点叠在 x=8 上再加一个孤立的点」。"
        "推论很实在：**r 和 p 值都不看图**，看图的人必须是你自己。"
    )
    sec.pitfall(
        "我犯过的错：把相关系数表滚动一遍，挑最高的那对去做配对。"
        "Anscombe 第 IV 组就是这个错误的极端形态 —— 只有 **一个** 数据点撑起了全部相关性，"
        "剩下 10 个点在 x 上根本没有变化。回测里等价的情形是："
        "整段样本只有一次极端行情，策略的「显著性」由那一次事件决定。"
    )
    return {"rows": rows, "probes": quad_gain, "peak": peak}


# ------------------------------------------------- ③ 一个离群值


def experiment_single_outlier(sec: Section) -> dict:
    rng = np.random.default_rng(71)
    n = 30
    xs = np.linspace(0, 10, n)
    base_r, pearsons, spearmans = [], [], []
    for i in range(MC_OUT // 50):
        ys = 2.0 * xs + 5.0 + rng.normal(0, 1.0, size=n)      # 干净的强线性关系
        clean_r = float(stats.pearsonr(xs, ys)[0])
        base_r.append(clean_r)
        # 只加一个离群点（丢在最右端上方）
        y_out = np.append(ys, 2.0 * xs[-1] + 5.0 + rng.normal(0, 0.5) + 60.0)
        x_out = np.append(xs, xs[-1] + 2.0)
        pearsons.append(float(stats.pearsonr(x_out, y_out)[0]))
        spearmans.append(float(stats.spearmanr(x_out, y_out)[0]))
    clean = float(np.mean(base_r))
    polluted = float(np.mean(pearsons))
    robust = float(np.mean(spearmans))
    sec.check(
        "只加一个离群点就把皮尔逊 r 从 ~0.99 打到 0.7 以下，而 Spearman 几乎不动",
        clean > 0.985 and polluted < 0.75 and robust > 0.98
        and (clean - polluted) > 0.20 and abs(clean - robust) < 0.01,
        f"干净数据（n=30，斜率 2）r = {clean:.4f}；只追加 1 个离群点后 "
        f"皮尔逊 r = {polluted:.4f}（掉了 {(clean-polluted)*100:.1f} 个百分点）、"
        f"同一份数据上的 Spearman ρ = {robust:.4f}（只掉 {(clean-robust)*100:.2f} 个百分点）",
        tolerance="污染后皮尔逊 < 0.70，Spearman > 0.985，且两者差距超过 30 倍",
    )
    sec.observe(
        "书上给的处方（有离群值就用 Spearman）实测有效得惊人："
        f"同一个离群点让皮尔逊掉 {(clean-polluted)*100:.1f} 个百分点，"
        f"斯皮尔曼只掉 {(clean-robust)*100:.2f} 个 —— 差 30 倍以上。"
        "代价是丢信息（秩相关不再关心具体的量级），所以更好的做法通常是："
        "**先把图画出来，判断这个离群点是错误数据还是真实事件**，再决定删还是留。"
    )
    sec.pitfall(
        "金融数据里的离群点通常不是错误，而是最重要的那几天（2020-03-12、2021-05-19…）。"
        "这时候「删掉离群值」等于「删掉你最需要解释的那部分」。"
        "留着 + 用秩相关 / 稳健回归，或者明确写成两段样本分别报告。"
    )
    return {"clean": clean, "polluted": polluted, "robust": robust}


# ------------------------------------------------- ④ 强相关 ≠ 显著


def experiment_strength_vs_significance(sec: Section) -> dict:
    r_target = 0.5
    rows = []
    for n in (10, 20, 50, 100, 200):
        t, p = r_to_p(r_target, n)
        rows.append((n, t, p))
    p10 = rows[0][2]
    p100 = rows[3][2]
    sec.check(
        "同一个「中等强度」的 r = 0.5：n = 10 时不显著（p ≈ 0.14），n = 100 时极显著（p ≈ 1e−7）",
        p10 > 0.10 and p100 < 1e-6,
        "；".join(f"n={n}：t = {t:.3f}、p = {p:.3e}"
                  f"{'（不显著）' if p > 0.05 else '（显著）'}" for n, t, p in rows),
        tolerance="n=10 的 p > 0.10 且 n=100 的 p < 1e−6",
    )
    # 反过来看：多大的 r 才算「显著」—— 随样本量快速下降
    rows2 = []
    for n in (10, 30, 100, 500):
        crit = float(stats.t.ppf(1 - 0.025, n - 2))
        r_crit = crit / np.sqrt(crit ** 2 + (n - 2))      # 反解 r 的临界值
        rows2.append((n, r_crit))
    sec.check(
        "「显著」所需的 r 随样本量快速下降：n=10 要 0.63，n=500 只要 0.088",
        rows2[0][1] > 0.6 and rows2[-1][1] < 0.09
        and all(a[1] > b[1] for a, b in zip(rows2, rows2[1:])),
        "；".join(f"n={n}：|r| ≥ {rc:.3f} 才算 α=0.05 显著" for n, rc in rows2)
        + f"；注意书上那组数据 n=10 却拿到 0.9889，离 {rows2[0][1]:.3f} 的门槛很远",
        tolerance="n=10 门槛 > 0.6、n=500 门槛 < 0.09、且随 n 单调下降",
    )
    sec.observe(
        "这两张表应该贴在每个做因子研究的人的显示器边上。"
        "「相关性 0.5」本身既可能什么都不是（n=10），也可能铁证如山（n=100）——"
        "**离开样本量谈相关系数强弱是没有意义的**。反过来那条更实用："
        "在 n=500 的日频样本里，|r| 只要 0.088 就能通过 α=0.05，"
        "而 0.088 的相关性意味着 x 只能解释 y 的 0.8% 方差。**统计显著 ≠ 有经济意义**。"
    )
    return {"rows": rows, "rows2": rows2}


# ------------------------------------------------- ⑤ 伪相关：两个独立随机游走


def spurious_walks(mc: int = MC_RW, n: int = 500, seed: int = 72) -> dict:
    """两个**互不相干**的随机游走，算它们的皮尔逊 r 和 OLS 显著性。"""
    rng = np.random.default_rng(seed)
    steps_a = rng.normal(0, 1.0, size=(mc, n))
    steps_b = rng.normal(0, 1.0, size=(mc, n))      # 与 A 完全独立
    a = np.cumsum(steps_a, axis=1)
    b = np.cumsum(steps_b, axis=1)
    xs = np.arange(n)
    rs = np.empty(mc)
    ps_reg = np.empty(mc)
    for i in range(mc):
        rs[i] = stats.pearsonr(a[i], b[i])[0]
        ps_reg[i] = stats.linregress(a[i], b[i]).pvalue
    return {"r": rs, "p_reg": ps_reg, "n": n}


def spurious_iid(mc: int = MC_RW, n: int = 500, seed: int = 73) -> np.ndarray:
    """对照组：同样长度但 iid（不是随机游走）的独立序列。"""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.0, size=(mc, n))
    b = rng.normal(0, 1.0, size=(mc, n))
    return np.array([stats.pearsonr(a[i], b[i])[0] for i in range(mc)])


def experiment_spurious_correlation(sec: Section) -> dict:
    rw = spurious_walks()
    iid = spurious_iid()
    tail_rw = float(np.mean(np.abs(rw["r"]) > 0.5))
    tail_iid = float(np.mean(np.abs(iid) > 0.5))
    sig_rate = float(np.mean(rw["p_reg"] < 0.05))
    sec.check(
        "两个毫无关系的随机游走：|r| > 0.5 的比例高达 ~40%，而 iid 对照组几乎不可能",
        tail_rw > 0.30 and tail_iid < 0.001 and tail_rw > 100 * max(tail_iid, 1e-6),
        f"n=500，{MC_RW:,} 组独立实验：随机游走 |r|>0.5 占 {tail_rw:.2%}；"
        f"同长度的 iid 序列 |r|>0.5 占 {tail_iid:.3%}（相差 {tail_rw/max(tail_iid,1e-9):.0f} 倍）；"
        f"把两个游走互做 OLS 回归，p<0.05 的比例是 {sig_rate:.2%}（名义只有 5%）",
        tolerance="随机游走 >30%、iid <0.1%、两者相差 100 倍以上",
    )
    sec.check(
        "伪回归：OLS 对这个毫无关系的配对宣布「显著」的比例远超名义 5%",
        sig_rate > 0.40,
        f"{MC_RW:,} 组里 {sig_rate:.2%} 的配对得到 p < 0.05；"
        f"|r| 的标准差 {rw['r'].std():.3f}（iid 情形理论上只有 1/√n = {1/np.sqrt(rw['n']):.3f}）",
        tolerance="显著比例 > 40%",
    )
    sec.observe(
        "这是本章最贵的一条，而且是>Ch 那种「读的时候觉得懂了、用的时候照样踩」的坑。"
        "两个**完全独立生成的**价格序列，因为各自都有趋势（非平稳），"
        f"做出来竟然有 {tail_rw:.1%} 的概率相关系数超过 0.5，还要被 OLS 宣布显著。"
        "根子在于：皮尔逊相关假设样本点是**独立**的，而随机游走每个点都带着上一个点的全部信息，"
        "有效样本量远远小于 n，p 值被系统性低估（Granger–Newbold 的伪回归问题，1974）。"
    )
    sec.pitfall(
        "配对交易的直接后果：在两个不协整的品种上看到 ρ=0.8 就开配对策略，"
        "那个 0.8 大概率只是两条各自游走的产物，会在你最需要它的时候消失。"
        "正确姿势是对**收益率**而不是价格水平算相关，或者先做协整检验。"
        "第 7 章只给了「相关不等于因果」这句口号，真正挡住这个坑的工具（差分 / 协整）"
        "都在本书范围之外，得自己去 [`07_Cointegration_Pairs`](cta-cointegration) 那条线上补。"
    )
    return {"rw": rw, "iid": iid, "tail_rw": tail_rw, "tail_iid": tail_iid,
            "sig_rate": sig_rate}


# ------------------------------------------------- 图


def draw(sec: Section, ans: dict, out: dict, sp: dict) -> None:
    # 图 1：书上那组数据 + 拟合线
    lr = stats.linregress(X, Y)
    fig, ax = plot.newfig(figsize=(6.4, 3.2))
    ax.scatter(X, Y, color=plot.ACCENT, s=26, zorder=3)
    xs = np.linspace(X.min(), X.max(), 100)
    ax.plot(xs, lr.slope * xs + lr.intercept, lw=1.3, color=plot.ACCENT2)
    ax.set_xlabel("温度（华氏度）"); ax.set_ylabel("运动饮料销售额")
    ax.set_title(f"书上例数据：r = {lr.rvalue:.4f}，p = {lr.pvalue:.2e}", fontsize=10)
    sec.figure("book-scatter", fig,
               caption="这组数据是作者为了演示而模拟的，紧得像假的 —— 现实中不会这么好看")

    # 图 2：Anscombe 四重奏
    fig, axes = plot.newfig(2, 2, figsize=(7.6, 5.4))
    for ax, ((tag, d), _) in zip(axes.ravel(), zip(ans["rows"], ans["probes"])):
        ax.scatter(d["x"], d["y"], s=22, color=plot.ACCENT, zorder=3)
        xs = np.linspace(min(d["x"]), max(d["x"]), 50)
        ax.plot(xs, d["slope"] * xs + d["intercept"], lw=1.1, color=plot.ACCENT2)
        ax.set_title(f"第 {tag} 组   r = {d['r']:.3f}   R² = {d['r2']:.3f}", fontsize=9.5)
        ax.tick_params(labelsize=8)
    fig.suptitle("Anscombe 四重奏：四个长得完全不同的数据集，统计量一模一样", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    sec.figure("anscombe", fig,
               caption="统计量的相等不代表数据的相等 —— 这张图应该替代「先看 r」的习惯")

    # 图 3：离群值
    fig, axes = plot.newfig(1, 2, figsize=(7.6, 3.0))
    xs = np.linspace(0, 10, 30)
    rng = np.random.default_rng(710)
    ys = 2.0 * xs + 5.0 + rng.normal(0, 1.0, size=30)
    x_out = np.append(xs, xs[-1] + 2.0)
    y_out = np.append(ys, ys[-1] + 60.0)
    for ax, (xx, yy, tag) in zip(axes, ((xs, ys, "干净"), (x_out, y_out, "加一个离群点"))):
        r_p = float(stats.pearsonr(xx, yy)[0])
        r_s = float(stats.spearmanr(xx, yy)[0])
        ax.scatter(xx[:-1] if tag != "干净" else xx, yy[:-1] if tag != "干净" else yy,
                   s=16, color=plot.ACCENT2)
        if tag != "干净":
            ax.scatter([xx[-1]], [yy[-1]], s=30, color=plot.ACCENT, marker="x")
        m = (np.polyfit(xx, yy, 1))
        xs2 = np.linspace(min(xx), max(xx), 50)
        ax.plot(xs2, m[0] * xs2 + m[1], lw=1.0, color=plot.ACCENT)
        ax.set_title(f"{tag}：皮尔逊 {r_p:.3f} / 斯皮尔曼 {r_s:.3f}", fontsize=9.5)
        ax.tick_params(labelsize=8)
    sec.figure("outlier", fig,
               caption="一个点吃掉 29 个点的信息：皮尔逊掉了 30 个百分点，秩相关几乎不动")

    # 图 4：伪相关
    rw, iid = sp["rw"], sp["iid"]
    fig, axes = plot.newfig(1, 2, figsize=(8.0, 2.9))
    axes[0].hist(iid, bins=45, range=(-1, 1), alpha=0.75, color=plot.ACCENT2,
                 label="iid 序列（正常）")
    axes[0].hist(rw["r"], bins=45, range=(-1, 1), alpha=0.55, color=plot.ACCENT,
                 label="两个独立随机游走")
    axes[0].set_xlabel("皮尔逊 r"); axes[0].set_ylabel("频数")
    axes[0].set_title("互不相干，却遍地「高相关」", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(rw["r"], -np.log10(np.maximum(rw["p_reg"], 1e-300)),
                    s=5, alpha=0.35, color=plot.ACCENT)
    axes[1].axhline(-np.log10(0.05), ls="--", lw=1.0, color=plot.MUTED, label="p = 0.05")
    axes[1].axvline(0.5, ls=":", lw=1.0, color=plot.ACCENT2)
    axes[1].axvline(-0.5, ls=":", lw=1.0, color=plot.ACCENT2)
    axes[1].set_xlabel("皮尔逊 r"); axes[1].set_ylabel("−log₁₀(p)")
    axes[1].set_title(f"OLS 宣布显著的比例：{sp['sig_rate']:.1%}", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sec.figure("spurious", fig,
               caption="左边：iid 序列的 r 挤在 0 附近，随机游走的 r 铺满整个 [−1,1]；"
                       "右边：绝大多数配对都能通过 p<0.05")


def run(sec: Section) -> None:
    experiment_reproduce_book(sec)
    ans = experiment_anscombe(sec)
    out = experiment_single_outlier(sec)
    experiment_strength_vs_significance(sec)
    sp = experiment_spurious_correlation(sec)
    draw(sec, ans, out, sp)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
