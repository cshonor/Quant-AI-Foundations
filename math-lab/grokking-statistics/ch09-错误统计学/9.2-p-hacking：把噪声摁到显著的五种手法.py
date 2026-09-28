"""9.2 p-hacking：测试数据不是无限的，你的自制力才是

书上讲的东西（第 9 章后半段）：

  · **得克萨斯神枪手谬误**：先朝谷仓开枪，再在弹孔周围画靶心。
    关键在于「谷仓门上每一个点被击中的概率都很低，但总有一个点会被击中」；
    真正的技能是**预测哪一个**会发生，而不是事后指出它发生了；
  · 「如果你把数据折磨得够久，它终将招供」——Ronald Coase；
  · p-hacking 的具体手法：只收集够用的数据、把不便的点当离群值剔除、
    挑变量、切子组、试模型参数、**挑随机种子**；
  · 因此机器学习从业者会再留一份**验证集**，作为从未参与迭代的最后一道防线。

这一节我做的是把上面每一条都变成一个能出数的对照实验：

  ① 多重比较：20 个纯噪声变量里至少一个显著的概率，实测与 1−0.95²⁰ 吻合；
  ② 剂量-反应：反复在同一个测试集上挑最好的模型，
     候选数 K 从 1 涨到 1000，测试分数从 0.50 一路涨到 0.75，
     而**同一批模型在全新数据上的分数始终是 0.50**；
  ③ 三层划分是不是真的能修复：用验证集挑完再拿测试集打分，偏差消失了；
  ④ 剔除离群点：允许自己删 5 个点，就有 74.5% 的纯噪声数据集能变得「显著」；
  ⑤ 神枪手谬误的定量版本：试了 K 个变量之后，最显著那个的期望 p 值就是 1/(K+1)。
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
    number="9.2",
    chapter="第 9 章 · 数据犯罪：Statistics Done Wrong",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="p-hacking：测试数据不是有限的，你的自制力才是",
    claim="得克萨斯神枪手谬误：先朝谷仓开枪，再绕着弹孔画靶心。"
          "谷仓门上每一个点被击中的概率都很低，但总有一个点会被击中 —— "
          "真正的技能在于预测是哪一个，而不是事后指出它发生了。"
          "『如果你把数据折磨得够久，它终将招供。』p-hacking 的常见手法包括："
          "只收集够用的数据、把不便的点当离群值剔除、挑变量、切子组、试参数、挑随机种子。"
          "所以从业者会再留一份从未参与迭代的验证集当最后一道防线。",
    proposition="① 20 个纯噪声变量里至少一个 p<0.05 的概率，实测与 1−0.95^20 = 0.6415 吻合；"
                "② 在同一个测试集上反复挑选：候选数从 1 涨到 1000，"
                "被选中模型的测试分数从 0.50 涨到 0.75，而在全新数据上始终是 0.50；"
                "③ 用验证集挑模型、用测试集打分，偏差就消失了（0.4938）；"
                "④ 允许自己删 5 个离群点，74.5% 的纯噪声数据集能变得统计显著；"
                "⑤ 试了 K 个变量后，最显著那个的期望 p 值就是 1/(K+1) —— "
                "与具体是哪个变量毫无关系。",
    source="第 9 章「P-hacking」「得克萨斯神枪手谬误」「过度拟合数据与模型」",
)

N_MULT = 12_000       # 重复次数取大一点，让多重比较那个比例稳到 ±0.5 个百分点
N_VAR = 20
N_TRIAL_LEAK = 200
N_TRIAL_VAL = 250
N_TRIAL_OUT = 600
SEED = 101


def _one_feature_score(x_tr, y_tr, x_te, y_te):
    """单特征『模型』：符号由训练集相关决定，阈值取训练集中位数。

    刻意用最简单的候选模型——p-hacking 的威力来自**搜索次数**，
    而不是模型有多聪明。
    """
    c = np.corrcoef(x_tr, y_tr)[0, 1]
    sgn = np.sign(c) if c != 0 else 1.0
    thr = float(np.median(sgn * x_tr))
    pred = (sgn * x_te >= thr).astype(float)
    return float((pred == y_te).mean())


# ------------------------------------------------- ① 多重比较
def experiment_multiple_comparisons(sec: Section) -> dict:
    rng = np.random.default_rng(SEED)
    hits = 0
    for _ in range(N_MULT):
        ps = []
        for _ in range(N_VAR):
            x = rng.normal(size=30)
            y = rng.normal(size=30)
            ps.append(float(stats.pearsonr(x, y)[1]))
        if min(ps) < 0.05:
            hits += 1
    rate = hits / N_MULT
    theory = 1 - 0.95 ** N_VAR

    sec.check(
        f"{N_VAR} 个纯噪声变量里至少出现一个 p<0.05 的概率是 {rate:.4f}，"
        f"与 1−0.95^{N_VAR} = {theory:.4f} 吻合",
        ok=abs(rate - theory) < 0.02,
        detail=f"重复 {N_MULT} 次「一次测 {N_VAR} 个不相关的变量对」的实验，"
               f"有 {hits} 次（{rate:.2%}）至少撞出一个 p<0.05。"
               f"理论值（假设彼此独立）1−0.95^{N_VAR} = {theory:.4f}，"
               f"两者相差 {abs(rate - theory):.4f}。"
               f"注意这意味着：**『20 个变量里有 1 个显著』根本不是发现，"
               f"它恰好就是 α=0.05 的定义本身。**"
               f"想要第 20 个变量仍然诚实地只有 5% 的误报率，"
               f"就得用 Bonferroni 之类的修正把门槛压到 0.05/20 = 0.0025。",
        tolerance="|实测 − 理论| < 0.02",
    )
    return {"rate": rate, "theory": theory, "hits": hits}


# ------------------------------------------------- ②  dose-response：搜索强度 ⇢ 虚高多少
def experiment_test_set_leakage(sec: Section) -> dict:
    rows = []
    for K in (1, 10, 100, 1000):
        old, fresh = [], []
        for s in range(N_TRIAL_LEAK):
            rng = np.random.default_rng(1000 + s)
            ntr, nte = 60, 40
            Xtr = rng.normal(size=(ntr, K))
            ytr = (rng.random(ntr) < 0.5).astype(float)
            Xte = rng.normal(size=(nte, K))
            yte = (rng.random(nte) < 0.5).astype(float)
            Xfresh = rng.normal(size=(nte, K))
            yfresh = (rng.random(nte) < 0.5).astype(float)
            accs = np.array([
                _one_feature_score(Xtr[:, j], ytr, Xte[:, j], yte) for j in range(K)
            ])
            fresh_accs = np.array([
                _one_feature_score(Xtr[:, j], ytr, Xfresh[:, j], yfresh) for j in range(K)
            ])
            best = int(np.argmax(accs))            # ← 按旧测试集的分数挑，这就是泄漏
            old.append(accs[best])
            fresh.append(fresh_accs[best])
        rows.append({"K": K, "old": float(np.mean(old)), "fresh": float(np.mean(fresh))})

    k1 = next(r for r in rows if r["K"] == 1)
    kk = next(r for r in rows if r["K"] == 1000)
    sec.check(
        f"在同一个测试集上反复挑：候选数 K=1 时选中模型的测试准确率是 {k1['old']:.4f}，"
        f"K=1000 时涨到 {kk['old']:.4f}；而同一批模型在**从未见过的新数据**上，"
        f"始终只有 {kk['fresh']:.4f}",
        ok=(kk["old"] > 0.70 and abs(kk["fresh"] - 0.5) < 0.03
            and all(rows[i]["old"] < rows[i + 1]["old"] for i in range(len(rows) - 1))),
        detail="K = 候选变量数（研究员『试过多少种解释』）：\n"
               + "\n".join(
                   f"  · K={r['K']:<5} 旧测试集 {r['old']:.4f} / 全新数据 {r['fresh']:.4f}"
                   for r in rows)
               + f"\n全部数据都是标准正态噪声与掷硬币标签，**没有任何一个模型有真实预测力**。"
               f"\n但光是『多试几次并留下最好看的那个』，就把 0.50 刷到了 {kk['old']:.4f}。"
               f"更阴险的地方在于全新数据上那 {kk['fresh']:.4f} —— "
               f"被挑出来的那个模型不会因为被选中而变好，"
               f"**是那 40 个测试样本被它的挑选过程污染了。**"
               f"这正是书上说的『训练/测试划分也会被滥用』："
               f"测试集只在**你只看一次**的时候才是无偏的。",
        tolerance="K=1000 时旧分数 > 0.70、新数据 |−0.5| < 0.03，且随 K 单调递增",
    )
    return {"rows": rows, "k1": k1, "kk": kk}


# ------------------------------------------------- ③ 三层划分能不能修
def experiment_validation_set(sec: Section, leak_old: float) -> dict:
    K = 1000
    ntr, nva, nte = 50, 40, 40
    val_scores, test_scores = [], []
    for s in range(N_TRIAL_VAL):
        rng = np.random.default_rng(2000 + s)
        Xtr = rng.normal(size=(ntr, K))
        ytr = (rng.random(ntr) < 0.5).astype(float)
        Xva = rng.normal(size=(nva, K))
        yva = (rng.random(nva) < 0.5).astype(float)
        Xte = rng.normal(size=(nte, K))
        yte = (rng.random(nte) < 0.5).astype(float)
        va = np.array([_one_feature_score(Xtr[:, j], ytr, Xva[:, j], yva) for j in range(K)])
        te = np.array([_one_feature_score(Xtr[:, j], ytr, Xte[:, j], yte) for j in range(K)])
        best = int(np.argmax(va))                  # ← 改成用验证集挑
        val_scores.append(va[best])
        test_scores.append(te[best])
    val_mean, test_mean = float(np.mean(val_scores)), float(np.mean(test_scores))

    sec.check(
        f"换成三层划分（用**验证集**挑选），同样 K={K}、同样纯噪声："
        f"验证集上是 {val_mean:.4f}，而从未参与任何挑选的测试集上只剩 {test_mean:.4f}",
        ok=abs(test_mean - 0.5) < 0.03 and val_mean > 0.70,
        detail=f"模型的筛选完全由验证集决定（{N_TRIAL_VAL} 次独立重复）。"
               f"验证集自己仍然虚高到 {val_mean:.4f} —— **它已经被挑选过程用掉了**；"
               f"但测试集一路上什么都没参与，分数回落到 {test_mean:.4f}，"
               f"与掷硬币的 0.5 无法区分。"
               f"对比上一项里『用测试集挑』的 {leak_old:.4f}："
               f"**同一个搜索过程，换个数据集来记录，差了 {leak_old - test_mean:.4f}。**"
               f"这就是书上说『从业者可能会再保留一个验证数据集』的确切原因 —— "
               f"不是流程更繁琐，而是每被挑选一次，那份数据就不再是证据了。",
        tolerance="|测试集 − 0.5| < 0.03 且验证集 > 0.70",
    )
    return {"val": val_mean, "test": test_mean, "K": K}


# ------------------------------------------------- ④ 剔除离群点
def experiment_outlier_pruning(sec: Section) -> dict:
    n, kmax = 40, 5
    rec: dict[int, list[float]] = {k: [] for k in range(kmax + 1)}
    for s in range(N_TRIAL_OUT):
        rng = np.random.default_rng(3000 + s)
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        keep = np.arange(n)
        for k in range(kmax + 1):
            xx, yy = x[keep], y[keep]
            rec[k].append(float(stats.pearsonr(xx, yy)[1]))
            if len(keep) <= n - kmax:
                break
            # 贪心：每次剔除「删掉它之后 |r| 最大」的那个点
            gains = [abs(float(stats.pearsonr(x[np.delete(keep, i)],
                                              y[np.delete(keep, i)])[0]))
                     for i in range(len(keep))]
            keep = np.delete(keep, int(np.argmax(gains)))
    rows = [{"k": k, "rate": float(np.mean(np.array(rec[k]) < 0.05))}
            for k in range(kmax + 1)]

    sec.check(
        f"允许自己删 5 个『异常点』，就有 {rows[-1]['rate']:.1%} 的纯噪声数据集"
        f"能熬到 p < 0.05 —— 而一个点都不删时是 {rows[0]['rate']:.1%}",
        ok=rows[0]["rate"] < 0.07 and rows[-1]["rate"] > 0.65,
        detail=f"每一步都用贪心策略：删掉那个能让 |r| 涨得最多的点（这就是人实际会做的事）。"
               f"{N_TRIAL_OUT} 个完全无关的数据集上的显著率：\n"
               + "\n".join(f"  · 删除 {r['k']} 个点 → {r['rate']:.1%}" for r in rows)
               + f"\n一个点都不删时是 {rows[0]['rate']:.1%}，正好复现名义的 5%，"
                 f"这说明检验本身是对的 —— **出问题的从来不是 p 值，"
                 f"是那条被藏起来的、通向最后一次删除的路径。**"
                 f"  Ronald Coase 那句『把数据折磨得够久，它终将招供』在这里是可以算出来的："
                 f"只删 5 个点就招了 {rows[-1]['rate']:.0%}。"
                 f"（书里问『我什么时候可以移除异常值』，答案是："
                 f"**在看到结果之前就想好规则，并且把删掉的点连同理由一起报出来。**）",
        tolerance="删 0 个时 < 7%，删 5 个时 > 65%",
    )
    return {"rows": rows}


# ------------------------------------------------- ⑤ 神枪手：min p 的期望位置
def experiment_min_p(sec: Section) -> dict:
    rows = []
    for K in (10, 100, 1000):
        mins = []
        for s in range(400):
            rng = np.random.default_rng(4000 + s)
            ps = [float(stats.pearsonr(rng.normal(size=30), rng.normal(size=30))[1])
                  for _ in range(K)]
            mins.append(min(ps))
        mins = np.array(mins)
        rows.append({"K": K, "mean": float(mins.mean()),
                     "median": float(np.median(mins)),
                     "theory_mean": 1.0 / (K + 1),
                     "theory_median": 1 - 0.5 ** (1.0 / K)})

    worst = max(abs(r["mean"] - r["theory_mean"]) / r["theory_mean"] for r in rows)
    sec.check(
        "神枪手谬误的定量版本：试了 K 个变量之后，最显著那个的期望 p 值就是 1/(K+1) —— "
        "与它是哪个变量毫无关系",
        ok=worst < 0.15,
        detail="\n".join(
            f"  · K={r['K']:<5} E[min p] 实测 {r['mean']:.6f} / 理论 1/(K+1) = {r['theory_mean']:.6f}"
            f"；中位数实测 {r['median']:.6f} / 理论 {r['theory_median']:.6f}"
            for r in rows)
        + f"\n最大相对偏差 {worst:.1%}。"
        f"\n向谷仓开 K 枪，再绕着最深的那个弹孔画靶 —— "
        f"画出来的靶心有多小，**只取决于你开了几枪**。"
        f"K=1000 时平均能拿到 p ≈ 0.001，听起来是千载难逢的证据，"
        f"但在 1000 次尝试里它恰好就是期望值。"
        f"（这也给出了一条操作性的自查：报告 min p 的时候，"
        f"必须同时报告 K。）",
        tolerance="相对偏差 < 15%",
    )
    return {"rows": rows, "worst": worst}


# ------------------------------------------------- 图
def draw(sec: Section, leak: dict, val: dict, out: dict, minp: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ks = [str(r["K"]) for r in leak["rows"]]
    old = [r["old"] for r in leak["rows"]]
    fresh = [r["fresh"] for r in leak["rows"]]
    xa = np.arange(len(ks))
    ax.plot(xa, old, "o-", lw=1.8, ms=6, color=plot.ACCENT, label="报告口：旧测试集")
    ax.plot(xa, fresh, "s-", lw=1.8, ms=6, color=plot.ACCENT2, label="真相：全新数据")
    ax.axhline(0.5, ls="--", lw=1.0, color="k", alpha=0.6)
    ax.text(len(xa) - 1, 0.515, "掷硬币 0.50", fontsize=8, ha="right")
    for i, (o, f) in enumerate(zip(old, fresh)):
        ax.annotate("", xy=(i, o), xytext=(i, f),
                    arrowprops=dict(arrowstyle="-", color=plot.MUTED, lw=1.2))
    ax.set_xticks(xa); ax.set_xticklabels(ks)
    ax.set_xlabel("候选模型个数 K（你试了多少次）")
    ax.set_ylabel("准确率")
    ax.set_title("没有模型变强过，只是箭头被拉长了", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ok_ = [r["k"] for r in out["rows"]]
    rr = [r["rate"] for r in out["rows"]]
    ax.plot(ok_, rr, "o-", lw=1.8, ms=6, color=plot.ACCENT)
    ax.axhline(0.05, ls="--", lw=1.1, color="k", alpha=0.6)
    ax.text(max(ok_), 0.075, "名义误报率 0.05", fontsize=8, ha="right")
    for k, v in zip(ok_, rr):
        ax.annotate(f"{v:.1%}", (k, v), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.5)
    ax.set_xlabel("允许自己剔除的点数"); ax.set_ylabel("能熬到 p < 0.05 的比例")
    ax.set_title("数据是清白的，路径不是", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("p-hacking", fig,
               caption="左边是搜索强度与虚高的剂量-反应曲线；右边是剔除离群点的分叉路径")


def run(sec: Section) -> None:
    multi = experiment_multiple_comparisons(sec)
    leak = experiment_test_set_leakage(sec)
    val = experiment_validation_set(sec, leak["kk"]["old"])
    out = experiment_outlier_pruning(sec)
    minp = experiment_min_p(sec)
    draw(sec, leak, val, out, minp)

    sec.observe(
        f"第 ② 项那个 {leak['kk']['old']:.4f} 是全章最值得记住的数字之一："
        f"它是在**没有人作弊**的前提下产生的 —— 每个模型都老老实实用训练集拟合，"
        f"每个准确率都算得很规范。唯一的动作是「看了一眼再决定报告哪一个」。"
        f"CTA 上这就对应：跑 1000 组参数、挑回测夏普最高的那组。"
        f"它的样本外期望和你第一次随手试的那组**完全一样**，"
        f"但账面会好看 {leak['kk']['old'] - 0.5:.2f}。"
    )
    sec.observe(
        f"三层划分那项给了唯一的解药：测试集上 {val['test']:.4f}。"
        f"注意代价 —— 验证集自己变成了 {val['val']:.4f} 的废纸。"
        f"所以样本是**消耗品**：每被你看一次它对证据的贡献就少一层。"
        f"实际操作建议：把所有「看了结果的决策」都关在时间切分的最早那一段里，"
        f"最后那段只允许做一件事：报数，然后不许再改任何东西。"
    )
    sec.pitfall(
        "书上把六种 p-hacking 手法并列放在一起，其实它们的危害不是一个量级。"
        "挑随机种子、切子组这类是纯粹的噪声挖掘；"
        "而『剔除离群点』本身是合法操作（书中自己给了通宵开车的例子）。"
        "分界线只有一条：**规则是否写在看到结果之前**。"
        f"同样删 5 个点，事前声明的是数据清洗，事后挑选的是 {out['rows'][-1]['rate']:.0%} 的造假。"
    )
    sec.pitfall(
        "别以为多重比较修正能兜住一切。Bonferroni 只对『事先列好的 K 个假设』有效，"
        "而真实分析里 K 往往是事后才数得清的（试了多少种子、砍了多少行数据、"
        "换过几个窗口）。第 ④ 项那个 74.5% 就是这么来的 —— "
        "它不是 20 次独立检验，是一条事后才被发现的分叉路径。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
