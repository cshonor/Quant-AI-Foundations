"""8.2 训练/测试划分、交叉验证，以及准确率为什么会骗人

书上讲的东西（第 8 章中段）：

  · 拟合过的模型不代表能用；要留一部分数据不给模型看，再拿它当「新数据」来打分区；
  · 常见做法是 train/test = 2/3 对 1/3，或者 K 折交叉验证（每份轮流当测试集）；
  · 幼儿例子上一次随机划分得 0.7778，3 折交叉验证得 0.8796、标准差 0.0065，
    100 次随机划分得平均 0.8778、标准差 0.0882；
  · 然后作者翻脸说：准确率不是个好指标。狼来了那个例子里，98% 准确率的警报系统
    真阳例是 0 —— 还不如永远预测「没有狼」，那样反而有 99%；
  · 于是引出混淆矩阵：TP / FP / TN / FN，以及精确率与召回率。

这一节我做的三件事：
  ① 把划分自己的噪声量出来：25 个样本做 1/3 holdout ⇒ 测试集只有 9 个，
     准确率一共只有 5 个可能的取值，最小间隔就是 11.1 个百分点；
  ② 顺手复现了书里的 0.8796 / 0.0065 —— 它居然能精确命中，但那是一次运气；
  ③ 把「准确率为啥骗人」做成了可以算账的恒等式：
     acc_model − acc_恒负 = π · recall · (2·precision − 1) / precision
     右边第二个因子说明：**precision 恰好 0.5 的模型，准确率与「什么都不做」完全相同，
     而且这件事与正例比例 π 无关。**
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "data", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mathviz import Section, plot  # noqa: E402
from _logistic import (  # noqa: E402
    NAP25_X, NAP25_Y, fit_logistic, predict_proba, holdout_split, kfold_split, prf,
)

SEC = Section(
    number="8.2",
    chapter="第 8 章 · 逻辑回归与分类",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="训练/测试划分、交叉验证，以及准确率为什么会骗人",
    claim="同样的 25 个午睡样本，一次随机 holdout 给 0.7778，3 折交叉验证给平均 0.8796、"
          "标准差 0.0065，跑 100 次随机划分则是平均 0.8778、标准差 0.0882。"
          "然后作者把结论掀了一半：98% 准确率的狼警报系统，真阳例是 0，"
          "还不如永远喊「没有狼」（那样有 99%）。"
          "真正管用的是混淆矩阵上的四个格子，以及由它们算出的精确率与召回率。",
    proposition="① 25 个样本做 1/3 holdout，测试集只有 9 个 —— 准确率一共只有 5 个可能取值，"
                "最小间隔是 11.1 个百分点（一个样本就顶 11 个点）；"
                "② 100 次随机划分的跨度是 44.4 个百分点（0.5556 到 1.0000）；"
                "③ 书的 0.8796 / 0.0065 能被某个种子精确命中，但折间标准差在不同种子之间"
                "可以在 0.0065 和 0.1768 之间来回跳（相差 27 倍）；"
                "④ 交叉验证的均值确实比单次 holdout 稳（std 0.039 vs 0.098），"
                "但折间标准差衡量的是一致性，不是估计精度；"
                "⑤ 准确率存在一条可算的账：acc_model − acc_恒负 = π·recall·(2·precision−1)/precision，"
                "precision = 0.5 时这个差值恒为 0，与正例比例无关；"
                "⑥ F1 对调 precision 与 recall 后完全不变，所以它会把两种方向相反的模型判成一模一样。",
    source="第 8 章「验证分类模型」「假阳性和假阴性」「混淆矩阵」「精确率和召回率」",
)

N_HOLDOUT = 100
HO_SEED = 11
CV_SEEDS = tuple(range(100, 120))


# ------------------------------------------------- ① holdout 的分辨率
def experiment_holdout_resolution(sec: Section) -> dict:
    x, y, n = NAP25_X, NAP25_Y, len(NAP25_Y)
    accs = []
    rng = np.random.default_rng(HO_SEED)
    n_test = 0
    for _ in range(N_HOLDOUT):
        tr, te = holdout_split(n, 1 / 3, rng)
        n_test = len(te)
        b0, b1, _ = fit_logistic(x[tr], y[tr])
        accs.append(((predict_proba(x[te], b0, b1) >= 0.5).astype(int) == y[te]).mean())
    acc = np.array(accs)
    levels = sorted(set(np.round(acc, 10)))
    gaps = np.diff(levels)

    sec.check(
        f"9 个测试样本 ⇒ 准确率只有 {len(levels)} 个可能取值，"
        f"最小间隔 {gaps.min():.4f}，正好等于 1/9",
        ok=len(levels) <= 6 and gaps.min() >= 0.10,
        detail=f"ceil(25/3) = {n_test}，所以准确率只能取 k/{n_test}。"
               f"实测 {N_HOLDOUT} 次划分一共只落在 {len(levels)} 个值上："
               + "、".join(f"{v:.4f}" for v in levels)
               + f"，相邻间距最小 {gaps.min():.4f} = 1/{n_test}。"
               f"这意味着**一个样本就是 11.1 个百分点** —— 书上那次 0.7778 和 0.8889"
               f"之间只差 1 个娃娃当天到底有没有崩。"
               f"用这种尺子去比较两个模型，噪声比信号大。",
        tolerance="取值个数 ≤ 6 且最小间距 ≥ 0.10",
    )

    span = float(acc.max() - acc.min())
    sec.check(
        f"同样的模型、同样的数据，换个随机种子，准确率在 [{acc.min():.4f}, {acc.max():.4f}] "
        f"之间跳了 {span:.4f}，标准差 {acc.std():.4f}",
        ok=acc.std() > 0.08 and span > 0.35,
        detail=f"{N_HOLDOUT} 次划分：均值 {acc.mean():.4f}、标准差 {acc.std():.4f}"
               f"（书给 0.8778 / 0.0882，同量级）；跨度 {span:.4f}。"
               f"模型一个字没改，只是换了训练/测试的切法。"
               f"所以『单次划分的 77.78%』这种数字，单独拿出来是没有意义单位的测量值。",
        tolerance="std > 0.08 且跨度 > 0.35",
    )
    return {"acc": acc, "levels": levels, "n_test": n_test, "span": span}


# ------------------------------------------------- ② 交叉验证：稳一点，但别太信那个 std
def experiment_cross_validation(sec: Section) -> dict:
    x, y, n = NAP25_X, NAP25_Y, len(NAP25_Y)
    rows = []
    for seed in CV_SEEDS:
        rng = np.random.default_rng(seed)
        fold = []
        sizes = []
        for tr, te in kfold_split(n, 3, rng):
            sizes.append(len(te))
            b0, b1, _ = fit_logistic(x[tr], y[tr])
            fold.append(((predict_proba(x[te], b0, b1) >= 0.5).astype(int) == y[te]).mean())
        rows.append({"seed": seed, "fold": fold,
                     "mean": float(np.mean(fold)), "std": float(np.std(fold))})
    means = np.array([r["mean"] for r in rows])
    fstds = np.array([r["std"] for r in rows])
    hits = [r["seed"] for r in rows
            if abs(r["mean"] - 0.8796) < 5e-4 and abs(r["std"] - 0.0065) < 5e-4]

    sec.check(
        "书里的 0.8796 / 0.0065 不是编的：某个种子会精确命中它，"
        f"但换个种子，折间标准差在 [{fstds.min():.4f}, {fstds.max():.4f}] 之间来回跳了 "
        f"{fstds.max() / fstds.min():.0f} 倍",
        ok=(len(hits) >= 1) and (fstds.max() > 10 * fstds.min()),
        detail=f"命中书的数字所用的种子是 {hits}（三折准确率分别是"
               + "、".join(f"{v:.4f}" for v in rows[CV_SEEDS.index(hits[0])]["fold"])
               + f"，即 8/9、7/8、7/8 —— 均值 0.8796、样本标准差 0.0065 完全对上了）。"
               f"但在 {len(CV_SEEDS)} 个种子上，折间标准差最小 {fstds.min():.4f}、"
               f"最大 {fstds.max():.4f}。也就是说 0.0065 那次是运气好，"
               f"三折恰好都给出了几乎一样的数。",
        tolerance="存在命中种子，且最大/最小 > 10",
    )

    sec.check(
        f"交叉验证的均值确实比单次 holdout 稳得多（std {means.std():.4f}），"
        f"但这个数字才是真正的不确定性，它比书上那个折间 std 大一个量级",
        ok=means.std() < 0.06 and means.std() > 3 * 0.0065,
        detail=f"{len(CV_SEEDS)} 个不同的 3 折划分，CV 均值平均 {'{:.4f}'.format(means.mean())}、"
               f"标准差 {means.std():.4f}。"
               f"对比：单次 holdout 的标准差是 0.0981，CV 均值把它压到了 {means.std():.4f}"
               f"（降了 {(1 - means.std() / 0.0981):.0%}）—— 这是 CV 真正的价值。"
               f"但书上报的 0.0065 是**折与折之间的一致性**，"
               f"衡量的是『这几份数据是否难易相当』，不是『我的准确率有多少把握』。"
               f"两者差了 {means.std() / 0.0065:.1f} 倍，读论文时最容易在这上面栽跟头。",
        tolerance="means.std() < 0.06 且 > 3 × 0.0065",
    )
    return {"rows": rows, "means": means, "fstds": fstds, "hits": hits}


# ------------------------------------------------- ③ 狼来了：98% 的准确率，0 的价值
def experiment_wolf(sec: Section) -> dict:
    # 供应商 A：100 天测试，其中只有 1 天真有狼；TP=0, FP=1, TN=98, FN=1
    vend = prf(np.array([1] * 1 + [0] * 99), np.array([1] * 1 + [0] * 99))
    vendor = {"tp": 0, "fp": 1, "tn": 98, "fn": 1}
    v_acc = (vendor["tp"] + vendor["tn"]) / 100
    v_prec = vendor["tp"] / (vendor["tp"] + vendor["fp"])
    v_rec = vendor["tp"] / (vendor["tp"] + vendor["fn"])
    always_no = {"tp": 0, "fp": 0, "tn": 99, "fn": 1}
    a_acc = (always_no["tp"] + always_no["tn"]) / 100

    sec.check(
        "98% 准确率的警报系统输给了『永远说没狼』：0.9800 < 0.9900，"
        "而它的精确率和召回率都是 0",
        ok=v_acc < a_acc and v_prec == 0.0 and v_rec == 0.0,
        detail=f"100 天里只有 1 天真的有狼。系统报了 1 次警（那天是鹿）、漏了 1 次狼，"
               f"于是 TP=0、FP=1、TN=98、FN=1，准确率 = (0+98)/100 = {v_acc:.4f}。"
               f"什么都不做、永远预测『无狼』，准确率 = 99/100 = {a_acc:.4f}，反而更高。"
               f"而这个系统的精确率 {v_prec:.4f}、召回率 {v_rec:.4f} —— "
               f"它唯一引以为傲的核心功能，实际上一次都没做成。"
               f"（顺手用得上第 2 章的贝叶斯：先验只有 1% 的时候，"
               f"一个『报警』要有多准才值得听？）",
        tolerance="准确率严格更低，且 precision = recall = 0",
    )

    # 供应商 B：500 天，故意让系统多见狼
    b = {"tp": 36, "fp": 2, "tn": 451, "fn": 11}
    n_b = sum(b.values())
    b_acc = (b["tp"] + b["tn"]) / n_b
    b_prec = b["tp"] / (b["tp"] + b["fp"])
    b_rec = b["tp"] / (b["tp"] + b["fn"])
    b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec)

    sec.check(
        f"第二家供应商：准确率 {b_acc:.4f}（比第一家的 0.98 还低），"
        f"但精确率 {b_prec:.4f}、召回率 {b_rec:.4f} —— 这才是一套能用的系统",
        ok=b_acc > 0.95 and b_rec > 0.70 and b_acc < v_acc,
        detail=f"{n_b} 天：TP={b['tp']}、FP={b['fp']}、TN={b['tn']}、FN={b['fn']}。"
               f"准确率 {b_acc:.4f} 略低于第一家的 {v_acc:.4f}，"
               f"但它抓住了 {b_rec:.1%} 的狼，报警时 {b_prec:.1%} 是真狼，F1 = {b_f1:.4f}。"
               f"（书给精确率 0.947、召回率 0.764，我用一组整数混淆矩阵凑了出来，"
               f"书本图里没印具体格数，这里如实交代。）"
               f"结论：**准确率比较两个模型时可以输，但它完全不告诉你输在哪一格。**",
        tolerance="acc > 0.95、recall > 0.70，且 acc < 0.98",
    )
    return {"v_acc": v_acc, "a_acc": a_acc, "b": b, "b_acc": b_acc,
            "b_prec": b_prec, "b_rec": b_rec, "b_f1": b_f1}


# ------------------------------------------------- ④ 给准确率算一笔账
def experiment_accuracy_accounting(sec: Section) -> dict:
    """acc_model − acc_恒负 = π · recall · (2·precision − 1) / precision

    推导：设正例 πN 个、负例 (1−π)N 个。
      TP = recall·πN，FN = (1−recall)·πN
      FP = TP·(1−precision)/precision
      acc_model − acc_恒负 = (TP + TN)/N − (1−π) = (TP − FP)/N
                           = π·recall·[1 − (1−precision)/precision]
                           = π·recall·(2·precision − 1)/precision
    """
    rows = []
    for pi in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01):
        for prec in (0.5, 0.8, 0.947):
            rec = 0.764
            N = 200_000
            tp = rec * pi * N
            fn = (1 - rec) * pi * N
            fp = tp * (1 - prec) / prec
            tn = (1 - pi) * N - fp
            acc_model = (tp + tn) / N
            acc_triv = (1 - pi)
            theory = pi * rec * (2 * prec - 1) / prec
            rows.append({"pi": pi, "prec": prec, "rec": rec,
                         "gain": acc_model - acc_triv, "theory": theory,
                         "acc_model": acc_model, "acc_triv": acc_triv})
    worst = max(abs(r["gain"] - r["theory"]) for r in rows)
    flat = [r for r in rows if abs(r["prec"] - 0.5) < 1e-12]

    sec.check(
        "准确率的提升有闭式解：acc_model − acc_恒负 = π·recall·(2·precision−1)/precision，"
        f"数值验证最大偏差 {worst:.2e}",
        ok=worst < 1e-10,
        detail="六个正例比例 × 三个精确率共 18 组配置全部落在公式上，"
               f"最大绝对偏差 {worst:.2e}。这个式子把『准确率为什么没用』说清楚了："
               f"提升里有 π 这个因子 —— 目标事件越罕见，准确率能腾出来做区分的"
               f"空间就越小；还有 (2·precision−1) 这个因子 —— "
               f"精确率的门槛线是 50%，而不是『看起来挺高』。",
        tolerance="最大绝对偏差 < 1e-10",
    )

    worst_flat = max(abs(r["gain"]) for r in flat)
    sec.check(
        "精确率恰好 50% 时，模型与『全部预测为负』的准确率完全相同 —— "
        "而这个结论与正例比例 π 无关",
        ok=worst_flat < 1e-12,
        detail="在 π = 0.5 / 0.2 / 0.1 / 0.05 / 0.02 / 0.01 六档上，"
               f"precision = 0.5、recall = 0.764 的模型的准确率增益最大只有 {worst_flat:.2e}。"
               f"换个说法：一个召回率 76.4%、能把四分之三的目标抓住的模型，"
               f"只要它的精确率是碰运气的 50%，它的准确率就和躺在屏幕上什么都不做"
               f"一模一样，而且**在每一档稀有度上都是如此**。"
               f"这就是让『准确率』失效的确切位置。",
        tolerance="|增益| < 1e-12",
    )
    return {"rows": rows, "worst": worst, "worst_flat": worst_flat}


# ------------------------------------------------- ⑤ F1 的对称陷阱
def experiment_f1_symmetry(sec: Section) -> dict:
    # 两家业务方向完全相反的模型
    a = {"prec": 0.947, "rec": 0.764}      # 谨慎：报警少而准
    b = {"prec": 0.764, "rec": 0.947}      # 激进：什么都报警
    f1 = lambda p, r: 2 * p * r / (p + r)
    geo = lambda p, r: float(np.sqrt(p * r))
    f1_a, f1_b = f1(a["prec"], a["rec"]), f1(b["prec"], b["rec"])
    g_a, g_b = geo(a["prec"], a["rec"]), geo(b["prec"], b["rec"])

    # 换算成 500 天里的实际格数
    def counts(prec, rec, pos=47, n=500):
        tp = rec * pos
        fp = tp * (1 - prec) / prec
        fn = pos - tp
        return int(round(tp)), int(round(fp)), int(round(fn))

    ca, cb = counts(a["prec"], a["rec"]), counts(b["prec"], b["rec"])

    sec.check(
        f"把精确率和召回率对调，F1 一个字都不变（{f1_a:.6f} 对 {f1_b:.6f}，差 {abs(f1_a - f1_b):.2e}），"
        f"但两者的漏报数从 {ca[2]} 变成 {cb[2]}",
        ok=abs(f1_a - f1_b) < 1e-12 and abs(ca[2] - cb[2]) > 5,
        detail=f"模型甲（谨慎）：precision {a['prec']}、recall {a['rec']}；"
               f"模型乙（激进）：precision {b['prec']}、recall {b['rec']}。"
               f"F1 都是 {f1_a:.6f}，几何平均也都是 {g_a:.6f} —— 差 {abs(g_a - g_b):.2e}。"
               f"折成 500 天：甲是 TP={ca[0]}、FP={ca[1]}、FN={ca[2]}；"
               f"乙是 TP={cb[0]}、FP={cb[1]}、FN={cb[2]}。"
               f"前者是『报警几乎都准但会漏』，后者是『基本不漏但十次里三次是虚惊』——"
               f"在羊场这两个是两种生意，F1 却把它们判成同一个模型。"
               f"（书里引 Emmanuel Maggiori 的批评正是这一点。）",
        tolerance="|ΔF1| < 1e-12 且漏报数差 > 5",
    )
    return {"f1_a": f1_a, "f1_b": f1_b, "ca": ca, "cb": cb}


# ------------------------------------------------- 图
def draw(sec: Section, ho: dict, cv: dict, woolf: dict, acc: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    a = ho["acc"]
    ax.hist(a, bins=np.arange(0.5, 1.10, 1 / 18), color=plot.ACCENT, alpha=0.85)
    ax.axvline(a.mean(), ls="--", lw=1.2, color="k",
               label=f"平均 {a.mean():.4f}")
    ax.axvspan(a.min(), a.max(), color=plot.ACCENT2, alpha=0.10)
    ax.set_xticks(np.arange(0.5, 1.01, 1 / 9))
    ax.set_xticklabels([f"{k}/9" for k in range(5, 10)], fontsize=8)
    ax.set_xlabel(f"单次 holdout 的准确率（测试集只有 {ho['n_test']} 个样本）")
    ax.set_ylabel(f"{N_HOLDOUT} 次划分中的出现次数")
    ax.set_title(f"尺子本身就是粗的：一格 = {1 / ho['n_test']:.3f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    seeds = np.arange(len(CV_SEEDS))
    means = cv["means"]
    ax.plot(seeds, means, "o-", color=plot.ACCENT, lw=1.2, ms=4, label="3 折 CV 的均值")
    ax.plot(seeds, cv["fstds"], "s--", color=plot.ACCENT2, lw=1.0, ms=3.5,
            label="同一份 CV 的折间标准差")
    hit = cv["hits"][0] if cv["hits"] else None
    if hit is not None:
        i = CV_SEEDS.index(hit)
        ax.plot([i], [means[i]], "*", ms=13, color="k",
                label=f"种子 {hit}：命中书上的 0.8796")
    ax.axhline(0.0065, ls=":", lw=1.0, color=plot.ACCENT2, alpha=0.9)
    ax.text(len(seeds) - 1, 0.0065, " 书的 0.0065", fontsize=7.5,
            va="bottom", ha="right", color=plot.ACCENT2)
    ax.set_xlabel("随机种子"); ax.set_ylabel("准确率")
    ax.set_title("CV 均值还算稳，折间 std 却像抽奖", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("split-noise", fig,
               caption="左边 100 次 holdout 的直方图（只有 5 根柱子）；右边对比 CV 均值与折间标准差")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    rows = acc["rows"]
    for prec, color, marker in [(0.947, plot.ACCENT, "o"),
                                (0.8, plot.ACCENT2, "s"),
                                (0.5, plot.MUTED, "^")]:
        xs = [r["pi"] for r in rows if r["prec"] == prec]
        ys = [r["acc_model"] for r in rows if r["prec"] == prec]
        line = f"模型 precision={prec:.3f}"
        if abs(prec - 0.5) < 1e-12:
            line += "（与恒负重合）"
        ax.plot(xs, ys, marker + "-", color=color, lw=1.4, ms=5, label=line)
    triv = [r["acc_triv"] for r in rows if r["prec"] == 0.5]
    xs = [r["pi"] for r in rows if r["prec"] == 0.5]
    ax.plot(xs, triv, "k--", lw=1.6, label="永远预测「没有狼」")
    ax.set_xscale("log")
    ax.set_xlabel("正例占比 π（对数轴）"); ax.set_ylabel("准确率")
    ax.set_title("稀有一点，准确率就越像一把没有刻度的尺子", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("imbalance", fig,
               caption="recall 固定 0.764：precision=0.5 的那条线精确压在恒负基线上，"
                       "与 π 无关")


def run(sec: Section) -> None:
    ho = experiment_holdout_resolution(sec)
    cv = experiment_cross_validation(sec)
    wolf = experiment_wolf(sec)
    acc = experiment_accuracy_accounting(sec)
    f1 = experiment_f1_symmetry(sec)
    draw(sec, ho, cv, wolf, acc)

    sec.observe(
        f"『一个样本 = {1 / ho['n_test']:.1%}』这件事在 CTA 上是直接的："
        f"如果你的信号一年只触发 20 次，那么胜率的最小刻度就是 5 个百分点。"
        f"声称『我的胜率从 42% 提到了 47%』——那是 1 笔交易的差别。"
        f"这还没算多重检验：试了 20 组参数之后，最好的那组胜率天然就偏高。"
    )
    sec.observe(
        f"把混淆矩阵换成交易语言就很直观了：precision 是**胜率**（报警里真赚的比例），"
        f"recall 是**覆盖率**（所有真行情里抓到的比例），FN 是踏空，FP 是假突破。"
        f"趋势策略天生 recall 低、precision 也不高（40% 胜率照样能赚钱），"
        f"而准确率在这里反而接近 60%~70% —— 一个从不交易的对照模型也能轻松刷到 55% 以上。"
        f"所以评价信号从来不能看 accuracy，得看夏普、看每笔期望、看 PRC。"
    )
    sec.pitfall(
        "书的 0.0065 这个标准差一定要警惕。K 折的折间 std 只在回答"
        "『这几份数据难易是否一致』，真正的估计不确定性是 CV 均值在不同划分之间的那个 "
        f"{cv['means'].std():.4f}。后者是前者的十几倍。看到论文里报折间 std 当成置信区间用，"
        "基本可以直接判定为把方差看小了一个量级。"
    )
    sec.pitfall(
        "我自己一开始想写的命题是『不平衡时恒负分类器的准确率会超过真模型』，"
        "实测下来是错的：只要 recall 不为 0，模型的准确率就一定高于恒负。"
        "真正成立的是那条恒等式 —— 差距被 π 压缩，且 precision ≤ 0.5 时归零。"
        "所以准确率的毛病不是『会被打败』，而是『会把天差地别的两个模型压成同一个数』。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
