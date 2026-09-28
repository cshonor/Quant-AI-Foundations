"""8.3 阈值从哪里切：ROC、PRC，以及为什么不平衡时 AUC 会装作没事

书上讲的东西（第 8 章末段）：

  · 阈值是可以调的旋钮。想少担风险就把阈值压到 0.30（召回变高、精确变低），
    想多办事就把阈值抬高（精确变高、召回变低）；
  · 于是每个阈值都有一张不同的混淆矩阵，需要一次看完全貌 —— 这就是 ROC 曲线：
    横轴假阳性率 FPR = FP/(FP+TN)，纵轴真阳性率 TPR = TP/(TP+FN)；
  · ROC 下面积 AUC 把性能压成一个数；没有预测能力的模型沿着对角线，AUC = 0.5；
  · 书自己的运行给 AUC = 0.9774、AP = 0.9658；
  · 但是！不平衡数据上 ROC 会失真，这时要改用精确率-召回率曲线（PRC），
    它的「无技巧」基线是一条横线 y = 正例占比，而不是 0.5。

这一节我自己也算了一遍 ROC/PRC（没有 sklearn，全手写的），并且把一个命题
做成了可以判决的实验：把负样本复制 100 倍，模型的**排序能力一个字没变**
（AUC 小数点后六位都还是 0.924908），但它的实用性全线崩塌
（AP 0.8431 → 0.4065，Brier 0.1137 → 0.1460）。书上说「不平衡就用 PRC」，
这里给出了它崩塌的具体数字。
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
    NAP100_X, NAP100_Y, fit_logistic, predict_proba, holdout_split,
    roc_prc, auc_trapezoid, auc_mannwhitney, average_precision, at_threshold,
)

SEC = Section(
    number="8.3",
    chapter="第 8 章 · 逻辑回归与分类",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="阈值从哪里切：ROC、PRC，以及为什么不平衡时 AUC 会装作没事",
    claim="阈值是个旋钮：压到 0.30 召回变高、精确变低，抬高则反之。"
          "每个阈值一张混淆矩阵，于是用 ROC 曲线一次看全（横 FPR、纵 TPR），"
          "再用曲线下面积 AUC 压成一个数；无技能模型沿对角线走，AUC = 0.5。"
          "但数据不平衡时 ROC 会失真，要换成精确率-召回率曲线，"
          "它的无技巧基线是 y = 正例占比这条横线。",
    proposition="① AUC 的两种算法是同一件事：梯形面积与 Mann–Whitney 秩和分毫不差；"
                "② 34 个测试样本上，随机标签的 AUC 中心在 0.50，但 95% 区间宽达 0.40 —— "
                "小测试集上 AUC 不到 0.72 都不能算『比随机强』；"
                "③ 阈值确实在做交换：0.20 给 recall 1.00/precision 0.65，"
                "0.90 给 recall 0.69/precision 0.82；"
                "④ Youden J 最优点落在阈值 0.4271，换算回业务语言是 2.92 小时；"
                "⑤ 把负样本复制 100 倍，AUC 小数点后六位纹丝不动（0.924908），"
                "而 AP 从 0.8431 掉到 0.4065 —— 这就是『不平衡时用 PRC』的确切含义；"
                "⑥ 同一个 AUC 底下，概率已经不校准了：Brier 从 0.1137 涨到 0.1460；"
                "⑦ PRC 的无技巧基线就是正例占比本身，与输入 x 无关。",
    source="第 8 章「实践中的验证」「ROC 曲线」「PRC」",
)

TEST_SEED = 3
NULL_SEED = 9
N_NULL = 300
IMB_K = (1, 2, 5, 10, 30, 100)


# ------------------------------------------------- ① 两套 AUC 算法互相验算
def fit_and_score():
    """用 TEST_SEED 做一次 1/3 holdout，返回测试集上的标签与概率。"""
    tr, te = holdout_split(len(NAP100_Y), 1 / 3, np.random.default_rng(TEST_SEED))
    b0, b1, _ = fit_logistic(NAP100_X[tr], NAP100_Y[tr])
    return te, NAP100_Y[te], predict_proba(NAP100_X[te], b0, b1), (b0, b1)


def experiment_auc_two_definitions(sec: Section, y_te, p_te) -> dict:
    fpr, tpr, prec, rec, thr = roc_prc(y_te, p_te)
    auc_area = auc_trapezoid(fpr, tpr)
    auc_rank = auc_mannwhitney(y_te, p_te)

    sec.check(
        "AUC 的两种算法是同一件事：梯形积分与 Mann–Whitney 秩和分毫不差"
        f"（{auc_area:.6f} 对 {auc_rank:.6f}，差 {abs(auc_area - auc_rank):.2e}）",
        ok=abs(auc_area - auc_rank) < 1e-10,
        detail=f"几何算法（ROC 曲线下的梯形面积）= {auc_area:.10f}；"
               f"概率算法（随机取一个正例和一个负例，正例得分更高的概率，"
               f"用平均秩处理并列）= {auc_rank:.10f}。"
               f"两者相差 {abs(auc_area - auc_rank):.2e}。"
               f"第二个定义解释了一切：AUC 只关心**排序**，"
               f"既不看概率的绝对值，也不看正负样本各自有多少 —— "
               f"这正是它稍后会在不平衡数据上『装作没事』的原因。"
               f"（书上一次运行给 0.9774，我这里换了个划分给 {auc_area:.4f}，"
               f"34 个测试样本上换个划分差几个百分点是正常的。）",
        tolerance="绝对差 < 1e-10",
    )
    return {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec, "thr": thr,
            "auc": auc_area, "auc_rank": auc_rank,
            "ap": average_precision(y_te, p_te)}


# ------------------------------------------------- ② AUC = 0.5 到底要多大才算
def experiment_null_distribution(sec: Section, y_te, p_te, auc_true: float) -> dict:
    rng = np.random.default_rng(NULL_SEED)
    null = []
    for _ in range(N_NULL):
        ys = rng.permutation(y_te)
        f2, t2, _, _, _ = roc_prc(ys, p_te)
        null.append(auc_trapezoid(f2, t2))
    null = np.array(null)
    lo, hi = float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
    z = (auc_true - null.mean()) / null.std()

    sec.check(
        f"把标签打乱 {N_NULL} 次：AUC 的零分布中心是 {null.mean():.4f}，"
        f"但 95% 区间宽达 [{lo:.4f}, {hi:.4f}] —— 这份测试集上不到 0.72 都不算证据",
        ok=abs(null.mean() - 0.5) < 0.05 and lo <= 0.5 <= hi and (hi - lo) > 0.30,
        detail=f"零分布均值 {null.mean():.4f}（理论 0.5）、标准差 {null.std():.4f}。"
               f"测试集只有 {len(y_te)} 个样本（其中 {int(y_te.sum())} 个正例），"
               f"AUC 自身的抽样噪声就有 ±{1.96 * null.std():.3f}。"
               f"所以在这个数据规模上，一个 0.70 的 AUC 完全落在噪声里，"
               f"是不足以宣称模型有用的 —— 这一点和 8.2 那个『9 个样本只有 5 档』是同一件事。"
               f"真实标签的 AUC = {auc_true:.4f}，偏离零分布中心 {z:.1f} 个标准差，这个才是信号。",
        tolerance="|均值−0.5| < 0.05、区间含 0.5、宽度 > 0.30",
    )
    return {"null": null, "lo": lo, "hi": hi, "z": z}


# ------------------------------------------------- ③ 阈值这根旋钮
def experiment_threshold_knob(sec: Section, y_te, p_te) -> dict:
    grid = [0.20, 0.30, 0.4271, 0.50, 0.70, 0.90]
    stats = {t: at_threshold(y_te, p_te, t) for t in grid}
    low, high = stats[0.20], stats[0.90]

    sec.check(
        f"阈值确实在交换精确率与召回率：0.20 时 recall {low['recall']:.3f} / "
        f"precision {low['precision']:.3f}，0.90 时回忆 {high['recall']:.3f} / "
        f"precision {high['precision']:.3f}，两者反向",
        ok=(low["recall"] > high["recall"]) and (low["precision"] < high["precision"]),
        detail=f"阈值 0.20：(TP={low['tp']}, FP={low['fp']}, FN={low['fn']})，"
               f"宁可错杀——放过了 0 次真崩溃，代价是 {low['fp']} 次白跑回家；"
               f"阈值 0.90：(TP={high['tp']}, FP={high['fp']}, FN={high['fn']})，"
               f"报得很少但准，代价是漏掉 {high['fn']} 次真崩溃。"
               f"准确率在这两端分别是 {low['accuracy']:.4f} 和 {high['accuracy']:.4f}，"
               f"看起来差不多，业务上一个是『每天早退』一个是『当众社死』。"
               f"（顺带说明了 8.2 的结论：准确率区分不了这两个模型。）",
        tolerance="recall 与 precision 同时反向变化",
    )

    # Youden J
    fpr, tpr, _, _, thr = roc_prc(y_te, p_te)
    J = tpr - fpr
    i = int(np.argmax(J))
    thr_star = float(thr[i])

    sec.check(
        f"Youden J = TPR − FPR 的最大值出现在阈值 {thr_star:.4f}"
        f"（TPR {tpr[i]:.3f}、FPR {fpr[i]:.3f}，J = {J[i]:.4f}）",
        ok=0.2 < thr_star < 0.8 and J[i] > 0.7,
        detail=f"ROC 上一共只有 {len(thr)} 个不同的阈值（= 测试集样本数），"
               f"所以 ROC 是一条阶梯而不是平滑曲线。"
               f"最优的那一级在阈值 {thr_star:.4f}。"
               f"注意默认的 0.5 和它落在同一个平台上"
               f"（之间的样本一个都没有跨过去），给出完全相同的混淆矩阵 —— "
               f"这也是小样本上『选阈值』这件事本身就很粗的原因。",
        tolerance="0.2 < thr* < 0.8 且 J > 0.7",
    )
    return {"stats": stats, "grid": grid, "thr_star": thr_star,
            "J": J, "i_star": i, "fpr": fpr, "tpr": tpr}


# ------------------------------------------------- ④ 不平衡：AUC 装作没事，AP 不
def experiment_imbalance(sec: Section, y_te, p_te, baseline: dict) -> dict:
    pos = p_te[y_te == 1]
    neg = p_te[y_te == 0]
    rows = []
    for k in IMB_K:
        yy = np.concatenate([np.ones(len(pos)), np.zeros(len(neg) * k)])
        ss = np.concatenate([pos, np.tile(neg, k)])
        f2, t2, _, _, _ = roc_prc(yy, ss)
        rows.append({
            "k": k, "prev": float(yy.mean()),
            "auc": auc_trapezoid(f2, t2),
            "ap": average_precision(yy, ss),
            "brier": float(np.mean((ss - yy) ** 2)),
        })
    aucs = np.array([r["auc"] for r in rows])
    aps = np.array([r["ap"] for r in rows])
    briers = np.array([r["brier"] for r in rows])
    spread = float(aucs.max() - aucs.min())
    drop = float(aps[0] - aps[-1])
    worst_prev = rows[-1]["prev"]

    sec.check(
        f"把负样本复制 100 倍（正例占比从 {rows[0]['prev']:.4f} 掉到 {worst_prev:.4f}），"
        f"AUC 一个字没变（波动 {spread:.2e}），AP 却从 {aps[0]:.4f} 崩到 {aps[-1]:.4f}",
        ok=spread < 1e-9 and drop > 0.30,
        detail="六档复制倍率上的完整结果（模型一次都没重训，只是重复了负样本）：\n"
               + "\n".join(
                   f"  · 负样本 ×{r['k']:<3} 正例占比 {r['prev']:.4f} → "
                   f"AUC {r['auc']:.6f} / AP {r['ap']:.4f} / Brier {r['brier']:.4f}"
                   for r in rows)
               + f"\nAUC 全程是同一个数（极差 {spread:.2e}），因为AUC 只数"
                 f"『正负样本的成对排序』，复制负样本不会改变任何一对的先后；"
                 f"而 AP 要把 precision 摊到每个 recall 上，"
                 f"precision 分母里的 FP 随着负样本被放大 → 曲线整体塌下来。"
                 f"这就是书上那句『如果不平衡就用 PRC』的确切含义，"
                 f"而且是可以直接量出来的：AUC 完全没反应，AP 掉了 {drop:.4f}。",
        tolerance="AUC 极差 < 1e-9 且 AP 跌幅 > 0.30",
    )

    sec.check(
        "AUC 不变 ≠ 概率还能用：同一个排序能力底下，Brier 从 "
        f"{briers[0]:.4f} 涨到 {briers[-1]:.4f}（概率已经系统性偏高）",
        ok=briers[-1] > briers[0] * 1.2,
        detail=f"模型输出的概率是按原始 prevalence {rows[0]['prev']:.4f} 校准的；"
               f"把 prevalence 压到 {worst_prev:.4f} 之后，它还在用老口径报概率，"
               f"于是平方误差（Brier）上升 {(briers[-1] / briers[0] - 1):.0%}。"
               f"而 AUC 一动不动。**排序对、数值错** —— "
               f"这在量化上是致命的组合：用它做排序选股没问题，"
               f"用它估计胜率、算仓位、做 Kelly 就全错。"
               f"（修正办法是把截距平移 ln(π'/(1−π')) − ln(π/(1−π))，"
               f"本书没讲，留作练习。）",
        tolerance="Brier 涨幅 > 20%",
    )
    return {"rows": rows, "spread": spread, "drop": drop}


# ------------------------------------------------- ⑤ PRC 的无技巧基线
def experiment_prc_baseline(sec: Section, y_te, p_te) -> dict:
    rng = np.random.default_rng(NULL_SEED + 1)
    # 常数预测：无论输入什么都报同一个概率
    const_aucs, const_aps = [], []
    const_val = float(y_te.mean())
    const_scores = np.full(len(y_te), const_val)
    f2, t2, _, _, _ = roc_prc(y_te, const_scores)
    const_aucs.append(auc_trapezoid(f2, t2))
    const_aps.append(average_precision(y_te, const_scores))

    # 随机打分：作为「随机」的对照组
    rand_ap = []
    for _ in range(200):
        s = rng.random(len(y_te))
        rand_ap.append(average_precision(y_te, s))
    rand_mean = float(np.mean(rand_ap))
    # 随机排序下 AP 的理论近似：(n_pos+1)/(n+1)，再由 Jensen 上偏
    n_pos = int(y_te.sum())
    approx = (n_pos + 1) / (len(y_te) + 1)

    sec.check(
        f"PRC 的无技巧基线就是正例占比本人：常数预测 {const_val:.4f} 的 AP 恰好是 "
        f"{const_aps[0]:.4f}；而我原以为随机打分会低于它，实测反而高 — "
        f"随机打分平均 AP = {rand_mean:.4f}",
        ok=abs(const_aps[0] - const_val) < 1e-9 and rand_mean > const_val,
        detail=f"拿一个恒定输出 {const_val:.4f} 的模型去算 PRC：AP = {const_aps[0]:.4f}，"
               f"正好等于 prevalence = {const_val:.4f} —— "
               f"因为此时任何阈值下要么全预测正、要么全预测负，"
               f"precision 恒等于全体的正例占比。"
               f"（它的 ROC-AUC 是 {const_aucs[0]:.4f}，退化成对角线。）"
               f"**但我原本的判据写反了**：我猜随机打分会落在 0.5 以下的这条基线之下，"
               f"实测 200 次随机打分的平均 AP 是 {rand_mean:.4f}，反而比基线高 "
               f"{rand_mean - const_val:.4f}。原因是 AP 等价于 "
               f"『每个正例所在排名处的 precision 之平均』Σ(i/kᵢ)/n_pos，"
               f"而 E[i/kᵢ] > i/E[kᵢ]（1/x 是凸函数，Jensen 不等式）"
               f"—— 随机排序的位置波动会把这个面积往上抬。"
               f"粗略的近似 (n_pos+1)/(n+1) = {approx:.4f} 已经接近实测。"
               f"结论：**PRC 的虚线比『随机』还要严格**，"
               f"一个完全随机打分的模型能画出比基线更好的曲线，"
               f"读 PRC 时不能直接拿柱子跟虚线比。",
        tolerance="AP − prevalence < 1e-9，且随机 AP > prevalence",
    )
    return {"const_val": const_val, "const_ap": const_aps[0], "rand_ap": rand_mean,
            "approx": approx}


# ------------------------------------------------- 图
def draw(sec: Section, base: dict, knob: dict, imb: dict, prcbase: dict,
         nul: dict, y_te: np.ndarray, p_te: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.plot(base["fpr"], base["tpr"], "o-", color=plot.ACCENT, lw=1.6, ms=3.5,
            label=f"模型 ROC（AUC = {base['auc']:.4f}）")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, label="无技能：AUC = 0.5")
    i = knob["i_star"]
    ax.plot([knob["fpr"][i]], [knob["tpr"][i]], "*", ms=15, color="k",
            label=f"Youden J 最优点（ΔJ = {knob['J'][i]:.3f}）")
    ax.set_xlabel("假阳性率 FPR"); ax.set_ylabel("真阳性率 TPR")
    ax.set_title("ROC：TPR 与 FPR 的兑换表", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.plot(base["rec"], base["prec"], "o-", color=plot.ACCENT2, lw=1.6, ms=3.5,
            label=f"模型 PRC（AP = {base['ap']:.4f}）")
    ax.plot([0, 1], [prcbase["const_val"]] * 2, "k--", lw=1.0,
            label=f"无技能基线 = 正例占比 {prcbase['const_val']:.4f}")
    ax.set_xlabel("召回率 Recall"); ax.set_ylabel("精确率 Precision")
    ax.set_title("PRC：书上说不平衡时该看这张", fontsize=10)
    ax.set_ylim(0.0, 1.05); ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("roc-prc", fig,
               caption="左 ROC、右 PRC，都来自同一次划分（seed=3）的测试数据")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    thr_grid = np.linspace(0.05, 0.99, 190)
    curves = {"precision": [], "recall": [], "f1": [], "accuracy": []}
    for t in thr_grid:
        d = at_threshold(y_te, p_te, t)
        for k in curves:
            curves[k].append(d[k])
    ax.plot(thr_grid, curves["precision"], lw=1.6, color=plot.ACCENT, label="精确率")
    ax.plot(thr_grid, curves["recall"], lw=1.6, color=plot.ACCENT2, label="召回率")
    ax.plot(thr_grid, curves["accuracy"], lw=1.2, color=plot.MUTED, ls=":", label="准确率")
    ax.axvline(0.5, ls="--", lw=1.0, color="k", alpha=0.5)
    ax.text(0.5, 0.02, " 默认 0.5", fontsize=7.5, color="k")
    ax.axvline(knob["thr_star"], ls=":", lw=1.2, color=plot.ACCENT)
    ax.text(knob["thr_star"], 0.02, f" J 最优 {knob['thr_star']:.3f}",
            fontsize=7.5, color=plot.ACCENT)
    ax.set_xlabel("阈值"); ax.set_ylabel("指标")
    ax.set_title("同一模型，换个阈值就是另一笔生意", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    null = nul["null"]
    ax.hist(null, bins=26, color=plot.MUTED, alpha=0.55, label="标签打乱后的 AUC")
    ax.axvline(0.5, ls="--", lw=1.2, color="k", label="理论 0.5")
    ax.axvspan(nul["lo"], nul["hi"], color=plot.MUTED, alpha=0.18)
    ax.axvline(base["auc"], lw=2.0, color=plot.ACCENT,
               label=f"真实标签 AUC = {base['auc']:.4f}")
    ax.set_xlabel("AUC"); ax.set_ylabel(f"{N_NULL} 次中的出现次数")
    ax.set_title(f"随机也能刷到 {nul['hi']:.2f}：这条才是噪声的尺码", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("threshold-tradeoff", fig,
               caption="左边是阈值旋钮；右边是 AUC 的零分布，真实值离它很远才算数")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    prev = [r["prev"] for r in imb["rows"]]
    ax.semilogx(prev, [r["auc"] for r in imb["rows"]], "o-", color=plot.ACCENT,
                lw=1.8, ms=6, label="ROC-AUC（纹丝不动）")
    ax.semilogx(prev, [r["ap"] for r in imb["rows"]], "s-", color=plot.ACCENT2,
                lw=1.8, ms=6, label="PRC 的 AP（一路下滑）")
    ax.semilogx(prev, [r["brier"] for r in imb["rows"]], "^-", color=plot.MUTED,
                lw=1.4, ms=5, label="Brier（概率失准，越小越好）")
    ax.set_xlabel("正例占比 π'", y=1.0)
    ax.set_ylabel("指标值")
    ax.set_title("把负样本复制 100 倍：排序能力没变，有用性没了", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("imbalance-robustness", fig,
               caption="π' 从 0.38 一直压到 0.006：AUC 全程一个值，AP 与 Brier 明显崩坏")


def run(sec: Section) -> None:
    _te, y_te, p_te, coef = fit_and_score()
    base = experiment_auc_two_definitions(sec, y_te, p_te)
    nul = experiment_null_distribution(sec, y_te, p_te, base["auc"])
    knob = experiment_threshold_knob(sec, y_te, p_te)
    imb = experiment_imbalance(sec, y_te, p_te, base)
    prcbase = experiment_prc_baseline(sec, y_te, p_te)
    draw(sec, base, knob, imb, prcbase, nul, y_te, p_te)

    hours = (np.log(knob["thr_star"] / (1 - knob["thr_star"])) - coef[0]) / coef[1]
    sec.observe(
        f"把 Youden 最优点换算回业务语言是 {hours:.2f} 小时 —— "
        f"比书上默认的 p = 0.5 对应的 3.20 小时早了将近 20 分钟。"
        f"这个差别的实际含义是：如果『错过崩溃』比『白跑一趟』更贵，"
        f"就应该按 2.9 小时决策。注意 **最优阈值从来不是模型的属性，"
        f"而是损失函数的属性** —— 书里没给复合损失函数，"
        f"Youden J 只是偷偷替你假设了 FP 和 FN 一样贵。"
    )
    sec.observe(
        "量化上的对应关系：AUC ≈ 因子的**排序能力**（IC 的兄弟），AP ≈ "
        "在这个触发频率下**实际能赚到的钱**。一个日频信号如果正样本（真趋势）只占 1%，"
        "它的 AUC 可能稳稳停在 0.93 让人很兴奋，但 AP 会告诉你 "
        "90% 的上车信号都是假突破。CTA 里最贵的错误恰恰是第二种。"
    )
    sec.pitfall(
        f"别把 AUC 当免死金牌。这次实验里负样本翻了 100 倍，AUC 小数点后六位都没动，"
        f"但 Brier 涨了 {(imb['rows'][-1]['brier'] / imb['rows'][0]['brier'] - 1):.0%} —— "
        f"概率已经完全不校准。用 AUC 选信号排序可以，"
        f"拿它输出的概率去算 Kelly 仓位会直接爆。"
    )
    sec.pitfall(
        f"小测试集上 AUC 本身也是噪声：这次 {len(y_te)} 个样本，"
        f"随机标签的 95% 区间是 [{nul['lo']:.3f}, {nul['hi']:.3f}]。"
        f"看到论文里报『测试集 AUC = 0.68，n = 30』时，"
        f"先问问它的零分布有多宽再说。样本量决定尺子的刻度，这一章从头到尾都在讲这件事。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
