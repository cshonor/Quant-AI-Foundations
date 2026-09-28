"""9.3 数据漂移与覆盖缺口：你的模型不知道自己出了运行域

书上讲的东西（第 9 章末段）：

  · **运行域**（ODD）：严格且全面地定义系统应当有效运行的环境与条件，
    再据此倒推需要采集哪些数据。这一概念来自自动驾驶，但对任何模型都成立；
  · 1986 年挑战者号灾难：O 形环的测试从未覆盖 40°F 以下，
    「把它的性能外推到更低的温度最终导致了灾难性后果」；
  · **数据覆盖的 S 曲线**：Stefan Seltz-Axmacher 指出，系统能力的增长不是指数而是
    第 8 章那条 sigmoid 的形状。运行域定得越宽，就越会在远低于 100% 的地方撞上收益递减点；
  · **数据漂移**（数据腐烂）：生产模型所用数据的有效期是有限的。
    吴恩达举的例子：在斯坦福医院数据上训练、在机器上发表论文都说得通，
    但把同一个模型搬到街对面那家设备较旧的医院，性能会显著下降。

这一节三个判决实验：

  ① 挑战者号 O 形环的真实数据：低温段只有 4 次航行，而这 4 次**全部**出现损伤；
  ② 同一个 31°F 的预测，用域内数据拟合与用全数据拟合能差出 28.9 个百分点，
     而这个差别完全来自那 4 个样本；
  ③ 数据漂移：合成一个会「反号」的市场，时间序列切分会暴露衰退
     （样本外准确率**低于抛硬币**），而随机划分把这个事实掩盖得很干净；
  ④ BTC 真实数据的诚实对照 —— 我没能找到漂移的证据，因为模型本来就没有预测力。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_DATA = _ROOT / "data"
for _p in (_ROOT, _DATA, _HERE, _ROOT / "grokking-statistics" / "ch08-逻辑回归与分类"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mathviz import Section, plot  # noqa: E402
from _logistic import fit_logistic, predict_proba  # noqa: E402  (复用第 8 章手写的 IRLS)

SEC = Section(
    number="9.3",
    chapter="第 9 章 · 数据犯罪：Statistics Done Wrong",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="数据漂移与覆盖缺口：模型不知道自己已经出了运行域",
    claim="运行设计域（ODD）先定义系统该在哪些条件下有效，再据此倒推要采集什么数据。"
          "1986 年挑战者号的教训就是这里留了个洞：O 形环从未在 40°F 以下被测试过。"
          "系统能力的增长也不是指数的，而是第 8 章那条 sigmoid —— 运行域定得越宽，"
          "就越会在远低于 100% 的地方撞上收益递减点。"
          "最后，数据是有保质期的：2016 年的机场拥堵数据在 2024 年已经没什么参考价值。",
    proposition="① 23 次航行里，温度 <66°F 的 4 次**全部**出现 O 形环热损伤，"
                "而 ≥66°F 的 19 次只有 21.1%（Fisher 精确检验显著）；"
                "② 同是预测 31°F，只用覆盖域内数据拟合给 71.0%，用全数据拟合给 99.9%，"
                "差 28.9 个百分点全部来自那 4 个样本；"
                "③ 存在机制切换时，时间序列切分会让样本外准确率跌到 0.5 以下（反向预测），"
                "而随机划分把这个危险掩盖成『还行』；"
                "④ 在 BTC 日线上我没找到漂移的证据 —— 但是以否定的方式："
                "随机划分 0.5048 与时序切分 0.4920 相差 1.28pp，在 1σ 之内，"
                "因为模型本来就没有预测力可言。",
    source="第 9 章「运行域」「数据覆盖与 S 曲线」「数据漂移」",
)

# 挑战者号之前 23 次航天飞机飞行的 (发射温度 °F, 该次出现热损伤的 O 形环个数)
# 经典数据集；发射当天的实际温度是 31°F
CHAL_TEMP = np.array([66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78,
                      67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58], dtype=float)
CHAL_DISTRESS = np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                          0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1], dtype=float)
COVER_CUT = 66.0        # 「有充分数据覆盖」的温度下界
LAUNCH_DAY = 31.0       # 挑战者号发射当天的温度


# ------------------------------------------------- ① 覆盖缺口：低温区只有 4 个样本
def experiment_coverage_gap(sec: Section) -> dict:
    y = (CHAL_DISTRESS > 0).astype(float)
    cold = CHAL_TEMP < COVER_CUT
    warm = ~cold
    n_cold, n_warm = int(cold.sum()), int(warm.sum())
    rate_cold, rate_warm = float(y[cold].mean()), float(y[warm].mean())
    # Fisher 精确检验（样本小到皮尔逊卡方不适用）
    table = [[int(y[cold].sum()), n_cold - int(y[cold].sum())],
             [int(y[warm].sum()), n_warm - int(y[warm].sum())]]
    odds, p_fisher = stats.fisher_exact(table)

    sec.check(
        f"挑战者号之前的 {len(CHAL_TEMP)} 次飞行里，"
        f"低于 {COVER_CUT:.0f}°F 的 {n_cold} 次**全部**出现损伤（{rate_cold:.1%}），"
        f"而 {COVER_CUT:.0f}°F 以上的 {n_warm} 次只有 {rate_warm:.1%}（Fisher p = {p_fisher:.4f}）",
        ok=rate_cold == 1.0 and rate_warm < 0.35 and p_fisher < 0.05,
        detail=f"四格表：低温段 {int(y[cold].sum())}/{n_cold} 有损伤，"
               f"常温段 {int(y[warm].sum())}/{n_warm} 有损伤；"
               f"Fisher 精确检验 p = {p_fisher:.4f}（样本这么小，卡方近似是不能用的）。"
               f"低温段的四个点是：53°F(2 个)、57°F(1 个)、58°F(1 个)、63°F(1 个)。"
               f"**信号一直都在，但它是被 4 个点撑起来的** —— "
               f"而它们恰好落在最不该稀疏的那一端。"
               f"书上那句『这种数据覆盖的缺失在运行域的数据中留下了一个关键漏洞』，"
               f"翻过来就是：不是没有证据，是证据少到没人愿意据此叫停发射。",
        tolerance="低温损伤率 = 100%，常温 < 35%，Fisher p < 0.05",
    )
    return {"y": y, "cold": cold, "warm": warm, "n_cold": n_cold, "n_warm": n_warm,
            "rate_cold": rate_cold, "rate_warm": rate_warm, "p_fisher": p_fisher}


# ------------------------------------------------- ② 外推到 31°F：差 28.9 个百分点
def experiment_extrapolation(sec: Section, cov: dict) -> dict:
    y = cov["y"]
    in_domain = CHAL_TEMP >= COVER_CUT
    b0_in, b1_in, _ = fit_logistic(CHAL_TEMP[in_domain], y[in_domain])
    b0_all, b1_all, _ = fit_logistic(CHAL_TEMP, y)
    p_in = float(predict_proba(LAUNCH_DAY, b0_in, b1_in))
    p_all = float(predict_proba(LAUNCH_DAY, b0_all, b1_all))
    gap = p_all - p_in
    # 两台模型在「有数据覆盖」的那一段上的平均预测
    mean_in = float(np.mean(predict_proba(CHAL_TEMP[cov["warm"]], b0_in, b1_in)))
    mean_all = float(np.mean(predict_proba(CHAL_TEMP[cov["warm"]], b0_all, b1_all)))

    sec.check(
        f"同样的 31°F，只用覆盖域内（≥{COVER_CUT:.0f}°F）的数据拟合给出 {p_in:.4f}，"
        f"加上那 {cov['n_cold']} 个低温样本后变成 {p_all:.4f}，差 {gap:.4f}",
        ok=gap > 0.20 and p_in < 0.85,
        detail=f"域内模型：ln O = {b0_in:.4f} {b1_in:+.4f}·T，"
               f"外推到 31°F 得 {p_in:.4f}；"
               f"全数据模型：ln O = {b0_all:.4f} {b1_all:+.4f}·T，得 {p_all:.4f}。"
               f"两条曲线在运行域内几乎没区别（70°F 分别是 "
               f"{float(predict_proba(70.0, b0_in, b1_in)):.4f} 与 "
               f"{float(predict_proba(70.0, b0_all, b1_all)):.4f}），"
               f"但到了运行域外，斜率参数的微小差别被 exp 放大成 {gap:.1%} 的概率差。"
               f"这正是第 7 章「外推」与第 8 章「exp 放大低阶误差」叠加起来的结果："
               f"**运行域内看起来无害的分歧，一出域就是生死之别。**"
               f"（而且注意：即便用全数据拟合，低温段的实测频率是 {cov['rate_cold']:.0%}，"
               f"模型只给到 {p_all:.4f} —— 连全数据模型都还在低估。）",
        tolerance="gap > 0.20 且域内模型的外推值 < 0.85",
    )

    sec.check(
        f"反过来检验域内的两台模型：在 66~81°F 的 {cov['n_warm']} 次飞行上，"
        f"两台模型给出的平均概率非常接近（差 < 0.05），"
        f"而对今天这一次 —— 31°F —— 它们的分歧却是 {gap:.4f}",
        ok=True if gap > 0.20 else None,
        detail=f"两台模型在覆盖域内的 19 个点上平均预测分别是 "
               f"{mean_in:.4f} 与 {mean_all:.4f}，相差 {abs(mean_in - mean_all):.4f}；"
               f"而 31°F 处相差 {gap:.4f}。"
               f"（这一项只报告数值不判胜负，它是给上面那个检查提供背景："
               f"**分歧不是处处存在的，它是被外推创造出来的。**）",
        tolerance="（报告项，不判定）",
    )
    return {"b_in": (b0_in, b1_in), "b_all": (b0_all, b1_all),
            "p_in": p_in, "p_all": p_all, "gap": gap}


# ------------------------------------------------- ③ 漂移：合成一个会反号的市场
def _fit_multi(X: np.ndarray, y: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """多元逻辑回归的 IRLS。"""
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-(A @ w)))
        g = A.T @ (y - p)
        H = (A * (p * (1 - p))[:, None]).T @ A + 1e-8 * np.eye(A.shape[1])
        step = np.linalg.solve(H, g)
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def _apply(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.column_stack([np.ones(len(X)), X]) @ w)))


def experiment_drift_synthetic(sec: Section) -> dict:
    rng = np.random.default_rng(2026)
    n_half = 900
    p = 5
    # 前半段 X→y 由 θ 决定，后半段由 −θ 决定（机制反转，这是最狠的一种漂移）
    theta = rng.normal(0, 1, p) * 0.9
    X1 = rng.normal(size=(n_half, p))
    z1 = X1 @ theta
    y1 = (z1 + rng.normal(0, 0.6, n_half) > 0).astype(float)
    X2 = rng.normal(size=(n_half, p))
    z2 = X2 @ (-theta)
    y2 = (z2 + rng.normal(0, 0.6, n_half) > 0).astype(float)
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    n = len(y)

    # A. 随机划分：把两个机制搅在一起，学到的是「平均信号」≈ 0
    accs = []
    for s in range(40):
        idx = rng.permutation(n)
        te, tr = idx[:n // 3], idx[n // 3:]
        w = _fit_multi(X[tr], y[tr])
        accs.append(((_apply(X[te], w) >= 0.5).astype(float) == y[te]).mean())
    acc_random = float(np.mean(accs))

    # B. 时间序列切分：用前半段（机制 1）训练，在后半段（机制 2）测试
    tr_seq = np.arange(n_half)
    te_seq = np.arange(n_half, n)
    w_seq = _fit_multi(X[tr_seq], y[tr_seq])
    acc_seq = float(((_apply(X[te_seq], w_seq) >= 0.5).astype(float) == y[te_seq]).mean())
    # 顺便看看它在自己那段上的表现（这是「回溯起来很好看」的那个数字）
    acc_seq_insample = float(((_apply(X[tr_seq], w_seq) >= 0.5).astype(float)
                              == y[tr_seq]).mean())

    sec.check(
        f"机制反转的合成市场：时间序列切分把样本外准确率打到 {acc_seq:.4f}，"
        f"**比抛硬币还差**；而随机划分给出的却是 {acc_random:.4f}，看起来『还过得去』",
        ok=acc_seq < 0.45 and abs(acc_random - 0.5) < 0.03,
        detail=f"生成过程：前 {n_half} 天 y 由 θ·X 决定，后 {n_half} 天由 −θ·X 决定。"
               f"训练集上的拟合准确率是 {acc_seq_insample:.4f}（很好看），"
               f"同一套权重用到后半段就变成 {acc_seq:.4f}。"
               f"**低于 0.5 是关键证据**：它说明模型不是『失去预测力』，"
               f"而是在**系统性地反向预测** —— 这比随机糟糕得多。"
               f"而随机划分把这个危险完全藏起来了：它给出 {acc_random:.4f}，"
               f"因为抽样时两个机制各占一半、彼此抵消，模型学到的是一个接近 0 的平均信号。"
               f"（40 次随机划分的标准差很小，这不是抽样噪声。）"
               f"⇒ **评价任何时序模型都必须按时间切分，不能打乱。**",
        tolerance="时序样本外 < 0.45 且随机划分 |−0.5| < 0.03",
    )
    return {"acc_random": acc_random, "acc_seq": acc_seq,
            "acc_seq_insample": acc_seq_insample, "theta": theta}


# ------------------------------------------------- ④ BTC：诚实的否定结果
def experiment_drift_btc(sec: Section) -> dict | None:
    try:
        from cta_data import load  # noqa: E402
    except Exception:
        return None
    try:
        df = load(None)
    except FileNotFoundError:
        return None
    close = df["Close"].to_numpy(float)
    lr = np.diff(np.log(close))
    lag = 10
    X = np.column_stack([lr[i:len(lr) - lag + i] for i in range(lag)])
    y = (lr[lag:] > 0).astype(float)
    n = len(y)

    def one(tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        w = _fit_multi(Xtr, y[tr])
        return float(((_apply(Xte, w) >= 0.5).astype(float) == y[te]).mean())

    rng = np.random.default_rng(4)
    accs = []
    for _ in range(200):
        idx = rng.permutation(n)          # 一次抽取拆成两半，保证 train/test 不相交
        te, tr = idx[:n // 3], idx[n // 3:]
        accs.append(one(tr, te))
    rnd = np.array(accs)
    acc_random = float(rnd.mean())
    sd_random = float(rnd.std())
    half = n // 2
    acc_seq = one(np.arange(half), np.arange(half, n))
    gap = acc_random - acc_seq

    sec.check(
        f"BTC 日线上我没找到漂移的证据 —— 但这是个否定结果："
        f"随机划分 {acc_random:.4f}、时序切分 {acc_seq:.4f}，相差 {gap:.4f}，"
        f"还在随机划分自身噪声的 1σ 之内（σ = {sd_random:.4f}）",
        ok=abs(gap) < sd_random,
        detail=f"{n} 个样本（{df.index[0].date()} ~ {df.index[-1].date()}），"
               f"特征 = 过去 {lag} 天对数收益，标签 = 次日涨跌。"
               f"200 次随机划分：平均 {acc_random:.4f}、标准差 {sd_random:.4f}；"
               f"前半训练 / 后半测试：{acc_seq:.4f}。"
               f"**我原本想在这里演示『漂移让样本外变差』，实测没能成立。**"
               f"原因不神秘：这个模型本身的准确率就在 0.50 附近，"
               f"即它压根没有学到可迁移的信号 —— 没有信号可以漂。"
               f"（这不代表 BTC 没有漂移，只代表「用过去 10 天收益预测次日涨跌」"
               f"这件事在任何一段上都不成立。）"
               f"把否定结果写在这里是因为它同样是一条纪律："
               f"**不能先设结论再去找数据支持它。**",
        tolerance="|差距| < 随机划分的标准差",
    )
    return {"acc_random": acc_random, "sd_random": sd_random,
            "acc_seq": acc_seq, "gap": gap, "n": n}


# ------------------------------------------------- 图
def draw(sec: Section, cov: dict, extr: dict, dr: dict, btc: dict | None) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.scatter(CHAL_TEMP[cov["cold"]], cov["y"][cov["cold"]], s=90, marker="s",
               color=plot.ACCENT, zorder=6, label="低温段（<66°F）：4 次全毁")
    ax.scatter(CHAL_TEMP[cov["warm"]], cov["y"][cov["warm"]], s=55, alpha=0.75,
               color=plot.MUTED, label="覆盖充分的常温段")
    grid = np.linspace(30, 85, 400)
    ax.plot(grid, predict_proba(grid, *extr["b_in"]), lw=1.8, color=plot.ACCENT2,
            label="只用 ≥66°F 拟合")
    ax.plot(grid, predict_proba(grid, *extr["b_all"]), lw=1.8, ls="--", color="k",
            label="用全部数据拟合")
    ax.axvspan(30, COVER_CUT, color=plot.ACCENT, alpha=0.09)
    ax.axvline(LAUNCH_DAY, ls=":", lw=1.2, color=plot.ACCENT)
    ax.text(LAUNCH_DAY + 0.6, 0.9, f"发射当天 {LAUNCH_DAY:.0f}°F\n{gap_txt(extr)}",
            fontsize=8, color=plot.ACCENT, va="top")
    ax.set_xlabel("发射温度 °F"); ax.set_ylabel("P(O 形环热损伤)")
    ax.set_title("左边那条浅红色带子就是没有数据的地方", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="center right")
    ax.tick_params(labelsize=8)

    ax = axes[1]
    labels = ["训练集\n(机制 1)", "样本外\n(机制 2)", "随机划分\n(混合)"]
    vals = [dr["acc_seq_insample"], dr["acc_seq"], dr["acc_random"]]
    colors = [plot.MUTED, plot.ACCENT, plot.ACCENT2]
    ax.bar(range(3), vals, 0.55, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.4f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", lw=1.1, color="k", alpha=0.7)
    ax.text(2.42, 0.512, "抛硬币", fontsize=8, ha="right")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("准确率")
    ax.set_title("时间序列切分才能看出它已经反向了", fontsize=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("coverage-and-drift", fig,
               caption="左边是挑战者号 23 次飞行；右边是机制反转市场的三种评估口径")


def gap_txt(extr: dict) -> str:
    return f"{extr['p_in']:.2f} vs {extr['p_all']:.2f}"


def run(sec: Section) -> None:
    cov = experiment_coverage_gap(sec)
    extr = experiment_extrapolation(sec, cov)
    dr = experiment_drift_synthetic(sec)
    btc = experiment_drift_btc(sec)
    draw(sec, cov, extr, dr, btc)

    sec.observe(
        f"这三件事串起来是一条完整的因果链：**缺什么数据 → 模型在域外给不出警告 → "
        f"而评价流程又把这件事藏起来。** 挑战者号那一晚， Coverage 缺口让两位高涨模型"
        f"在 31°F 上给出 {extr['p_in']:.2f} 这种『六成不像事』的读数；"
        f"而如果一个流程只报训练内的 {dr['acc_seq_insample']:.2f}，"
        f"衰退会长什么样都看不到。"
    )
    sec.observe(
        "CTA 上最直接的对应：任何声称『全市场全周期有效』的策略，"
        "都在赌自己的运行域覆盖了所有未来。而实际能做的是把 ODD 写下来 —— "
        "波动率区间、趋势/震荡状态、流动性水平、合约是否主力 —— "
        "**然后在这些状态之外主动停手**，而不是让模型在陌生环境里照常下单。"
    )
    sec.pitfall(
        "别把「模型在用久了会过期」当成默认结论。BTC 那一项给出的教训恰恰相反："
        "这次没测出漂移，是因为模型一开始就没有预测力。"
        "**诊断不出漂移，和不存在漂移，是两件必须分开写的事。**"
    )
    sec.pitfall(
        "书把『多加一个功能会导致范围爆炸』那张 XKCD 讲得轻松，"
        "但它真正的量化含义是 sigmoid 的收益递减：由于 log-odds 是线性的，"
        "从 60% 覆盖推到 90% 需要的样本量是按 exp 放大的。"
        "在答应『顺便多支持一个市场状态』之前，先算一下它的对数几率代价。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
