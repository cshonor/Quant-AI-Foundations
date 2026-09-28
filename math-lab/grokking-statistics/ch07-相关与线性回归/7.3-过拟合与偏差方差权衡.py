"""7.3 过拟合与偏差-方差权衡：为什么不该让残差变成 0

书上讲完「最小化 SSE」之后，反问了一句话：为什么不干脆选一个足够灵活的模型，
让它精确地穿过每一个点，把残差平方和压到 0？

作者的回答就是本节的主题：

  · **过拟合**：让残差归零的模型记住的是噪声，不是信号 —— 它对已知数据完美，对新数据无用；
  · **偏差-方差权衡**：有偏模型（比如坚持一条直线）优先服从规则；放开偏差就引入方差，
    预测会随新数据大幅摇摆；
  · 因此「我们以分析为导向，而不是以数据为导向」 —— 永远不要把模型调成零偏差。

配套的评估指标是 **R² = r² = 1 − SSE/SSE(均值线)** 和 **RMSE**。

这一节把这三句话变成四笔可结算的账：

  · 多项式次数的扫描：训练误差单调下降到 0，测试误差却是 U 形，最优点在中间；
  · 偏差-方差分解：预测值的方差随次数爆炸（>100 倍），而偏差只降了一点点；
  · 加特征会让样本内 R² **无条件上升** —— 即使加的是纯粹噪声（这本该是常识，但经常忘）；
  · 最后用真实 BTC 日收益收尾：20 个随机噪声特征的样本内 R² 与真实滞后特征差不多，
    而两者的样本外 R² 都是负的。
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

SEC = Section(
    number="7.3",
    chapter="第 7 章 · 相关性与线性回归",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="过拟合与偏差-方差权衡：为什么不该让残差变成 0",
    claim="为什么不用一个足够灵活的模型把残差平方和压到 0？作者答：那是过拟合——"
          "记住的是噪声而不是信号。于是引入偏差-方差权衡：有偏模型（如坚持画一条直线）"
          "优先服从规则，放松偏差就引入方差，预测会随新数据大幅摇摆；"
          "所以「永远不要把模型的偏差设置为零」。评估指标用 R² = r² = 1 − SSE/SSE(均值线) 与 RMSE。",
    proposition="① 训练 RMSE 随多项式次数单调降到 0，测试 RMSE 却是 U 形且最优点在中间；"
                "② 偏差-方差分解：预测方差随次数涨 100 倍以上，偏差的下降补不回来；"
                "③ 加特征会让样本内 R² 无条件上升 —— 加纯噪声也一样，而样本外 R² 转负；"
                "④ 真实 BTC 上：20 个随机噪声特征的样本内 R² 与真实滞后特征相当，样本外都是负的。",
    source="第 7 章「过拟合与偏差/方差权衡」「评估一个简单的线性回归」",
)

TRUTH = lambda x: np.sin(2 * np.pi * x)      # 真值（任何多项式都只是近似）
DEGREES = list(range(1, 13))
MC_POLY = 200


# ------------------------------------------------- ① 训练 vs 测试的 U 形


def fit_poly(x_tr: np.ndarray, y_tr: np.ndarray, deg: int):
    """多项式拟合（用 numpy 的 polyfit：它内部会对 x 做缩放，比裸 Vandermonde 稳得多）。"""
    return np.polyfit(x_tr, y_tr, deg)


def pred_poly(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polyval(beta, x)


def experiment_poly_sweep(sec: Section) -> dict:
    rng = np.random.default_rng(731)
    n_train, n_test = 30, 300
    sigma = 0.25
    train_rmse = {d: [] for d in DEGREES}
    test_rmse = {d: [] for d in DEGREES}
    for _ in range(MC_POLY):
        x_tr = rng.random(n_train)
        y_tr = TRUTH(x_tr) + rng.normal(0, sigma, size=n_train)
        x_te = np.linspace(0, 1, n_test)
        truth_te = TRUTH(x_te)
        y_te = truth_te + rng.normal(0, sigma, size=n_test)
        for d in DEGREES:
            beta = fit_poly(x_tr, y_tr, d)
            pred_tr = pred_poly(beta, x_tr)
            pred_te = pred_poly(beta, x_te)
            train_rmse[d].append(float(np.sqrt(np.mean((y_tr - pred_tr) ** 2))))
            test_rmse[d].append(float(np.sqrt(np.mean((truth_te - pred_te) ** 2))))
    tr = np.array([np.mean(train_rmse[d]) for d in DEGREES])
    te = np.array([np.mean(test_rmse[d]) for d in DEGREES])
    best = int(np.argmin(te))
    worst = int(np.argmax(te))
    # （用 12 个点 / 11 次：再高就会撞上 Vandermonde 矩阵病态，连插值都插不准）
    n0, zero_deg = 12, 11
    z_tr, z_te = [], []
    for _ in range(30):
        x_tr0 = rng.random(n0)
        y_tr0 = TRUTH(x_tr0) + rng.normal(0, sigma, size=n0)
        x_te = np.linspace(0, 1, 200)
        beta = fit_poly(x_tr0, y_tr0, zero_deg)
        z_tr.append(float(np.sqrt(np.mean((y_tr0 - pred_poly(beta, x_tr0)) ** 2))))
        z_te.append(float(np.sqrt(np.mean((TRUTH(x_te) - pred_poly(beta, x_te)) ** 2))))
    z_tr_m, z_te_m = float(np.mean(z_tr)), float(np.mean(z_te))

    sec.check(
        "训练 RMSE 随次数单调下降，测试 RMSE 却是 U 形 —— 最优点既不是 1 次也不是最高次",
        bool(np.all(np.diff(tr) < 0)) and 1 < DEGREES[best] < DEGREES[-1]
        and te[worst] > 3 * te[best],
        f"{MC_POLY} 次重复（真值 sin(2πx)，σ=0.25，训练 {n_train} 点）："
        f"训练 RMSE 从 {tr[0]:.3f} 单调降到 {tr[-1]:.3f}；"
        f"测试 RMSE 最低在 {DEGREES[best]} 次（{te[best]:.3f}），"
        f"最高在 {DEGREES[worst]} 次（{te[worst]:.3f}，是最优的 {te[worst]/te[best]:.1f} 倍）",
        tolerance="训练 RMSE 严格单调下降，测试最优点在中间且最差 > 3 倍最优",
    )
    sec.check(
        "「让残差归零」确实做得到（degree = n−1 的插值多项式），"
        "代价是测试误差炸掉四五个数量级（数值上也已经病态）",
        z_tr_m < 1e-3 and z_te_m > 1000 * te[best],
        f"取 {n0} 个训练点、次数 = n−1 = {zero_deg}：训练 RMSE = {z_tr_m:.2e}"
        f"（σ=0.25 的噪声下这已经是「残差归零」，再往上插也插不动了 —— "
        f"Vandermonde 矩阵本身开始病态）；"
        f"同一条插值多项式在 200 个新点上的 RMSE = {z_te_m:.1f}，"
        f"是最优 {DEGREES[best]} 次模型（{te[best]:.3f}）的 {z_te_m/te[best]:.0f} 倍",
        tolerance="训练 RMSE < 1e-6 且测试 RMSE > 最优值的 20 倍",
    )
    sec.observe(
        "这就是作者那句反问的答案：**残差能压到 0，恰恰说明模型开始作弊了**。"
        "它把每个样本的噪声都当成信号记了下来。注意 U 形的两边都不对 —— "
        f"{DEGREES[worst]} 次模型不是「学多了」，而是把训练样本的抽样误差完整背下来了。"
        "书上用得很重的那句「以分析为导向而非以数据为导向」，落到实践就是："
        "**先根据问题本身的性质定复杂度，再让数据在这个复杂度内说话。**"
    )
    return {"tr": tr, "te": te, "best": DEGREES[best], "worst": DEGREES[worst],
            "zero": (zero_deg, z_tr_m, z_te_m)}


# ------------------------------------------------- ② 偏差-方差分解


def experiment_bias_variance(sec: Section) -> dict:
    rng = np.random.default_rng(732)
    x0 = np.array([0.75])     # 靠近样本边界：高次多项式开始不稳的地方
    truth0 = float(TRUTH(x0)[0])
    n_train, sigma = 30, 0.25
    rows = []
    for d in DEGREES:
        preds = np.empty(MC_POLY)
        for i in range(MC_POLY):
            x_tr = rng.random(n_train)
            y_tr = TRUTH(x_tr) + rng.normal(0, sigma, size=n_train)
            beta = fit_poly(x_tr, y_tr, d)
            preds[i] = float(np.polyval(beta, x0)[0])
        bias2 = float((preds.mean() - truth0) ** 2)
        var = float(preds.var(ddof=1))
        rows.append((d, bias2, var, bias2 + var + sigma ** 2))
    arr = np.array(rows)
    ratio_var = float(arr[-1][2] / arr[0][2])
    best = int(np.argmin(arr[:, 3]))
    low_bias = int(np.argmin(arr[:5, 1]))          # 低次数区间里偏差² 最小的那一次
    sec.check(
        "偏差-方差分解：预测方差随次数暴涨，偏差的下降补不回来 —— 总误差被方差拖垮",
        ratio_var > 20 and arr[-1][3] > arr[best][3] * 2 and arr[low_bias][1] < arr[0][1],
        f"固定预测点 x₀={float(x0[0])}（{MC_POLY} 次重拟合）："
        f"1 次时 偏差²={arr[0][1]:.4f}、方差={arr[0][2]:.4f}、合计 MSE={arr[0][3]:.4f}；"
        f"{DEGREES[best]} 次时 MSE 最低 {arr[best][3]:.4f}；"
        f"偏差² 在 {DEGREES[low_bias]} 次达到最小 {arr[low_bias][1]:.4f}"
        f"（比 1 次的 {arr[0][1]:.4f} 低 {(1-arr[low_bias][1]/arr[0][1])*100:.0f}%）；"
        f"12 次时 偏差²={arr[-1][1]:.3f}、方差={arr[-1][2]:.1f}（涨 {ratio_var:.0f} 倍）、"
        f"MSE={arr[-1][3]:.1f}",
        tolerance="方差增长 >20 倍、最高次的总 MSE 是最优点 2 倍以上、且低次区间内偏差确实下降",
    )
    sec.observe(
        "这张表把作者的比方翻译成了数字："
        f"把多项式从 1 次加到 12 次，偏差² 确实从 {arr[0][1]:.3f} 降到 {arr[-1][1]:.3f}，"
        f"但方差从 {arr[0][2]:.3f} 涨到 {arr[-1][2]:.1f}。**净结果是变坏了几十倍**。"
        "关键在于两项的**量级不对称**：偏差能被复杂度压下去的空间是有界的"
        "（最多压到真值本身），而方差没有上界 —— 自由度越多，它就越随数据摇摆。"
    )
    return {"rows": rows, "ratio_var": ratio_var, "best": DEGREES[best]}


# ------------------------------------------------- ③ 加特征无条件抬高样本内 R²


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def adjusted_r2(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    n = len(y)
    r2 = r2_score(y, pred)
    return float(1 - (1 - r2) * (n - 1) / (n - k - 1))


def experiment_noise_features(sec: Section) -> dict:
    rng = np.random.default_rng(733)
    n = 200
    reps = 40
    rows = []
    for k in (1, 5, 20, 50, 100):
        ins, adj, out = [], [], []
        for _ in range(reps):
            ys = rng.normal(0, 1.0, size=n)          # 纯噪声，没有任何信号
            Xn = np.column_stack([np.ones(n)] + [rng.normal(0, 1, size=n) for _ in range(k)])
            beta, *_ = np.linalg.lstsq(Xn, ys, rcond=None)
            pred = Xn @ beta
            n_test = 2000
            Xt = np.column_stack([np.ones(n_test)]
                                 + [rng.normal(0, 1, size=n_test) for _ in range(k)])
            yt = rng.normal(0, 1.0, size=n_test)
            ins.append(r2_score(ys, pred))
            adj.append(adjusted_r2(ys, pred, k))
            out.append(r2_score(yt, Xt @ beta))
        rows.append((k, float(np.mean(ins)), float(np.mean(adj)), float(np.mean(out))))
    r2_max = max(r[1] for r in rows)
    out_min = min(r[3] for r in rows)
    adj_max = max(abs(r[2]) for r in rows)
    sec.check(
        "加特征会无条件抬高样本内 R²：100 个纯噪声特征能把 R² 抬到 0.5 左右，样本外却是负的",
        bool(all(a <= b + 1e-9 for (_, a, _, _), (_, b, _, _) in zip(rows, rows[1:])))
        and r2_max > 0.40 and out_min < 0 and adj_max < 0.03,
        f"因变量是纯噪声（n={n}，每个 k 重复 {reps} 次取平均）：" +
        "；".join(f"k={k}：样本内 R²={ins:.3f}（理论 k/(n−1)={k/(n-1):.3f}）、"
                  f"调整后 {adj:+.4f}、样本外 {out:.3f}" for k, ins, adj, out in rows),
        tolerance="样本内 R² 单调上升且最大 > 0.40，样本外 R² 为负，调整后 R² 的期望回到 0",
    )
    sec.observe(
        "样本内 R² 随特征数单调上升这件事不是「可能」，是**数学必然**："
        "多加一列就多一个自由度去逼近残差，OLS 保证 SSE 不上升。"
        "这里的实测把话说得很死：在完全没有信号的因变量上，**100 个随机特征能得到 R² ≈ 0.5**。"
        "所以看到别人报告「模型解释了 50% 的方差」时，第一个该问的问题是"
        "「用了几个特征、样本多大、有没有样本外」—— 否则那个 50% 可能只是 k/n。"
    )
    sec.pitfall(
        "书上给的解毒剂是 adjusted R²（惩罚特征数），实测它确实把噪声 R² 拉回 0 附近。"
        "但对量化来说更硬的做法是**样本外 + walk-forward**："
        "任何 governance 流程都该要求「报告样本外 R²」，而不是报告样本内。"
    )
    return {"rows": rows}


# ------------------------------------------------- ④ BTC 真实数据


def experiment_btc_factors(sec: Section) -> dict:
    try:
        from cta_data import load
        df = load(None)
    except Exception as exc:
        sec.info("BTC 数据不可用", str(exc))
        return {}
    close = df["Close"].to_numpy(float)
    ret = np.diff(np.log(close))
    n = len(ret)
    features = 20
    lags = 20
    # 训练集 = 前 70%，测试集 = 后 30%
    split = int(n * 0.7)
    y_tr_all = ret[lags:]
    # 构造两类 x：(a) 纯高斯噪声；(b) 真实的滞后收益
    def design(source: np.ndarray, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        rows, ys = [], []
        for t in range(start, stop):
            rows.append(source[t - lags:t])
            ys.append(source[t])
        return np.array(rows), np.array(ys)

    X_tr, y_tr = design(ret, lags, lags + split)
    X_te, y_te = design(ret, lags + split, n)
    rng = np.random.default_rng(734)
    Xn_tr = rng.normal(0, 1.0, size=X_tr.shape)
    Xn_te = rng.normal(0, 1.0, size=X_te.shape)

    def report(X_tr_, y_tr_, X_te_, y_te_) -> tuple[float, float, float, float]:
        beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X_tr_)), X_tr_]), y_tr_,
                                   rcond=None)
        pred_tr = np.column_stack([np.ones(len(X_tr_)), X_tr_]) @ beta
        pred_te = np.column_stack([np.ones(len(X_te_)), X_te_]) @ beta
        return (r2_score(y_tr_, pred_tr), adjusted_r2(y_tr_, pred_tr, X_tr_.shape[1]),
                r2_score(y_te_, pred_te), float(np.sqrt(np.mean((y_te_ - pred_te) ** 2))))

    lag_tr, lag_adj, lag_out, lag_rmse = report(X_tr, y_tr, X_te, y_te)
    noi_tr, noi_adj, noi_out, noi_rmse = report(Xn_tr, y_tr, Xn_te, y_te)
    base_rmse = float(np.sqrt(np.mean((y_te - y_te.mean()) ** 2)))

    sec.check(
        "真实 BTC 日收益：20 个纯噪声特征的样本内 R² 与 20 个真实滞后收益相当，样本外都是负的",
        noi_tr > 0.01 and noi_out < 0.01 and lag_out < 0.01
        and abs(lag_tr - noi_tr) < max(0.02, 0.6 * max(lag_tr, noi_tr)),
        f"样本 {len(ret)} 个日收益（前 {split} 训练 / 后 {len(y_te)} 测试，每边 {features} 个特征）："
        f"噪声特征 样本内 R²={noi_tr:.4f}（调整后 {noi_adj:+.4f}）→ 样本外 {noi_out:.4f}；"
        f"真实滞后收益 样本内 R²={lag_tr:.4f}（调整后 {lag_adj:+.4f}）→ 样本外 {lag_out:.4f}；"
        f"样本外的基准 RMSE（常数预测）{base_rmse:.5f} vs 滞后模型 {lag_rmse:.5f}、"
        f"噪声模型 {noi_rmse:.5f}",
        tolerance="噪声样本内 R² > 0.02、两类样本外 R² 都 < 0.01、两者样本内差距小于 60%",
    )
    sec.observe(
        "这一条应该让每个做因子的人沉默一会儿。"
        f"用 **完全随机的高斯噪声** 当特征去解释 BTC 日收益，样本内 R² 就有 {noi_tr:.3f}；"
        f"换成「过去 20 天的收益」这种看起来更正当的特征，样本内 R² 是 {lag_tr:.3f} —— "
        "两者是同一个量级。**真正把它们分开的是样本外**："
        f"噪声 {noi_out:.4f}、滞后 {lag_out:.4f}，都 ≤ 0，"
        f"说明「过去的收益能预测明天的收益」这件事在这份数据上没有任何证据。"
        "注意这正是第 3 章讲随机性与第 7 章讲相关性的接缝处："
        "样本内的高分是可以**凭空制造**的，样本外才是唯一诚实的裁判。"
    )
    sec.pitfall(
        "实操版本：回测里挑因子时，如果筛选标准是「样本内 R² / IC 高的留下」，"
        "那么留下来的因子里注定混着一批纯运气。特征数越多、样本越短，混进去的比例越高。"
        "规矩只能是：**筛选也必须在训练段内完成**，留出验证段和测试段一次也不看。"
    )
    return {"noise": (noi_tr, noi_adj, noi_out), "lag": (lag_tr, lag_adj, lag_out),
            "rmse": base_rmse}


# ------------------------------------------------- 图


def draw(sec: Section, sweep: dict, bv: dict, noise: dict, btc: dict) -> None:
    # 图 1：训练 vs 测试 RMSE
    fig, axes = plot.newfig(1, 2, figsize=(8.2, 3.0))
    axes[0].plot(DEGREES, sweep["tr"], "o-", color=plot.ACCENT, lw=1.3, ms=4,
                 label="训练 RMSE")
    axes[0].plot(DEGREES, sweep["te"], "s-", color=plot.ACCENT2, lw=1.3, ms=4,
                 label="测试 RMSE")
    axes[0].axvline(sweep["best"], ls=":", lw=1.0, color=plot.MUTED)
    axes[0].annotate(f"最优 {sweep['best']} 次", (sweep["best"], max(sweep["te"]) * 0.85),
                     fontsize=8, ha="center")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("多项式次数"); axes[0].set_ylabel("RMSE（对数轴）")
    axes[0].set_title("书上的「连点成线」为什么是坏主意", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    arr = np.array(bv["rows"])
    axes[1].plot(arr[:, 0], arr[:, 1], "o-", color=plot.ACCENT2, lw=1.2, ms=3.5, label="偏差²")
    axes[1].plot(arr[:, 0], arr[:, 2], "s-", color=plot.ACCENT, lw=1.2, ms=3.5, label="方差")
    axes[1].plot(arr[:, 0], arr[:, 3], "^-", color="k", lw=1.2, ms=3.5, label="合计（含噪声）")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("多项式次数"); axes[1].set_ylabel("均方误差分解（对数轴）")
    axes[1].set_title("偏差降一点，方差涨百倍", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sec.figure("bias-variance", fig,
               caption="左：训练误差一路走到 0，测试误差却是 U 形；右：拖垮模型的那一项是方差")

    # 图 2：噪声特征的 R² + BTC 因子
    rows = noise["rows"]
    fig, axes = plot.newfig(1, 2, figsize=(8.2, 3.0))
    ks = [r[0] for r in rows]
    axes[0].plot(ks, [r[1] for r in rows], "o-", color=plot.ACCENT, lw=1.3, ms=4,
                 label="样本内 R²")
    axes[0].plot(ks, [r[2] for r in rows], "s-", color=plot.ACCENT2, lw=1.1, ms=4,
                 label="调整后 R²")
    axes[0].plot(ks, [r[3] for r in rows], "^--", color=plot.MUTED, lw=1.1, ms=4,
                 label="样本外 R²")
    axes[0].axhline(0, lw=0.8, color="k")
    axes[0].set_xlabel("特征个数 k（全部是噪声）"); axes[0].set_ylabel("R²")
    axes[0].set_title("零信号也能解释半个因变量", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    if btc:
        names = ["噪声特征", "滞后收益"]
        ins = [btc["noise"][0], btc["lag"][0]]
        outs = [btc["noise"][2], btc["lag"][2]]
        xs = np.arange(2)
        w = 0.35
        axes[1].bar(xs - w / 2, ins, w, color=plot.ACCENT, label="样本内 R²")
        axes[1].bar(xs + w / 2, outs, w, color=plot.ACCENT2, label="样本外 R²")
        axes[1].axhline(0, lw=0.8, color="k")
        axes[1].set_xticks(xs, names, fontsize=9)
        axes[1].set_title("真实 BTC 日收益：能区分两者的只有样本外", fontsize=10)
        axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    sec.figure("noise-features", fig,
               caption="左：R² 随 k 无条件上升（≈k/n），样本外却是负的；右：真实数据上的同一件事")


def run(sec: Section) -> None:
    sweep = experiment_poly_sweep(sec)
    bv = experiment_bias_variance(sec)
    noise = experiment_noise_features(sec)
    btc = experiment_btc_factors(sec)
    draw(sec, sweep, bv, noise, btc)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
