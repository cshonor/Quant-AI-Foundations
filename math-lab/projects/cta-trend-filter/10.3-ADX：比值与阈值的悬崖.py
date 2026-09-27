"""10.3 ADX：比值消灭了量纲，也制造了一道不连续的悬崖

ADX = DX 的 Wilder 平滑，DX = 100·|+DI − −DI| / (+DI + −DI)。
这一节要钉死三件事：
  1. Wilder 递推等价于一个几何加权和（递推不是黑箱，闭式解可以独立验证它）；
  2. 分母会归零：横盘到没有任何方向运动时，+DI = −DI = 0（不是 50/50），DX 直接 NaN；
  3. 计算是良态的（给价格加噪声，ADX 只抖一点点），但"ADX ≥ 20"这个决策是不连续的
     ——良态的指标 + 悬崖式的阈值 = 参数微调就能换一套交易。
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

import cta_data  # noqa: E402
import _indicators as ind  # noqa: E402
from mathviz import Section, num, plot  # noqa: E402

SEC = Section(
    number="10.3",
    chapter="CTA 专题 · 趋势强度指标",
    book="贯穿案例：CTA 趋势策略（BTCUSDT 日线）",
    title="ADX：比值消掉量纲，也留下一道不连续的悬崖",
    claim="ADX 只衡量趋势强弱、不判断方向；+DI/−DI 才区分多空。"
          "ADX < 20 视为震荡，不开新仓。",
    proposition="① Wilder 递推 = 几何加权和的闭式解（两条独立公式必须逐点吻合）；"
                "② 完全没有方向运动时 +DI = −DI = 0 而非 50/50，DX 为 NaN；"
                "③ ADX 用「和」还是「均值」做量纲，结果完全相同（常数因子被比值约掉）；"
                "④ ADX 数值本身对报价噪声稳健，但「ADX ≥ 20」的布尔决策在阈值附近会翻转。",
    source="Wilder 1978 / TA-Lib ta_ADX.c（lookback = 2N−1）",
)

N = 14
ADX_THRESHOLD = 20.0
N_YEARS = 3.0
RNG = np.random.default_rng(31415)


# ------------------------------------------------------------------ ① 递推 = 几何加权和


def wilder_closed(x: np.ndarray, n: int, upto: int) -> np.ndarray:
    """闭式解：v_i = (1−α)^{i−n}·v_n + Σ_{k=n+1..i} (1−α)^{i−k}·x_k（N 项和口径）。

    完全不写递推，只用幂与求和——用来独立验证 `ind.wilder` 的循环实现。
    """
    alpha = 1.0 / n
    out = np.full(len(x), np.nan)
    seed = np.nansum(x[1:n])
    out[n] = seed - seed / n + x[n]
    for i in range(n + 1, min(upto, len(x))):
        k = np.arange(n + 1, i + 1)
        out[i] = (1 - alpha) ** (i - n) * out[n] + np.sum((1 - alpha) ** (i - k) * x[k])
    return out


def experiment_closed_form(sec: Section, high, low, close) -> None:
    tr = ind.true_range(high, low, close)
    upto = N + 160
    rec = ind.wilder(tr, N, scale="sum")
    closed = wilder_closed(tr, N, upto)
    m = np.isfinite(rec[:upto]) & np.isfinite(closed[:upto])
    rel = float(np.max(np.abs(rec[:upto][m] - closed[:upto][m]) / np.abs(closed[:upto][m])))

    sec.check(
        "Wilder 递推与「几何加权和」闭式解逐点吻合",
        rel < 1e-10,
        f"对比 {int(m.sum())} 个点，最大相对差 {rel:.3e}",
        tolerance="相对差 < 1e-10",
    )
    sec.observe(
        "把递推展开成 v_i = (1−α)^{i−n}·v_n + Σ(1−α)^{i−k}·x_k，"
        "Wilder 平滑就是「越老的数据权重越低」的加权和，权重几何衰减、衰减率只由 N 决定。"
        "这点一旦看清，后面你自己写指标就知道该验什么：不是验「有没有值」，是验权重序列。"
    )


# ------------------------------------------------------------------ ② 分母归零：横盘不是 50/50


def experiment_flat_market(sec: Section) -> None:
    m = 120
    centre = 50_000.0
    # 区间逐根收缩：high 递减、low 递增 ⇒ up<0、dn<0 ⇒ +DM = −DM = 0，但 TR > 0
    spread = 400.0 * np.exp(-np.arange(m) / 40.0)
    high = centre + spread
    low = centre - spread
    close = np.full(m, centre)

    adx, pdi, mdi = ind.wilder_adx(high, low, close, N)
    tail = slice(N * 2, None)
    finite_adx = np.isfinite(adx[tail]).sum()
    sec.check(
        "完全没有方向运动时 ADX 为 NaN（不是 0，也不是某个默认值）",
        finite_adx == 0,
        f"第 {2*N} 根之后的 {len(adx[tail])} 根里，有限值 {finite_adx} 个",
        tolerance="有限值个数 = 0",
    )
    sec.check(
        "+DI 与 −DI 同时为 0 —— 横盘不是「多空各 50」，而是「两边都没有」",
        bool(np.all(pdi[tail] == 0.0)) and bool(np.all(mdi[tail] == 0.0)),
        f"+DI 尾部取值集合 {sorted(set(np.round(pdi[tail], 12)))}，"
        f"−DI 尾部取值集合 {sorted(set(np.round(mdi[tail], 12)))}",
        tolerance="恒为 0",
    )
    sec.observe(
        "直觉会说「没有趋势时多空力量均衡，±DI 应该各 50」。实测是 0 和 0："
        "DX 的分子分母同时归零，结果是 NaN 而不是 0。"
        "这条的工程含义很硬：**每个用 ADX 做过滤的策略都必须显式处理 NaN**，"
        "否则 nan ≥ 20 会被判成 False（安静地不开仓），也可能在某些实现里判成 True。"
    )
    sec.pitfall(
        "np.nan >= 20 → False，不报错、不警告。"
        "这正是「过滤器悄悄吞掉信号」的一类来源：不是逻辑写错，是 NaN 的传播。"
    )

    fig, axes = plot.newfig(1, 2, figsize=(7.4, 2.9))
    axes[0].plot(high, lw=1.0, color=plot.ACCENT, label="High")
    axes[0].plot(low, lw=1.0, color=plot.ACCENT2, label="Low")
    axes[0].set_title("区间逐根收缩：没有任何方向运动", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(pdi, lw=1.2, color=plot.ACCENT, label="+DI")
    axes[1].plot(mdi, lw=1.2, color=plot.ACCENT2, label="−DI")
    axes[1].plot(adx, lw=1.2, color="#639922", label="ADX（全 NaN）")
    axes[1].set_title("±DI 恒为 0 → DX = 0/0 → ADX = NaN", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=9)
    sec.figure("flat-market", fig, caption="横盘时 ADX 不是 0 而是 NaN——过滤器必须显式处理")


# ------------------------------------------------------------------ ③ 量纲被比值约掉


def experiment_scale_invariance(sec: Section, high, low, close) -> None:
    """「和口径」与「均值口径」的差别，本质上只是初值不同——这一点能被测出来。"""
    a_sum, _, _ = ind.wilder_adx(high, low, close, N, scale="sum")
    a_mean, _, _ = ind.wilder_adx(high, low, close, N, scale="mean")
    m = np.isfinite(a_sum) & np.isfinite(a_mean)
    rel = np.abs(a_sum[m] - a_mean[m]) / np.abs(a_mean[m])

    head = float(np.mean(rel[:100]))          # 刚出值的 100 根：还在消化种子
    tail = float(np.mean(rel[-200:]))         # 最后 200 根：早就忘掉种子了
    sec.check(
        "两种口径的差别只是「初值不同」，会随时间衰减到可忽略",
        tail < 1e-4 and head > 10 * max(tail, 1e-12),
        f"有效点 {int(m.sum())} 个：前 100 根平均相对差 {head:.2e}，"
        f"最后 200 根平均相对差 {tail:.2e}（衰减了 {head/max(tail,1e-15):.0f} 倍）",
        tolerance="尾部 < 1e-4 且头部 > 尾部 10 倍",
    )
    sec.observe(
        "我原本以为两种口径的 ADX 会「分毫不差」——因为 DX 是比值，常数因子 N 应该被约掉。"
        "实测前 100 根差 {:.2e}、尾部才收敛到 {:.2e}：原因是两种口径差的**不只是**常数因子，"
        "种子的取样窗口也不同（TA-Lib 累加前 n−1 项，教科书取前 n 项均值）。"
        "也就是说这个分歧是**初值差异**，会按 10.2 的 (1−1/N)^k 衰减，而不是恒等。"
        "⇒ 结论仍成立（用哪个口径都行），但理由要改：不是「精确相等」，是「预热期后相等」。".format(head, tail)
    )
    sec.observe(
        "这条在 ATR 上不成立：ATR 是绝对量纲，两种口径差整整 N 倍，"
        "且不会被任何比值约掉。把 ATR 当止损距离用时，口径搞错 = 止损放大 14 倍。"
        "**比值能消掉常数因子，消不掉量纲。**"
    )


# ------------------------------------------------------------------ ④ 良态的指标 + 不连续的决策


def experiment_threshold(sec: Section, high, low, close) -> None:
    fast, slow = ind.sma(close, 20), ind.sma(close, 60)
    raw_long = ind.crossed_above(fast, slow)
    raw_short = ind.crossed_below(fast, slow)
    adx, _, _ = ind.wilder_adx(high, low, close, N)
    sig = (raw_long | raw_short) & np.isfinite(adx)
    adx_at_sig = adx[sig]
    base = adx >= ADX_THRESHOLD

    # (a) 数值稳健性：给价格加 0.05% 噪声，ADX 抖多少
    eps = 5e-4
    h2 = high * (1 + eps * RNG.standard_normal(len(high)))
    l2 = low * (1 - eps * RNG.standard_normal(len(low)))
    adx2, _, _ = ind.wilder_adx(h2, l2, close, N)
    m = np.isfinite(adx) & np.isfinite(adx2)
    drift = float(np.mean(np.abs(adx[m] - adx2[m])))
    worst = float(np.max(np.abs(adx[m] - adx2[m])))

    sec.check(
        "ADX 数值本身是良态的：报价抖动 0.05%，ADX 平均只变不到 0.5",
        drift < 0.5,
        f"{int(m.sum())} 个有效点，平均 |ΔADX| = {drift:.4f}，最大 {worst:.3f}（ADX 的量纲是 0–100）",
        tolerance="平均漂移 < 0.5",
    )

    # (b) 决策不连续：同样这点噪声，布尔信号翻转几根
    base2 = adx2 >= ADX_THRESHOLD
    m2 = np.isfinite(adx) & np.isfinite(adx2)
    flips = int(np.sum(base[m2] != base2[m2]))
    near = np.abs(adx_at_sig - ADX_THRESHOLD) / ADX_THRESHOLD
    wobble = int(np.sum(near < 0.05))          # 落在阈值 ±5% 带内的交叉信号
    sec.check(
        "但布尔决策是不连续的：阈值带内存在「摇摆信号」，微调就会换一套交易",
        flips > 0 or wobble > 0,
        f"0.05% 报价噪声 → {flips} 根 bar 的「ADX≥20」判定翻转；"
        f"全部 {len(adx_at_sig)} 次均线交叉里，有 {wobble} 次的 ADX 落在 20±5% 的摇摆带内，"
        f"最接近阈值的一次 ADX = {adx_at_sig[int(np.argmin(near))]:.2f}",
        tolerance="翻转数 > 0 或摇摆信号 > 0",
    )
    sec.observe(
        "这是本节最该记住的反差：ADX 作为数值是良态的（噪声 0.05% → ADX 几乎不动），"
        "但「ADX ≥ 20」作为决策是不连续的（阶跃函数）。"
        "于是参数扫描的收益曲面一定是锯齿状：阈值从 19.9 挪到 20.1，可能就多了/少了一笔交易。"
        "样本内把阈值调得「刚好最好」，调的其实是噪声。"
    )
    sec.pitfall(
        "做参数扫描时，把「交易数随阈值的阶梯」和「收益随阈值的锯齿」一起画出来。"
        "如果收益的峰只对应 1–2 笔交易的增减，那这个最优阈值没有任何样本外意义。"
    )

    # 图：阈值扫描的阶梯 + 价格/ADX 上的摇摆信号
    thresholds = np.arange(10, 40.1, 0.5)
    counts = [int(np.sum(adx_at_sig >= t)) for t in thresholds]
    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.0))
    axes[0].plot(thresholds, counts, drawstyle="steps-post", lw=1.3, color=plot.ACCENT)
    axes[0].axvline(ADX_THRESHOLD, color=plot.ACCENT2, lw=1.2, ls="--", label="常用阈值 20")
    axes[0].set_xlabel("ADX 阈值")
    axes[0].set_ylabel("通过的交叉信号数")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("信号数是阈值的阶梯函数", fontsize=11)

    axes[1].plot(adx, lw=1.0, color=plot.ACCENT, label="ADX(14)")
    axes[1].axhline(ADX_THRESHOLD, color=plot.ACCENT2, lw=1.2, ls="--", label="阈值 20")
    idx = np.where(sig)[0]
    axes[1].plot(idx, adx[idx], "o", ms=4, color="#639922", label="均线交叉")
    axes[1].set_xlabel("bar")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("绿点落在虚线附近的，就是摇摆信号", fontsize=11)
    for ax in axes:
        ax.tick_params(labelsize=9)
    sec.figure("threshold", fig,
               caption="左：信号数随阈值阶梯式下降；右：阈值线附近的交叉信号是最不稳定的那几笔")


def run(sec: Section) -> None:
    df = cta_data.load(N_YEARS)
    o, high, low, close = cta_data.ohlc(df)
    experiment_closed_form(sec, high, low, close)
    experiment_flat_market(sec)
    experiment_scale_invariance(sec, high, low, close)
    experiment_threshold(sec, high, low, close)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
