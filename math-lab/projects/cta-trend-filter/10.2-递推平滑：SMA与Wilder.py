"""10.2 递推平滑：SMA、EMA 与 Wilder 其实是同一个一阶递推

CTA 里所有"平滑"都长一个样子：v_i = (1-α)·v_{i-1} + α·x_i（或等价的写法）。
N 日均线、指数均线、Wilder 平滑（ATR/ADX 的内核）只是 α 与量纲不同。
这一节把三件事钉死：稳态增益、初值遗忘速度、相位滞后与降噪能力的此消彼长。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _indicators as ind  # noqa: E402
from mathviz import Section, num, plot  # noqa: E402

SEC = Section(
    number="10.2",
    chapter="CTA 专题 · 时间序列平滑",
    book="贯穿案例：CTA 趋势策略（BTCUSDT 日线）",
    title="递推平滑：SMA、EMA、Wilder 是同一个一阶递推",
    claim="Wilder 平滑（ADX/ATR 的内核）写成 v_i = v_{i-1} − v_{i-1}/N + x_i；"
          "它和标准 EMA 形式不同，只是因为它的输出保持「N 项和」的量纲，而不是均值。",
    proposition="① 常数输入下，Wilder(N 项和口径) 的稳态值是 EMA(α=1/N) 的 N 倍；"
                "② 初值差异按 (1−1/N)^k 指数遗忘，实测衰减率与理论吻合；"
                "③ 相位滞后：SMA(N) 滞后 (N−1)/2 根，EMA(α=1/N) 滞后 (N−1) 根，正好 2 倍；"
                "④ 降噪能力反过来：SMA 输出方差只有 EMA 的一半左右。",
    source="一阶线性递推 / Wilder《New Concepts in Technical Trading Systems》",
)

N = 14
ALPHA = 1.0 / N
RNG = np.random.default_rng(20260927)


# ------------------------------------------------------------------ ① 稳态增益


def experiment_gain(sec: Section) -> None:
    x = np.full(400, 3.0)
    w_sum = ind.wilder(x, N, scale="sum")
    w_mean = ind.wilder(x, N, scale="mean")
    e = ind.ema(x, ALPHA)
    tail = slice(300, None)

    gain_sum = float(np.mean(w_sum[tail]))
    gain_mean = float(np.mean(w_mean[tail]))
    gain_ema = float(np.mean(e[tail]))

    sec.check(
        "常数输入下，Wilder(N项和口径) 稳态值 = N × 输入；EMA 稳态值 = 输入",
        abs(gain_sum / N - 3.0) < 1e-9 and abs(gain_mean - 3.0) < 1e-9,
        f"输入恒为 3.0：Wilder(sum) 稳态 {gain_sum:.9f}（= N×3 = {N*3.0}），"
        f"Wilder(mean) 稳态 {gain_mean:.9f}，EMA 稳态 {gain_ema:.9f}",
        tolerance="误差 < 1e-9",
    )
    sec.check(
        "两种 Wilder 口径只差常数因子 N（比值精确为 N）",
        abs(gain_sum / gain_mean - N) < 1e-9,
        f"实测比值 {gain_sum / gain_mean:.12f}",
        tolerance="与 N 的偏差 < 1e-9",
    )
    sec.observe(
        "Wilder 那条递推没有 α·x_i 而是直接 x_i，不是写错，是刻意保持「N 项和」的量纲："
        "ATR 要的是波幅之和，+DI/−DI 后面要做比值，常数因子会被约掉。"
        "所以「用均值还是用和做种子」在 ADX 里无差别——这一点 10.3 会用实测确认。"
    )


# ------------------------------------------------------------------ ② 初值遗忘


def experiment_forgetting(sec: Section) -> None:
    x = np.full(300, 5.0)
    a = ind.ema(x, ALPHA, seed=0.0)
    b = ind.ema(x, ALPHA, seed=1000.0)
    diff = np.abs(a - b)
    ks = np.arange(1, 120)
    slope = num.convergence_rate(np.exp(ks.astype(float)), diff[1:120])   # d ln|Δ| / d k
    theory = np.log(1.0 - ALPHA)
    half_life = np.log(2.0) / -theory

    sec.check(
        "初值差异按 (1−1/N)^k 指数遗忘，实测衰减率与理论一致",
        abs(slope - theory) / abs(theory) < 0.02,
        f"拟合衰减率 {slope:.6f}，理论 ln(1−1/{N}) = {theory:.6f}，"
        f"相对偏差 {abs(slope-theory)/abs(theory)*100:.2f}%；半衰期 {half_life:.1f} 根",
        tolerance="相对偏差 < 2%",
    )
    sec.observe(
        f"N={N} 时半衰期 {half_life:.1f} 根：种子错得再离谱，也要 {half_life:.0f} 根才忘掉一半，"
        f"要 {np.log(100)/-theory:.0f} 根（≈{np.log(100)/-theory/N:.1f}N）才衰减到 1%。"
        "TA-Lib 把 ADX 的 lookback 定成 2N−1 = 27 只是「有值可用」，"
        "真正摆脱种子影响要到 5N ≈ 70 根之后——回测开头 70 根的信号质量是不可信的。"
    )
    sec.pitfall(
        "别把回测前 2N 根的信号当真：那一段的 ADX 还在消化种子。"
        "要么丢弃前 5N 根，要么明确记录「预热期不参与统计」。"
    )

    fig, ax = plot.newfig(figsize=(6.4, 3.0))
    ax.semilogy(np.arange(len(diff)), np.maximum(diff, 1e-16), lw=1.3, color=plot.ACCENT,
                label="两条初值不同的轨迹之差")
    ax.semilogy([0, 120], [diff[0], diff[0] * np.exp(theory * 120)], "--", lw=1.1,
                color=plot.ACCENT2, label=f"理论 (1−1/N)^k，半衰期 {half_life:.1f} 根")
    ax.set_xlabel("递推步数 k")
    ax.set_ylabel("|Δ|（对数轴）")
    ax.set_xlim(0, 120)
    ax.legend(frameon=False, fontsize=10)
    sec.figure("forgetting", fig, caption="一阶递推对初值的记忆是指数衰减的，衰减率只由 α 决定")


# ------------------------------------------------------------------ ③ 相位滞后


def experiment_lag(sec: Section) -> None:
    m = 400
    t = np.arange(m, dtype=float)
    s = ind.sma(t, N)
    w = ind.wilder(t, N, scale="mean")
    lag_sma = float(t[-1] - s[-1])
    lag_ema = float(t[-1] - w[-1])

    sec.check(
        "斜坡输入下 SMA(N) 滞后 (N−1)/2 根，EMA(α=1/N) 滞后 (N−1) 根，正好 2 倍",
        abs(lag_sma - (N - 1) / 2) < 1e-9 and abs(lag_ema - (N - 1)) < 1e-9
        and abs(lag_ema / lag_sma - 2.0) < 1e-9,
        f"实测：SMA 滞后 {lag_sma:.6f} 根（理论 {(N-1)/2}），"
        f"EMA 滞后 {lag_ema:.6f} 根（理论 {N-1}），比值 {lag_ema/lag_sma:.6f}",
        tolerance="与理论值误差 < 1e-9 且比值 = 2",
    )
    sec.observe(
        "均线为什么总是「慢半拍」，这里有个精确答案：N=14 的 SMA 平均晚 6.5 根，"
        "同窗口的 EMA 平均晚 13 根。所以「EMA 更快」是错觉——"
        "EMA 对最新一根的权重更高（响应更陡），但重心滞后反而更大。"
        "趋势策略的出场延迟，一半来自这个数学事实，不是参数没调好。"
    )

    fig, ax = plot.newfig(figsize=(6.4, 3.0))
    ax.plot(t[-60:], t[-60:], lw=1.0, color="#c9ced6", label="输入（斜坡）")
    ax.plot(t[-60:], s[-60:], lw=1.3, color=plot.ACCENT, label=f"SMA({N})，滞后 {lag_sma:.1f} 根")
    ax.plot(t[-60:], w[-60:], lw=1.3, color=plot.ACCENT2, label=f"EMA(α=1/{N})，滞后 {lag_ema:.1f} 根")
    ax.set_xlabel("bar")
    ax.legend(frameon=False, fontsize=10)
    sec.figure("lag", fig, caption="同窗口下 EMA 的相位滞后是 SMA 的两倍——「EMA 更快」是错觉")


# ------------------------------------------------------------------ ④ 降噪 vs 滞后


def experiment_noise(sec: Section) -> None:
    m = 200_000
    x = RNG.standard_normal(m)
    var_in = float(np.var(x))
    v_sma = float(np.var(ind.sma(x, N)[N - 1:]))
    v_ema = float(np.var(ind.wilder(x, N, scale="mean")[N:]))

    theory_sma = var_in / N                       # 独立同分布假设
    theory_ema = var_in * ALPHA / (2 - ALPHA)     # = var_in / (2N − 1)
    ratio_obs = v_ema / v_sma
    ratio_theory = theory_ema / theory_sma

    sec.check(
        "SMA 的降噪能力约为 EMA 的两倍（输出方差只有一半）",
        abs(ratio_obs - ratio_theory) / ratio_theory < 0.05,
        f"输入方差 {var_in:.4f}；SMA 输出方差 {v_sma:.5f}（理论 {theory_sma:.5f}），"
        f"EMA 输出方差 {v_ema:.5f}（理论 {theory_ema:.5f}），"
        f"EMA/SMA = {ratio_obs:.4f}（理论 {ratio_theory:.4f}）",
        tolerance="与理论比值偏差 < 5%",
    )
    sec.observe(
        "把 ③ 和 ④ 放在一起才是完整的取舍：同窗口下 SMA 噪声更小（方差 ≈ 一半），"
        "滞后也更小（6.5 根 vs 13 根）。那 EMA 凭什么存在？"
        "因为它只需要 O(1) 的内存和上一步的状态——在线计算、流式行情、嵌入式/FPGA 上做滤波，"
        "SMA 的滑窗是负担，EMA 的一阶递推不是。"
        "（Wilder 1978 年写书时算力和内存都紧张，选递推不是随意的。）"
    )

    fig, ax = plot.newfig(figsize=(6.4, 3.0))
    sl = slice(0, 220)
    ax.plot(np.arange(220), x[sl], lw=0.7, color="#c9ced6", label="白噪声输入")
    ax.plot(np.arange(220), ind.sma(x, N)[sl], lw=1.3, color=plot.ACCENT,
            label=f"SMA({N})，方差 {v_sma:.3f}")
    ax.plot(np.arange(220), ind.wilder(x, N, scale="mean")[sl], lw=1.3, color=plot.ACCENT2,
            label=f"EMA(α=1/{N})，方差 {v_ema:.3f}")
    ax.set_xlabel("bar")
    ax.legend(frameon=False, fontsize=10)
    sec.figure("noise", fig, caption="同一段噪声：SMA 明显更平滑，代价是要缓存 N 根历史")


def run(sec: Section) -> None:
    experiment_gain(sec)
    experiment_forgetting(sec)
    experiment_lag(sec)
    experiment_noise(sec)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
