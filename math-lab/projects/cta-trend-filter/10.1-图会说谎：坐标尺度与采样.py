"""10.1 图会说谎：坐标、尺度与采样密度

动机：学数学要靠图形建立直觉——这句是对的。但"图"是一个经过三重选择的产物：
选什么坐标、选什么尺度、选多密的采样点。这三重选择每一个都能把结论翻转，
而图本身不会告诉你它翻了。本节用 CTA 的真实行情把三次翻转各做一遍。
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
from mathviz import Section, num, plot  # noqa: E402

SEC = Section(
    number="10.1",
    chapter="CTA 专题 · 图形直觉的边界",
    book="贯穿案例：CTA 趋势策略（BTCUSDT 日线）",
    title="图会说谎：坐标、尺度与采样密度",
    claim="「数学要靠图形理解」——图是建立直觉的入口，但图本身不提供判决："
          "换坐标轴、换尺度、换采样频率，同一份数据能给出相反的结论。",
    proposition="① 存在回撤事件，其「绝对金额」排序与「百分比」排序相反（线性轴与对数轴结论冲突）；"
                "② 同一序列降采样后，最大回撤被系统性低估，且采样越粗低估越多（单调）；"
                "③ 采样间隔超过 Nyquist 界时，图上看到的周期是一个根本不存在的周期。",
    source="Nyquist–Shannon 采样定理 / 对数坐标的尺度不变性",
)

N_YEARS = 3.0


# ------------------------------------------------------------------ ① 线性轴 vs 对数轴


def episode_drawdowns(close: np.ndarray) -> list[dict]:
    """把回撤切成一段段「从峰值到谷底」的事件，每段记录最大绝对跌幅与最大百分比跌幅。"""
    peak = np.maximum.accumulate(close)
    dd_abs = peak - close
    dd_pct = dd_abs / peak
    episodes, in_dd = [], False
    for i in range(len(close)):
        if dd_abs[i] > 0 and not in_dd:
            in_dd, start = True, i
            cur = {"start": start, "abs": dd_abs[i], "pct": dd_pct[i], "i_abs": i, "i_pct": i}
        elif dd_abs[i] > 0:
            if dd_abs[i] > cur["abs"]:
                cur["abs"], cur["i_abs"] = dd_abs[i], i
            if dd_pct[i] > cur["pct"]:
                cur["pct"], cur["i_pct"] = dd_pct[i], i
        elif in_dd:
            episodes.append(cur)
            in_dd = False
    if in_dd:
        episodes.append(cur)
    return episodes


def experiment_scale(sec: Section, df) -> None:
    close = df["Close"].to_numpy(float)
    eps = [e for e in episode_drawdowns(close) if e["pct"] > 0.02]      # 忽略 2% 以下的毛刺
    inversions = sum(
        1 for a in eps for b in eps
        if a is not b and a["abs"] > b["abs"] and a["pct"] < b["pct"]
    )
    pairs = len(eps) * (len(eps) - 1)
    top_abs = max(eps, key=lambda e: e["abs"])
    top_pct = max(eps, key=lambda e: e["pct"])
    dates = df.index

    # 两种排序的一致性：Spearman ρ = 1 才是「线性轴与对数轴结论一致」
    abss = np.array([e["abs"] for e in eps])
    pcts = np.array([e["pct"] for e in eps])

    def rank_desc(v: np.ndarray) -> np.ndarray:
        order = np.argsort(-v)
        r = np.empty(len(v), dtype=int)
        r[order] = np.arange(1, len(v) + 1)
        return r

    rank_abs, rank_pct = rank_desc(abss), rank_desc(pcts)
    rho = float(np.corrcoef(rank_abs, rank_pct)[0, 1])

    # 最能说明问题的对照：百分比几乎相同，金额却差好几倍
    pair = None
    for a in eps:
        for b in eps:
            if a is b or a["abs"] <= b["abs"] or abs(a["pct"] / b["pct"] - 1) >= 0.20:
                continue
            score = a["abs"] / b["abs"]
            if pair is None or score > pair[0]:
                pair = (score, a, b)
    ratio, fa, fb = pair if pair else (1.0, top_abs, top_pct)

    sec.check(
        "回撤按「金额」排序与按「百分比」排序不一致（线性轴与对数轴给出不同答案）",
        inversions > 0 and rho < 0.99 and ratio > 2.0,
        f"{len(eps)} 段回撤里 {inversions}/{pairs} 对顺序反转，两种排序的 Spearman ρ = {rho:.3f}；"
        f"最典型的一对：{dates[fa['i_pct']].date()} 跌 {fa['pct']*100:.1f}%（{fa['abs']:,.0f} USDT）、"
        f"{dates[fb['i_pct']].date()} 跌 {fb['pct']*100:.1f}%（{fb['abs']:,.0f} USDT）"
        f"——百分比几乎一样，金额差 {ratio:.1f} 倍",
        tolerance="反转对数 > 0、ρ < 0.99 且对照对金额比 > 2",
    )

    # 纯粹的尺度效应：同一百分比回撤，在不同价位对应的金额差一个价格比
    lo, hi = float(np.min(close)), float(np.max(close))
    pct = 0.20
    sec.check(
        "同一百分比回撤，在价格低点/高点对应的绝对金额之比 = 价格极值之比",
        abs((hi * pct) / (lo * pct) - hi / lo) < 1e-12,
        f"样本内价格 {lo:,.0f} → {hi:,.0f}（{hi/lo:.1f} 倍）："
        f"{pct:.0%} 回撤 = {lo*pct:,.0f} USDT（低位）vs {hi*pct:,.0f} USDT（高位），差 {hi/lo:.1f} 倍",
        tolerance="恒等（误差 < 1e-12）",
    )

    sec.observe(
        f"两次回撤的**百分比几乎一样**（{fa['pct']*100:.1f}% vs {fb['pct']*100:.1f}%），"
        f"但金额差 {ratio:.1f} 倍（{fa['abs']:,.0f} vs {fb['abs']:,.0f} USDT）——"
        "因为一次发生在 10 万价位，另一次在 3 万价位。图上看着一样惨，账户里差 3 倍多。"
        f"（诚实交代：我原以为两种口径的**排序**会大面积冲突，实测 ρ = {rho:.3f} 高度一致，"
        f"只有 {inversions}/{pairs} 对反序——BTC 单边上涨时大回撤在两种口径下同时都大。）"
        f"另一个确定性事实：同一 20% 回撤，在样本低位只值 {lo*pct:,.0f} USDT、"
        f"高位值 {hi*pct:,.0f} USDT（价格跨 {hi/lo:.1f} 倍）。"
        "⇒ 资金曲线默认画对数轴，回撤看百分比，止损距离用 ATR 这类相对量而不是固定点数。"
    )

    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.0))
    for ax, logy, tag in ((axes[0], False, "线性轴：跌幅以「点」计"),
                          (axes[1], True, "对数轴：跌幅以「倍」计")):
        ax.plot(dates, close, lw=1.0, color=plot.ACCENT)
        for e, col, lab in ((fa, plot.ACCENT2, f"{fa['pct']*100:.1f}% 回撤 = {fa['abs']:,.0f} USDT"),
                            (fb, "#639922", f"{fb['pct']*100:.1f}% 回撤 = {fb['abs']:,.0f} USDT")):
            ax.plot([dates[e["start"]], dates[e["i_pct"]]], [close[e["start"]], close[e["i_pct"]]],
                    marker="o", ms=4, lw=1.8, color=col, label=lab)
        if logy:
            ax.set_yscale("log")
        ax.set_title(tag, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    sec.figure("scale", fig, caption="换一个坐标轴，「哪一次跌得最惨」的答案就变了")


# ------------------------------------------------------------------ ② 降采样低估回撤


def max_dd(series: np.ndarray) -> float:
    peak = np.maximum.accumulate(series)
    return float(np.max(1.0 - series / peak))


def experiment_sampling(sec: Section, df) -> None:
    daily = df["Close"]
    weekly = daily.resample("W").last()
    monthly = daily.resample("ME").last()
    dd = (max_dd(daily.to_numpy(float)), max_dd(weekly.to_numpy(float)),
          max_dd(monthly.to_numpy(float)))

    sec.check(
        "采样越粗，最大回撤被低估越多（日 > 周 > 月，单调）",
        dd[0] > dd[1] > dd[2],
        f"日频 {dd[0]*100:.1f}% → 周频 {dd[1]*100:.1f}% → 月频 {dd[2]*100:.1f}%；"
        f"周频少算 {(dd[0]-dd[1])*100:.1f} 个百分点，月频少算 {(dd[0]-dd[2])*100:.1f} 个百分点",
        tolerance="严格单调递减",
    )

    sec.observe(
        "回撤是「路径最大值的补」，降采样等于把路径上的点抽掉，抽掉的点里就藏着真正的谷底。"
        "这条在 CTA 里的后果很具体：用周线回测风控参数，会系统性低估你将来要挨的回撤。"
    )

    fig, ax = plot.newfig(figsize=(7.2, 3.0))
    for s, name, col in ((daily, "日频", plot.ACCENT),
                         (weekly, "周频", plot.ACCENT2),
                         (monthly, "月频", "#639922")):
        v = s.to_numpy(float)
        ax.plot(s.index, v / v[0], lw=1.2 if name == "日频" else 1.0,
                color=col, label=f"{name}（最大回撤 {max_dd(v)*100:.1f}%）",
                alpha=1.0 if name == "日频" else 0.85)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylabel("归一化净值（对数）")
    sec.figure("resampling", fig, caption="同一段行情：采样频率决定了你能看见多深的回撤")


# ------------------------------------------------------------------ ③ 混叠：图上的周期不存在


def dominant_period(t: np.ndarray, y: np.ndarray) -> float:
    """FFT 找主频 → 周期（天数）。"""
    y = y - y.mean()
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=float(t[1] - t[0]))
    k = int(np.argmax(spec[1:])) + 1
    return 1.0 / freq[k]


def experiment_alias(sec: Section) -> None:
    true_period = 7.0
    dt_sparse, dt_dense = 13.0, 2.0                  # Nyquist 要求 dt < 3.5 天
    t_sparse = np.arange(400) * dt_sparse
    t_dense = np.arange(2600) * dt_dense
    p_sparse = dominant_period(t_sparse, np.sin(2 * np.pi * t_sparse / true_period))
    p_dense = dominant_period(t_dense, np.sin(2 * np.pi * t_dense / true_period))

    # 理论混叠频率：ν = f·dt（cycles/sample），折回 [0, 0.5] 后再除以 dt
    nu = (1.0 / true_period) * dt_sparse
    f_alias = abs(nu - round(nu)) / dt_sparse
    p_theory = 1.0 / f_alias

    sec.check(
        "采样间隔超过 Nyquist 界时，图上量到的周期与真实周期差一个数量级",
        abs(p_sparse - true_period) / true_period > 5.0,
        f"真实 {true_period:.0f} 天；Δt={dt_sparse:.0f}d 稀疏采样量到 {p_sparse:.1f} 天"
        f"（理论混叠值 {p_theory:.1f} 天，吻合度 {abs(p_sparse-p_theory)/p_theory*100:.2f}%），"
        f"相对偏差 {abs(p_sparse-true_period)/true_period*100:.0f}%",
        tolerance="相对偏差 > 500%",
    )
    sec.check(
        "把采样加密到 Nyquist 界以内，周期可被正确还原",
        abs(p_dense - true_period) / true_period < 0.02,
        f"Δt={dt_dense:.0f}d（< Nyquist 界 {true_period/2:.1f}d）量到 {p_dense:.2f} 天，"
        f"误差 {abs(p_dense-true_period)/true_period*100:.2f}%",
        tolerance="相对误差 < 2%",
    )

    sec.observe(
        f"真实信号是 7 天周期，但每 13 天取一个点画出来的图，主频是 {p_sparse:.0f} 天——"
        "这个周期在数据里根本不存在，是采样制造出来的鬼影（aliasing）。"
        "CTA 里的对应场景：拿周线/月线去看季节性、拿 1 分钟线去猜日内的周期，"
        "看到的「规律」很可能只是采样artifact。"
    )
    sec.pitfall(
        "判据很简单也很硬：采样间隔必须 < 目标周期的一半（Nyquist）。"
        "在频谱上找周期之前，先算一遍这个不等式，否则拟合出来的东西没有物理意义。"
    )

    fig, axes = plot.newfig(1, 2, figsize=(7.4, 3.0))
    t_fine = np.arange(0, 120, 0.05)
    for ax, t, dt, p_est, tag in (
        (axes[0], t_sparse, dt_sparse, p_sparse, f"Δt={dt_sparse:.0f}d（超过 Nyquist）"),
        (axes[1], t_dense, dt_dense, p_dense, f"Δt={dt_dense:.0f}d（满足 Nyquist）"),
    ):
        ax.plot(t_fine, np.sin(2 * np.pi * t_fine / true_period), lw=0.8, color="#c9ced6",
                label=f"真实 {true_period:.0f} 天周期")
        cut = t[t <= 120]
        ax.plot(cut, np.sin(2 * np.pi * cut / true_period), "o", ms=3.5, color=plot.ACCENT,
                label="采样点")
        ax.plot(t_fine, np.sin(2 * np.pi * t_fine / p_est), "--", lw=1.2, color=plot.ACCENT2,
                label=f"图上看出来的 {p_est:.1f} 天")
        ax.set_title(tag, fontsize=11)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
    sec.figure("alias", fig, caption="左：采样太稀，量到 91 天的假周期；右：采样够密，还原出 7 天")


def run(sec: Section) -> None:
    df = cta_data.load(N_YEARS)
    sec.info("数据", f"{len(df)} 根 BTCUSDT 日线，{df.index.min().date()} → {df.index.max().date()}")
    experiment_scale(sec, df)
    experiment_sampling(sec, df)
    experiment_alias(sec)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
