"""10.4 资金曲线：连乘、几何平均与「收益一样但回撤完全不同」

净值 V_T = Π(1+r_t)。这个连乘式有三个直接后果，每一个都能被测出来：
  1. 连乘可交换 ⇒ 收益的**顺序**不影响终值；
  2. 但 max(V)/V 依赖路径 ⇒ 顺序强烈影响最大回撤；
  3. 几何平均 ≤ 算术平均，差额约 σ²/2 ⇒ 波动本身在吃掉收益（波动拖累）。
第 4 件事是诚实问题：Sharpe 也是一个估计量，有标准误，3 年日线根本分辨不出 1.0 和 0.3。
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
    number="10.4",
    chapter="CTA 专题 · 资金曲线与风险度量",
    book="贯穿案例：CTA 趋势策略（BTCUSDT 日线）",
    title="资金曲线：连乘可交换，回撤却依赖路径",
    claim="净值 = Π(1+r_t)；总收益看终值，风险看 max(1 − V_t / running_max)。"
          "Sharpe = 年化均值 / 年化标准差。",
    proposition="① 几何均值 ≤ 算术均值，差额 ≈ σ²/2（波动拖累，实测与理论吻合）；"
                "② 打乱收益顺序：终值不变（连乘可交换），最大回撤的分布却明显散开；"
                "③ 回撤 d 之后需要涨 d/(1−d) 才回本——当 d > 50% 时这个倍数超过 2；"
                "④ Sharpe 的标准误 √(1+SR²/2)/T · √365 可由 bootstrap 复现，"
                "3 年日线的标准误大约在 0.7 量级。",
    source="AM–GM 不等式 / Lo (2002) The Statistics of Sharpe Ratios",
)

N_YEARS = 3.0
PPY = 365
RNG = np.random.default_rng(2718281)


# ------------------------------------------------------------------ ① 波动拖累


def experiment_volatility_drag(sec: Section, close: np.ndarray) -> None:
    r = close[1:] / close[:-1] - 1.0
    mu_a = float(np.mean(r))
    mu_g = float(np.exp(np.mean(np.log(1.0 + r))) - 1.0)
    var = float(np.var(r, ddof=1))
    gap, theory = mu_a - mu_g, var / 2.0

    sec.check(
        "算术均值 − 几何均值 ≈ σ²/2（波动拖累可量化）",
        abs(gap - theory) / theory < 0.30,
        f"算术日均 {mu_a*100:.4f}%，几何日均 {mu_g*100:.4f}%，差 {gap*1e6:.2f} bp/日；"
        f"σ²/2 = {theory*1e6:.2f} bp/日，相对偏差 {abs(gap-theory)/theory*100:.1f}%；"
        f"按 365 天折算，波动每年吃掉约 {(1-np.exp(-theory*PPY))*100:.1f}%",
        tolerance="与理论偏差 < 30%",
    )
    sec.observe(
        "「平均每天赚 0.05%，一年应该赚 18%」——实测要打折，折扣就是 σ²/2。"
        "波动越大，几何收益被算术收益甩得越远。这条对 CTA 特别要命："
        "趋势策略大部分时间在小亏小赚里震荡，靠少数几波大趋势赚钱，"
        "σ 的天花板比 σ 的平均值更能决定你最后拿到多少。"
    )


# ------------------------------------------------------------------ ② 顺序：终值不变，回撤全变


def experiment_order(sec: Section, close: np.ndarray) -> None:
    r = close[1:] / close[:-1] - 1.0
    equity = np.cumprod(1.0 + r)
    dd_base = ind.max_drawdown(equity)
    finals, dds = [], []
    for _ in range(300):
        p = RNG.permutation(r)
        e = np.cumprod(1.0 + p)
        finals.append(e[-1])
        dds.append(ind.max_drawdown(e))
    finals, dds = np.array(finals), np.array(dds)
    spread_final = float(np.std(finals) / np.mean(finals))
    cv_dd = float(np.std(dds) / np.mean(dds))

    sec.check(
        "打乱收益顺序：终值不变（连乘可交换），但最大回撤明显改变",
        spread_final < 1e-9 and cv_dd > 0.05,
        f"300 次打乱：终值相对标准差 {spread_final:.2e}（可交换性 ⇒ 恒定），"
        f"最大回撤均值 {np.mean(dds)*100:.1f}%、标准差 {np.std(dds)*100:.1f}%、"
        f"变异系数 {cv_dd*100:.1f}%（原始顺序下 {dd_base*100:.1f}%）",
        tolerance="终值相对标准差 < 1e-9 且回撤变异系数 > 5%",
    )
    sec.observe(
        "同一个收益集合（同一策略、同一段行情），只改顺序，终值一模一样，回撤却能从 "
        f"{np.min(dds)*100:.0f}% 变到 {np.max(dds)*100:.0f}%。"
        "⇒ 用「最终收益」选策略是无效的（它对路径完全不敏感），"
        "而用「最大回撤」选策略要非常小心（它对路径极度敏感，样本外几乎不复现）。"
        "这也是为什么单看一条资金曲线评价策略 = 看一张图的形状下结论——正是 10.1 说的陷阱。"
    )
    sec.pitfall(
        "回撤类指标（MaxDD、Calmar、 ulcer index）都是路径统计量，"
        "样本内的数值基本不可外推。要比较就用「收益集合不变、只改顺序」做零假设，"
        "看你的策略是否真的比随机顺序更好——这一步能挡掉一大半自欺。"
    )

    fig, ax = plot.newfig(figsize=(6.4, 3.0))
    ax.hist(dds * 100, bins=28, color=plot.ACCENT, alpha=0.85, label="打乱顺序后的最大回撤")
    ax.axvline(dd_base * 100, color=plot.ACCENT2, lw=1.6, label=f"真实顺序 {dd_base*100:.1f}%")
    ax.set_xlabel("最大回撤 %")
    ax.set_ylabel("频次")
    ax.legend(frameon=False, fontsize=10)
    sec.figure("shuffle", fig,
               caption="收益集合完全相同、只改顺序：终值恒定，最大回撤却散成一条分布")


# ------------------------------------------------------------------ ③ 回撤恢复的非线性


def experiment_recovery(sec: Section) -> None:
    df_all = cta_data.load(None)                     # 全量历史，含 2021–2022 熊市
    close = df_all["Close"].to_numpy(float)
    d = ind.max_drawdown(close)
    need = d / (1.0 - d)
    sec.check(
        "回撤 d 之后需要涨 d/(1−d) 才回本；d > 50% 时所需涨幅超过回撤本身的 2 倍",
        need / d > 2.0,
        f"全样本最大回撤 {d*100:.1f}% ⇒ 需要上涨 {need*100:.0f}% 才回到前高，"
        f"是回撤幅度的 {need/d:.2f} 倍",
        tolerance="d > 50% 且倍数 > 2",
    )
    ds = np.linspace(0.05, 0.9, 200)
    fig, ax = plot.newfig(figsize=(6.4, 3.0))
    ax.plot(ds * 100, ds / (1 - ds) * 100, lw=1.4, color=plot.ACCENT)
    ax.plot(ds * 100, ds * 100, "--", lw=1.0, color="#c9ced6", label="线性参照 y = x")
    ax.plot([d * 100], [need * 100], "o", ms=6, color=plot.ACCENT2,
            label=f"实测最大回撤 {d*100:.0f}% → 需涨 {need*100:.0f}%")
    ax.set_xlabel("回撤 %")
    ax.set_ylabel("回本所需涨幅 %")
    ax.legend(frameon=False, fontsize=10)
    sec.figure("recovery", fig, caption="回撤与回本涨幅不是线性关系：亏 50% 要涨 100%，亏 80% 要涨 400%")


# ------------------------------------------------------------------ ④ Sharpe 是个估计量


def experiment_sharpe_error(sec: Section, close: np.ndarray) -> None:
    r = close[1:] / close[:-1] - 1.0
    sr = ind.sharpe(r, PPY)
    t = len(r)
    se_theory = np.sqrt((1.0 + 0.5 * sr ** 2) / t) * np.sqrt(PPY)

    boot = []
    for _ in range(400):
        idx = RNG.integers(0, t, t)
        boot.append(ind.sharpe(r[idx], PPY))
    se_boot = float(np.std(boot, ddof=1))

    sec.check(
        "Sharpe 的标准误 √(1+SR²/2)/T·√365 可由 bootstrap 复现",
        abs(se_boot - se_theory) / se_theory < 0.40,
        f"T={t} 根，年化 Sharpe = {sr:.2f}；理论标准误 {se_theory:.2f}，"
        f"bootstrap 实测 {se_boot:.2f}，相对偏差 {abs(se_boot-se_theory)/se_theory*100:.1f}%",
        tolerance="相对偏差 < 40%",
    )
    sec.observe(
        f"3 年日线算出来的 Sharpe = {sr:.2f}，但它的标准误就有 {se_theory:.2f}。"
        f"也就是说 95% 置信区间大约是 [{sr-1.96*se_theory:.2f}, {sr+1.96*se_theory:.2f}]——"
        "这个宽度足以同时容纳「不错的策略」和「白捡的运气」。"
        "任何拿 3 年日线跑出来的 Sharpe 差异（0.8 vs 1.2）去做参数选择，都是在拟合噪声。"
    )
    sec.pitfall(
        "报 Sharpe 必须同时报 T 和标准误。判据：要分辨两个 Sharpe 相差 Δ，"
        "大约需要 T ≈ 365·(1+SR²/2)·(1.96/Δ)² 根日线。"
        "想分辨 Δ=0.4（SR=1 时），需要约 9000 根日线 ≈ 25 年——BTC 都没有这么长的历史。"
    )


def run(sec: Section) -> None:
    df = cta_data.load(N_YEARS)
    close = df["Close"].to_numpy(float)
    sec.info("数据", f"{len(df)} 根 BTCUSDT 日线，{df.index.min().date()} → {df.index.max().date()}")
    experiment_volatility_drag(sec, close)
    experiment_order(sec, close)
    experiment_recovery(sec)
    experiment_sharpe_error(sec, close)


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
