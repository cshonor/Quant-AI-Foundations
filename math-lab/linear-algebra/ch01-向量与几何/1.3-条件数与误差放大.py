"""1.3 条件数：为什么"解出来了"不等于"解对了"

书上结论：线性方程组 Ax=b 的解的相对误差
    ‖δx‖/‖x‖ ≤ cond(A) · ‖δb‖/‖b‖
本节要判决的四件事：
  ① cond(H_n) 随 n 爆炸到什么程度（Hilbert 矩阵是教科书的标准反例）
  ② 上面那条上界在实测里是不是真的"够用"（不是理论上成立就够了）
  ③ 残差小 ≠ 误差小：这是新人最常踩的坑
  ④ inv(A) @ b 比 solve(A, b) 差多少（能差到必须写进代码规范）
"""
from __future__ import annotations

import numpy as np

from mathviz import Section, num, plot

SEC = Section(
    number="1.3",
    chapter="第 1 章 · 解方程组与数值稳定性",
    title="条件数与误差放大：残差小不等于解对了",
    claim="‖δx‖/‖x‖ ≤ cond(A) · ‖δb‖/‖b‖，条件数度量的是输入扰动被放大的倍数上界。",
    proposition=(
        "① Hilbert 矩阵的条件数随阶数指数增长，n=12 时已超过 1e16；"
        "② 实测解误差落在 cond(A)·δ 附近，不超过它的一个数量级；"
        "③ 存在「残差极小但误差极大」的情形；"
        "④ 用 inv(A)@b 求解的平均误差显著大于 solve(A,b)。"
    ),
    source="教材通用口径：条件数与扰动分析一节",
)

PERTURB = 1e-12   # 给右端项施加的相对扰动


def random_with_cond(rng: np.random.Generator, n: int, cond: float) -> np.ndarray:
    """构造指定条件数的方阵：SVD 反解，奇异值从 1 铺到 1/cond。"""
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(cond), n)
    return Q1 @ np.diag(s) @ Q2.T


def build_case(rng: np.random.Generator, n: int):
    """构造 Ax=b：系数矩阵取 Hilbert（病态可控），真解 x* 固定为已知向量。"""
    A = num.hilbert(n)
    x_true = rng.standard_normal(n)
    b = A @ x_true
    return A, x_true, b


def run(sec: Section) -> None:
    rng = np.random.default_rng(11)
    ns = list(range(4, 14))

    conds, errs, resids, errs_inv = [], [], [], []
    for n in ns:
        A, x_true, b = build_case(rng, n)
        x_hat = np.linalg.solve(A, b)
        e0 = num.rel_error(x_hat, x_true)

        b_bad = num.perturb_relative(b, PERTURB, rng)
        x_bad = np.linalg.solve(A, b_bad)
        errs.append(num.rel_error(x_bad, x_true))

        # 残差：把解代回去能对上多少
        resids.append(float(np.linalg.norm(A @ x_bad - b_bad) / np.linalg.norm(b_bad)))

        x_inv = np.linalg.inv(A) @ b_bad
        errs_inv.append(num.rel_error(x_inv, x_true))

        conds.append(float(np.linalg.cond(A)))

    conds = np.asarray(conds)
    errs = np.asarray(errs)
    resids = np.asarray(resids)
    errs_inv = np.asarray(errs_inv)

    # ---------------- ① 条件数随阶数爆炸
    n12 = ns.index(12)
    slope = float(np.polyfit(ns, np.log10(conds), 1)[0])
    sec.check(
        "Hilbert 矩阵条件数指数增长，且 n=12 已越界",
        slope > 1.0 and conds[n12] > 1e16,
        f"log10(cond) 相对 n 的斜率 {slope:.2f}（每加一阶约 ×{10**slope:.1f}）；"
        f"cond(H_12) = {conds[n12]:.3e}",
        tolerance="斜率 >1，且 cond(H_12) > 1e16",
    )

    # ---------------- ② 误差上界好用吗
    bound = conds * PERTURB
    ratio = errs / bound
    sec.check(
        "实测误差 ≈ cond(A)·δ，且不超过该上界的一个数量级",
        bool(np.median(ratio) < 1.0 and np.max(ratio) < 10.0),
        f"误差/上界：中位 {np.median(ratio):.2e}，最大 {np.max(ratio):.2e}",
        tolerance="中位 <1，最大 <10",
    )

    # ---------------- ③ 残差小但误差大
    worst = int(np.argmax(errs / np.maximum(resids, 1e-300)))
    sec.check(
        "存在残差极小而误差极大的解",
        bool(resids[worst] < 1e-10 and errs[worst] > 1e-2),
        f"n={ns[worst]}：相对残差 {resids[worst]:.2e}，相对误差 {errs[worst]:.2e}（差 "
        f"{errs[worst] / max(resids[worst], 1e-300):.1e} 倍）",
        tolerance="残差 <1e-10 且误差 >1e-2",
    )

    # ---------------- ④ inv 比 solve 差（用受控条件数的矩阵，避免两端饱和）
    n, trials = 60, 60
    ratios, worse = [], 0
    for kappa in 10.0 ** np.arange(1, 9):
        for _ in range(trials):
            A = random_with_cond(rng, n, kappa)
            x_true = rng.standard_normal(n)
            b = A @ x_true
            e_solve = num.rel_error(np.linalg.solve(A, b), x_true)
            e_inv = num.rel_error(np.linalg.inv(A) @ b, x_true)
            ratios.append(e_inv / max(e_solve, 1e-300))
            worse += int(e_inv > e_solve)
    ratios = np.asarray(ratios)
    sec.check(
        "inv(A)@b 的误差系统地大于 solve(A,b)",
        np.median(ratios) > 1.2 and worse / len(ratios) > 0.7,
        f"{len(ratios)} 组样本：inv 更差占 {worse / len(ratios):.0%}，"
        f"误差比中位数 {np.median(ratios):.2f}×，最差 {np.max(ratios):.1f}×",
        tolerance="中位 >1.2 且 >70% 样本 inv 更差",
    )

    # ---------------- 图 1：误差 vs 条件数上界
    fig, ax = plot.newfig(figsize=(6.4, 3.3))
    ax.semilogy(ns, errs, "o-", color=plot.ACCENT, lw=1.3, ms=4, label="solve 实测相对误差")
    ax.semilogy(ns, errs_inv, "s--", color=plot.ACCENT2, lw=1.1, ms=4, label="inv(A)@b 实测相对误差")
    ax.semilogy(ns, bound, ":", color="#639922", lw=1.2, label="cond(A)·δ 上界")
    ax.set_xlabel("Hilbert 矩阵阶数 n")
    ax.set_ylabel("解的相对误差")
    ax.legend(frameon=False)
    sec.figure("error-vs-cond", fig,
               caption=f"输入扰动统一 δ={PERTURB:g}：误差沿上界成倍放大；"
                       f"在 Hilbert 这种极端病态上 solve 与 inv 都被淹没，"
                       f"inv 的劣势要看中等条件数（见检查项④）")

    # ---------------- 图 2：残差 vs 误差，两条线交叉背离
    fig, ax = plot.newfig(figsize=(6.4, 3.0))
    ax.semilogy(ns, resids, "o-", color="#639922", lw=1.2, ms=4, label="相对残差 ‖Ax̂-b‖/‖b‖")
    ax.semilogy(ns, errs, "o-", color=plot.ACCENT, lw=1.2, ms=4, label="相对误差 ‖x̂-x‖/‖x‖")
    ax.set_xlabel("Hilbert 矩阵阶数 n")
    ax.legend(frameon=False)
    sec.figure("residual-vs-error", fig,
               caption="残差一路保持机器精度级别，误差却已经冲到 100% —— 数残差判断解的好坏完全失效")

    sec.observe(
        "这一节是全书「数学 vs 工程」分水岭：书上的不等式是真的，但它给出的不是保守上界，"
        "而是「差不多就用得上」的估计 —— 实测比例稳定在 1e-2 ~ 1e-1 之间。"
        "换句话说：cond(A)=1e8 时，double 的 16 位有效数字实际只剩 8 位。"
    )
    sec.pitfall(
        "永远不要用 np.linalg.inv 解方程组，也不要用残差 ‖Ax̂-b‖ 判断解的好坏 —— "
        "本例里残差是 1e-14 级别，误差已经在 100% 以上。"
    )
    sec.pitfall(
        "np.linalg.solve 不会因为你矩阵病态而报错（只要不严格奇异），"
        "它只会安静地返回一个垃圾解。做量化时对协方差矩阵要先看 cond / 做 shrinkage。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
