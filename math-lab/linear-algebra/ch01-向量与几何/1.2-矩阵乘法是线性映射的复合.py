"""1.2 矩阵乘法：它是线性映射的复合

书上结论：(AB)x = A(Bx)，所以"先 B 后 A"这件事写成乘法；但 AB ≠ BA。
本节要判决的四件事：
  ① (AB)x 与 A(Bx) 在浮点里是否真的一样
  ② 不可交换到什么程度（不是"有时不等"，而是"基本都差得离谱"）
  ③ 结合律在纯数学里成立，在浮点里不成立 —— 差多少，什么时候会咬人
  ④ 既然结果一样，为什么实现顺序依然重要 —— 因为代价不一样
"""
from __future__ import annotations

import time

import numpy as np

from mathviz import Section, num, plot

SEC = Section(
    number="1.2",
    chapter="第 1 章 · 矩阵与线性映射",
    title="矩阵乘法 = 线性映射的复合",
    claim="(AB)x = A(Bx)，复合映射对应矩阵乘积；乘法不可交换，但满足结合律与分配律。",
    proposition=(
        "① 复合等式在浮点下成立到机器精度；"
        "② 随机矩阵的 ‖AB-BA‖ 相对量级远大于机器精度；"
        "③ 结合律在浮点下只是近似，且该误差与条件数基本无关（乘法本身是 forward stable）；"
        "④ 不同的乘法结合顺序结果等价，但浮点运算次数能差一个数量级。"
    ),
    source="教材通用口径：矩阵乘法定义与运算律一节",
)


def experiment_composition(sec: Section, rng: np.random.Generator) -> None:
    """① 复合：左乘一个乘积 == 依次左乘。"""
    n, trials = 40, 2000
    worst = 0.0
    for _ in range(trials):
        A = rng.standard_normal((n, n))
        B = rng.standard_normal((n, n))
        x = rng.standard_normal(n)
        worst = max(worst, num.rel_error((A @ B) @ x, A @ (B @ x)))
    sec.check(
        "(AB)x 与 A(Bx) 的差别在机器精度内",
        worst < 1e-12,
        f"2000 次试验最大相对差 {worst:.3e}",
        tolerance="< 1e-12",
    )

    A = rng.standard_normal((12, 12))
    B = rng.standard_normal((12, 12))
    worst_t = num.rel_error((A @ B).T, B.T @ A.T)
    sec.check("(AB)^T = B^T A^T", worst_t < 1e-14, f"相对差 {worst_t:.3e}", tolerance="< 1e-14")


def experiment_commutativity(sec: Section, rng: np.random.Generator) -> np.ndarray:
    """② 不可交换不是"偶尔"，是"普遍"。"""
    n, trials = 8, 3000
    A = rng.standard_normal((trials, n, n)) * 10
    B = rng.standard_normal((trials, n, n)) * 10
    na = np.linalg.norm(A, axis=(1, 2))
    nb = np.linalg.norm(B, axis=(1, 2))
    diff = np.linalg.norm(A @ B - B @ A, axis=(1, 2)) / (na * nb)
    frac_big = float((diff > 1e-3).mean())
    frac_zero = float((diff < 1e-12).mean())
    sec.check(
        "随机方阵几乎从不交换",
        frac_big > 0.95 and frac_zero < 0.01,
        f"相对差 >1e-3 的占 {frac_big:.1%}，巧合交换（<1e-12）的占 {frac_zero:.2%}",
        tolerance=">95% 明显不交换 / <1% 巧合",
    )
    return diff


def random_with_cond(rng: np.random.Generator, n: int, cond: float) -> np.ndarray:
    """构造指定条件数的方阵：SVD 反解，奇异值从 1 铺到 1/cond。"""
    Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
    Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
    s = np.logspace(0, -np.log10(cond), n)
    return Q1 @ np.diag(s) @ Q2.T


def experiment_associativity(sec: Section, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """③ 结合律在浮点里是近似的 —— 但误差跟条件数没关系，这是容易被讲错的一点。"""
    n, trials = 12, 40
    worst = 0.0
    for _ in range(trials):
        A = rng.standard_normal((n, n)) * 10
        B = rng.standard_normal((n, n))
        C = rng.standard_normal((n, n))
        worst = max(worst, num.rel_error((A @ B) @ C, A @ (B @ C)))
    sec.check(
        "结合律在浮点下只是近似成立（差距确实非零）",
        worst > 1e-17,
        f"{trials} 组随机矩阵最大相对差 {worst:.3e}",
        tolerance="需 > 1e-17（即确实不相等）",
    )

    # 把条件数从 1 拉到 1e12，看乘法误差跟不跟
    conds, errs = [], []
    for kappa in 10.0 ** np.arange(0, 13, 2):
        A = random_with_cond(rng, n, kappa)
        B = rng.standard_normal((n, n))
        C = rng.standard_normal((n, n))
        conds.append(float(np.linalg.cond(A)))
        errs.append(num.rel_error((A @ B) @ C, A @ (B @ C)))
    conds, errs = np.asarray(conds), np.asarray(errs)
    growth = float(errs[-1] / max(errs[0], 1e-300))
    sec.check(
        "乘法的浮点误差不随条件数爆炸（forward stable）",
        growth < 20.0,
        f"cond(A) 从 {conds[0]:.0e} 拉到 {conds[-1]:.0e}，相对误差只增长 {growth:.1f} 倍"
        f"（{errs[0]:.2e} → {errs[-1]:.2e}）",
        tolerance="< 20 倍",
    )
    return conds, errs


def flops_chain(dims: tuple, mode: str) -> int:
    """(m×k)(k×n)(n×p) 两种结合顺序的浮点乘法次数。"""
    (m, k), (_, n), (_, p) = dims
    if mode == "left":
        return 2 * m * k * n + 2 * m * n * p          # (AB)C
    return 2 * k * n * p + 2 * m * k * p              # A(BC)


def experiment_cost(sec: Section, rng: np.random.Generator) -> None:
    """④ 结果一样，代价差很多 —— 这是实现层面绕不开的一条。"""
    best = None
    for m, k, n, p in [(200, 50, 10, 100), (10, 100, 5, 50), (100, 1000, 100, 1),
                       (500, 20, 500, 1), (300, 40, 300, 2)]:
        dims = ((m, k), (k, n), (n, p))
        left, right = flops_chain(dims, "left"), flops_chain(dims, "right")
        ratio = left / max(right, 1)
        if best is None or ratio > best[4]:
            best = (dims, left, right, None, ratio)

    dims, left, right, _, ratio = best
    sec.check(
        "存在代价相差一个数量级的结合顺序",
        ratio >= 5.0,
        f"形状 {dims[0][0]}×{dims[0][1]}, {dims[1][0]}×{dims[1][1]}, {dims[2][0]}×{dims[2][1]}："
        f"(AB)C = {left:,} FLOPs，A(BC) = {right:,} FLOPs，差 {ratio:.1f} 倍",
        tolerance="≥ 5 倍",
    )

    # 代数算完还得跑一遍才敢信：把所有维度同乘 s 倍（两种顺序的 FLOPs 都乘 s³，比值不变）
    (m, k), (_, n), (_, p) = dims
    s = max(1, int(round((2e8 / max(left, right)) ** (1 / 3))))
    A = rng.standard_normal((m * s, k * s))
    B = rng.standard_normal((k * s, n * s))
    C = rng.standard_normal((n * s, p * s))
    t0 = time.perf_counter()
    _ = (A @ B) @ C
    t_left = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = A @ (B @ C)
    t_right = time.perf_counter() - t0
    sec.check(
        "实测耗时与 FLOPs 预测同号",
        (t_left > t_right) == (left > right),
        f"放大 {s} 倍后：(AB)C 实测 {t_left * 1e3:.1f} ms，A(BC) 实测 {t_right * 1e3:.1f} ms",
        tolerance="方向一致即可",
    )
    sec.info(
        "衍生结论",
        "np.matmul / torch.mm 不会帮你选顺序；einsum 和 opt_einsum 才会。"
        "这条不是数学，是把数学写成代码时必须知道的部分。",
    )


def run(sec: Section) -> None:
    rng = np.random.default_rng(7)

    experiment_composition(sec, rng)
    diff = experiment_commutativity(sec, rng)
    conds, errs = experiment_associativity(sec, rng)
    experiment_cost(sec, rng)

    # ---- 图 1：不可交换是常态
    fig, ax = plot.newfig(figsize=(6.2, 3.0))
    ax.hist(diff, bins=70, color=plot.ACCENT, alpha=0.9)
    ax.set_xlabel(r"$\|AB-BA\|_F\,/\,(\|A\|_F\|B\|_F)$")
    ax.set_ylabel("样本数")
    ax.set_title("3000 组随机 8×8 方阵：几乎没有一组能交换")
    sec.figure("commutativity", fig, caption="若交换律成立，直方图应该全挤在最左边那一个 bin 里")

    # ---- 图 2：把条件数拉上去，乘法误差基本不跟
    fig, ax = plot.newfig(figsize=(6.2, 3.2))
    ax.semilogx(conds, errs, "o-", color=plot.ACCENT, lw=1.2, ms=4)
    ax.axhline(float(np.mean(errs)), ls="--", lw=1.1, color=plot.ACCENT2,
               label=f"均值 {np.mean(errs):.1e}")
    ax.set_xlabel("矩阵条件数 cond(A)")
    ax.set_ylabel("(AB)C 与 A(BC) 的相对差")
    ax.set_ylim(0, max(errs) * 2.2)
    ax.legend(frameon=False)
    sec.figure("associativity", fig,
               caption="条件数拉到 1e12，乘法误差几乎纹丝不动 —— 出错的是「求解」，不是「相乘」")

    sec.observe(
        "我原本以为「结合律误差会随 cond(A) 放大」，实测并不成立："
        "把 cond(A) 从 1 拉到 1e12，两种结合顺序的相对差只在一个数量级里抖动。"
        "原因很直接 —— 矩阵乘法是 forward stable，它的误差上界里出现的是 ‖A‖·‖B‖ 的乘积量级，"
        "跟条件数无关。条件数真正发威的场合是「求逆 / 解方程」，也就是下一个小节。"
        "把这两件事分清楚，是看懂数值线性代数的第一步。"
    )
    sec.observe(
        "④ 那条是纯工程的：三个矩阵连乘，(AB)C 和 A(BC) 数学上完全等价，"
        "代价却能差几十倍。numpy / PyTorch 的 matmul 不会替你重排，einsum + opt_einsum 才会。"
    )
    sec.pitfall(
        "单元测试里别用严格等号去卡 A@B@C 与 A@(B@C)，它们确实不相等；"
        "也别指望靠它判断矩阵病态 —— 要判断病态，去 1.3 节看残差与误差的分道扬镳。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
