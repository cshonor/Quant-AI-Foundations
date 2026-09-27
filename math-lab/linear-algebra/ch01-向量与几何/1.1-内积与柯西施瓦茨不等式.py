"""1.1 内积、夹角与柯西-施瓦茨不等式

书上结论：|u·v| ≤ ‖u‖·‖v‖，等号当且仅当 u, v 线性相关。
本节要判决的三件事：
  ① 随机向量到底有没有"例外"（不等式是否只在数学里成立、在浮点里也成立）
  ② 等号附近收敛有多快（O(ε²) 而不是 O(ε)）
  ③ 工程上照抄公式会踩什么坑（cos 偶尔 > 1）
"""
from __future__ import annotations

import numpy as np

from mathviz import Section, num, plot

SEC = Section(
    number="1.1",
    chapter="第 1 章 · 向量与几何",
    title="内积、夹角与柯西-施瓦茨不等式",
    claim="|u·v| ≤ ‖u‖·‖v‖，等号当且仅当 u, v 线性相关。",
    proposition=(
        "① 随机 20 万组向量里没有任何一组违背该不等式（相对超出 ≤ 1e-14）；"
        "② 逼近共线时 1-|cosθ| 的收敛阶是 O(ε²)；"
        "③ float64 下不同算法路径算出的 cos 存在 eps 量级分歧，工程实现必须 clip。"
    ),
    source="教材通用口径：内积定义 → C-S 不等式一节",
)

N_SAMPLES = 200_000


def _cos(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """逐行计算夹角余弦。"""
    nu = np.linalg.norm(u, axis=1)
    nv = np.linalg.norm(v, axis=1)
    return (u * v).sum(axis=1) / (nu * nv)


def experiment_holds(sec: Section, rng: np.random.Generator) -> dict:
    """① C-S 在随机样本上是否有反例。"""
    worst_rel, hit = -np.inf, 0
    records = {}
    for n in (2, 3, 10, 100):
        u = rng.standard_normal((N_SAMPLES // 4, n))
        v = rng.standard_normal((N_SAMPLES // 4, n))
        lhs = np.abs((u * v).sum(axis=1))
        rhs = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
        rel = (lhs - rhs) / rhs
        worst_rel = max(worst_rel, float(rel.max()))
        hit += int((rel > 1e-14).sum())
        records[n] = np.abs((u * v).sum(axis=1) / rhs)
    sec.check(
        "随机样本违背 C-S 的次数 = 0",
        hit == 0,
        f"全部 ≤ 上界；连最接近等号的一组也差 {abs(worst_rel):.2e} 相对量，反例 {hit} 次",
        tolerance="> 1e-14 计为反例",
    )
    return records


def experiment_equality_speed(sec: Section, rng: np.random.Generator) -> np.ndarray:
    """② 等号附近的收敛速度：教科书只说"等号 iff 共线"，没说多快逼近。"""
    n = 8
    u = rng.standard_normal(n)
    u /= np.linalg.norm(u)
    z = rng.standard_normal(n)
    w = z - (z @ u) * u          # 正交化
    w /= np.linalg.norm(w)
    eps = np.logspace(-1, -6, 40)
    v = np.cos(np.arcsin(eps))[:, None] * u + eps[:, None] * w   # 夹角 sinθ = eps
    c = np.abs(_cos(np.broadcast_to(u, v.shape), v))
    gap = 1.0 - c
    order = num.convergence_rate(eps, gap)
    sec.check(
        "1-|cosθ| 相对 ε 的收敛阶 ≈ 2",
        1.8 <= order <= 2.2,
        f"最小二乘拟合斜率 {order:.3f}",
        tolerance="1.8 ~ 2.2",
    )
    return np.column_stack([eps, gap, order * np.ones_like(eps)])


def experiment_float_paths(sec: Section, rng: np.random.Generator) -> None:
    """③ 三条"等价"的公式在浮点里并不等价。"""
    n, m = 6, 50_000
    u = rng.standard_normal((m, n)) * 1e3
    v = rng.standard_normal((m, n)) * 1e-3      # 故意让量纲差 6 个数量级

    c1 = (u * v).sum(axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    un = u / np.linalg.norm(u, axis=1, keepdims=True)
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    c2 = (un * vn).sum(axis=1)
    c3 = np.einsum("ij,ij->i", u, v) / np.sqrt(
        np.einsum("ij,ij->i", u, u) * np.einsum("ij,ij->i", v, v)
    )
    spread = max(
        float(np.abs(c1 - c2).max()), float(np.abs(c1 - c3).max()),
        float(np.abs(c2 - c3).max()),
    )
    over = int((np.abs(c1) > 1.0).sum())
    sec.check(
        "三条等价公式的最大分歧在 eps 量级",
        spread < 1e-12,
        f"最大分歧 {spread:.3e}，|cos| 越过 1 的样本数 {over}/{m}",
        tolerance="< 1e-12",
    )

    # 构造性地演示后果：值一旦越过 1，反三角函数直接 nan
    bad = float(np.nextafter(1.0, 2.0))          # 1.0 之后的第一个可表示浮点数
    with np.errstate(invalid="ignore"):
        raw = float(np.arccos(bad))
        clipped = float(np.arccos(np.clip(bad, -1.0, 1.0)))
    sec.check(
        "cos 一旦越过 1，arccos 就废掉；clip 可救",
        np.isnan(raw) and clipped == 0.0,
        f"nextafter(1,2)={bad!r}：arccos → {raw}，clip 后 → {clipped};"
        f"（本例 5 万样本里真实越界 {over} 次，需要构造才能稳定复现）",
        tolerance="nan → 0.0",
    )


def experiment_correlation(sec: Section, rng: np.random.Generator) -> None:
    """顺手把"相关系数就是中心化后的夹角余弦"这条也判一下 —— 做量化会天天用到。"""
    x = rng.standard_normal((400, 3)) @ np.array([[1.0, 0.8, 0.0],
                                                  [0.2, 1.0, 0.5],
                                                  [0.0, 0.3, 1.0]])
    xc = x - x.mean(axis=0)
    i, j = 0, 1
    cos_ij = float(xc[:, i] @ xc[:, j] / (np.linalg.norm(xc[:, i]) * np.linalg.norm(xc[:, j])))
    r_ij = float(np.corrcoef(x[:, i], x[:, j])[0, 1])
    sec.check(
        "Pearson r == 中心化后的夹角余弦",
        abs(cos_ij - r_ij) < 1e-12,
        f"cos = {cos_ij:.12f}, r = {r_ij:.12f}, 差 {abs(cos_ij - r_ij):.2e}",
        tolerance="< 1e-12",
    )


def run(sec: Section) -> None:
    rng = np.random.default_rng(20260927)

    records = experiment_holds(sec, rng)
    curve = experiment_equality_speed(sec, rng)
    experiment_float_paths(sec, rng)
    experiment_correlation(sec, rng)

    # ---- 图 1：高维里随机向量几乎正交（C-S 上界远没达到）——这是 ℓ2 高维反直觉的第一课
    fig, ax = plot.newfig(1, 2, figsize=(7.4, 2.9))
    for k, n in enumerate((2, 100)):
        ax[k].hist(records[n], bins=90, color=plot.ACCENT, alpha=0.85)
        ax[k].set_title(f"n = {n}：随机向量夹角余弦分布")
        ax[k].set_xlabel("cos θ")
        if k == 0:
            ax[k].set_ylabel("样本数")
    fig.suptitle("维度越高，随机向量越接近正交 —— C-S 的上界离现实越来越远", fontsize=11)
    sec.figure("cos-distribution", fig,
               caption="同为 5 万个样本对：n=2 时夹角铺满 [0,1] 且两端上翘（服从反正弦律，不是 bug），"
                       "n=100 时几乎全挤在 0 附近")
    sec.observe(
        "图 1 左图 cos 在 ±1 处上翘是数学结果（二维随机方向的夹角余弦服从反正弦律 "
        "1/(π√(1-c²))），不是采样出错 —— 这种「先怀疑代码还是先怀疑直觉」的分辨能力，"
        "正是写代码建直觉要练的东西。"
    )

    # ---- 图 2：等号邻域的收敛速度
    fig, ax = plot.newfig(figsize=(6.0, 3.2))
    eps, gap, _ = curve[:, 0], curve[:, 1], curve[:, 2]
    ax.loglog(eps, gap, "o", ms=3.5, color=plot.ACCENT, label="实测 1-|cosθ|")
    ax.loglog(eps, 0.5 * eps**2, "--", lw=1.2, color=plot.ACCENT2, label=r"$\varepsilon^2/2$ 参考线")
    ax.set_xlabel(r"$\varepsilon$（两向量方向的偏离量）")
    ax.set_ylabel("1 - |cos θ|")
    ax.legend(frameon=False)
    sec.figure("equality-rate", fig,
               caption=f"等号附近是二阶收敛：斜率 {float(curve[0, 2]):.2f}，不是一阶")

    sec.observe(
        "不等式在浮点世界依然站得住（20 万样本无反例），但它给的是上界，不是典型值："
        "n=100 时随机两向量的 |cosθ| 几乎恒小于 0.3，"
        "这条正好解释了「高维里距离集中、余弦相似度失去区分度」。"
    )
    sec.pitfall(
        "工程实现一律先 np.clip(c, -1, 1) 再取 arccos：本例 5 万样本没能自然造出越界，"
        "但 nearest-after(1.0) 这个值就让 arccos 直接返回 nan —— 属于「极少发生但一发生就污染整条链路」的类型。"
    )
    sec.pitfall(
        "量纲差六个数量级时（本例 u~1e3、v~1e-3）误差会明显放大，"
        "做特征工程时先归一化再算相似度是硬规矩。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
