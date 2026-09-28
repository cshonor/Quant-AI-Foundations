"""9.1 混杂变量与幸存者偏差：真正的问题在没被采集的那部分数据里

书上讲的东西（第 9 章前半段）：

  · 纽约邮报那个标题：「16:8 饮食法让心脏病死亡风险增加 91%」。研究是观察性的、
    数据靠患者自我报告，直到第八段才有一位斯坦福教授出来泼冷水；
  · **混杂变量**：变量 Z 同时导致 X 和 Y，但 Z 从没被采集进来，
    于是我们会把「孩子出现在厨房」当成「饼干消失」的原因，其实是狗；
  · 五种数据偏差：选择偏差、回忆偏差、锚定偏差、确认偏差、幸存者偏差；
  · **幸存者偏差**：只看活下来的样本。LinkedIn 上那篇「30 家最强上市公司的 10 个特质」
    没拿失败公司做对照；二战时瓦尔德指出该给返航飞机**没有弹孔**的地方加装甲。

这一节我把这四件事各做成了一个能出数的判决实验：

  ① 混杂变量：让 Z 同时驱动 X 与 Y，边际相关能到 0.8，但控制 Z 之后的偏相关掉到 0；
  ② 辛普森悖论（混杂的极端形态）：每组内部都是正相关，合起来看变成负相关；
  ③ 幸存者偏差：一个**所有人真实期望都严格为 0** 的策略池里，
     只看活下来的那些，照样能捧出一个看起来闪闪发光的「明星」；
  ④ 选择偏差：航空公司只调查自己的乘客，75% 会说最喜欢这家 —— 真实值只有 30%。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT, _ROOT / "data", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mathviz import Section, plot  # noqa: E402

SEC = Section(
    number="9.1",
    chapter="第 9 章 · 数据犯罪：Statistics Done Wrong",
    book="《精通统计学》(Grokking Statistics) · Thomas Nield",
    title="混杂变量与幸存者偏差：答案在没被采集的那部分数据里",
    claim="纽时/纽邮标题里「91% 风险上升」这类结论，往往漏了一个从未进过数据集的变量。"
          "混杂变量 Z 同时驱动 X 和 Y 却没被采集，我们会把孩子当成偷饼干的贼，"
          "其实是没进数据的狗。同一类问题还有五种认知/数据偏差，"
          "其中最致命的是幸存者偏差：只看活下来的样本 —— "
          "LinkedIn 上研究 30 家最强公司、不给失败公司做对照；"
          "二战时瓦尔德指出该给返航飞机没有弹孔的部位加装甲。",
    proposition="① 让 Z 同时驱动 X 与 Y：边际相关能到 0.8，控制 Z 后的偏相关回到 0（且不显著）；"
                "② 辛普森悖论：两组内部斜率都是正的，合并后的回归斜率变成负的；"
                "③ 纯噪声（真实夏普期望严格为 0）的策略池里，"
                "观测夏普的标准差是 1/√T，于是 10000 个运气策略中最亮的那颗能亮到夏普 2 以上；"
                "④ 只调查自己的乘客，回答「最喜欢这家航司」的比例会从真实的 30% 膨胀到 75%，"
                "数值与贝叶斯公式完全吻合。",
    source="第 9 章「混杂变量」「数据偏差」「选择偏差」「幸存者偏差」",
)

N_MC = 4000
SEED = 17


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


# ------------------------------------------------- ① 混杂变量：控制 Z 之后还剩什么
def experiment_confounder(sec: Section) -> dict:
    rng = np.random.default_rng(SEED)
    z = rng.normal(0, 1, N_MC)                       # 从未被采集的那个变量（= 邻居家的狗）
    x = 1.2 * z + rng.normal(0, 0.6, N_MC)           # 孩子在厨房出现
    y = 0.9 * z + rng.normal(0, 0.6, N_MC)           # 饼干消失

    r_xy, p_xy = _pearson(x, y)
    # 偏相关（一阶）：r_xy·z = (r_xy − r_xz·r_yz) / √((1−r_xz²)(1−r_yz²))
    r_xz, _ = _pearson(x, z)
    r_yz, _ = _pearson(y, z)
    r_partial = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    # 偏相关的 t 检验，自由度 n − 3（多扣一个被控制住的变量）
    t = r_partial * np.sqrt((N_MC - 3) / (1 - r_partial ** 2))
    p_partial = float(2 * stats.t.sf(abs(t), N_MC - 3))

    sec.check(
        f"混杂变量能让两个毫无因果关系的变量相关到 r = {r_xy:.4f}，"
        f"而一旦把 Z 控制住，偏相关掉到 {r_partial:.4f}（p = {p_partial:.3f}，不显著）",
        ok=abs(r_xy) > 0.7 and abs(r_partial) < 0.05 and p_partial > 0.05,
        detail=f"构造：Z ~ N(0,1)（从不进数据集），X = 1.2Z + N(0,0.6²)，Y = 0.9Z + N(0,0.6²)。"
               f"X 与 Y 之间没有任何箭头，它们的联系全靠 Z。"
               f"实测边际 r(X,Y) = {r_xy:.4f}（p = {p_xy:.2e}，强相关）；"
               f"一阶偏相关 r(X,Y | Z) = {r_partial:.4f}，自由度用 n−3 = {N_MC - 3}，"
               f"p = {p_partial:.3f}，完全不显著。"
               f"这就是书上「饼干不见了，孩子在场」那幅漫画的定量版本："
               f"**数据里没出现的东西，才是最难被怀疑的那个东西。**"
               f"实用提醒：偏相关只在 Z 被采集了之后才算得出来，"
               f"而书开头那项 16:8 禁食研究连患者的饮食内容都没采集。",
        tolerance="|边际 r| > 0.7 且 |偏 r| < 0.05 且 p > 0.05",
    )

    sec.check(
        f"这套数据里 X 与 Z、Y 与 Z 的相关本身都很高："
        f"r(X,Z) = {r_xz:.4f}、r(Y,Z) = {r_yz:.4f}，"
        f"三者的乘积恰好解释了边际相关本身",
        ok=abs(r_xz * r_yz - r_xy) / abs(r_xy) < 0.10,
        detail=f"当 X = aZ + ε₁、Y = bZ + ε₂ 时，边际相关近似等于 r(X,Z)·r(Y,Z)。"
               f"实测 r(X,Z)·r(Y,Z) = {r_xz * r_yz:.4f}，与边际 r = {r_xy:.4f} "
               f"相差 {abs(r_xz * r_yz - r_xy) / abs(r_xy):.1%}。"
               f"这条近似给了个快速体检法：**如果边际相关能被两段相关乘积解释掉，"
               f"那它就不是一个因果线索。**",
        tolerance="相对误差 < 10%",
    )
    return {"x": x, "y": y, "z": z, "r_xy": r_xy, "r_partial": r_partial,
            "p_partial": p_partial, "r_xz": r_xz, "r_yz": r_yz}


# ------------------------------------------------- ② 辛普森悖论
def experiment_simpson(sec: Section) -> dict:
    rng = np.random.default_rng(SEED + 1)
    n1 = n2 = 220
    x1 = rng.uniform(1.0, 5.0, n1)
    y1 = 2.0 * x1 + 18.0 + rng.normal(0, 1.2, n1)      # 组 1：人数少的低剂量组？
    x2 = rng.uniform(6.0, 10.0, n2)
    y2 = 2.0 * x2 + 5.0 + rng.normal(0, 1.2, n2)       # 组 2：截距被 Z 压低

    def slope(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.polyfit(a, b, 1)[0])

    s1, s2 = slope(x1, y1), slope(x2, y2)
    sx = np.concatenate([x1, x2])
    sy = np.concatenate([y1, y2])
    s_all = slope(sx, sy)
    r_all, p_all = _pearson(sx, sy)

    sec.check(
        f"辛普森悖论：两组内部斜率都是正的（{s1:.3f} 与 {s2:.3f}），"
        f"合并之后整体斜率反转成 {s_all:.3f}",
        ok=s1 > 0 and s2 > 0 and s_all < 0,
        detail=f"两个子组各自严格是 Y = 2X + c 的形式（组内斜率都约等于 +2），"
               f"但组 1 的 X 取值集中在 1~5 且截距高，组 2 的 X 集中在 6~10 且截距低。"
               f"合并后整体斜率 {s_all:.3f}、整体相关系数 {r_all:.4f}（p = {p_all:.2e}）。"
               f"这时如果有人只看汇总表做决策，得到的结论与看分组表完全相反。"
               f"**混杂变量在这里不是一个变量，而是『属于哪一组』这个标签本身** —— "
               f"它常被忽略，正因为它是二维表的一行而不是一列。",
        tolerance="两个子斜率 > 0 且合并斜率 < 0",
    )
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "s1": s1, "s2": s2,
            "s_all": s_all, "r_all": r_all}


# ------------------------------------------------- ③ 幸存者偏差：纯噪声里的明星
def experiment_survivorship(sec: Section) -> dict:
    rng = np.random.default_rng(SEED + 2)
    n_strat = 10_000
    n_days = 252 * 3                      # 3 年日频
    years = n_days / 252.0
    vol = 0.01                            # 日波动 1%，人人相同、真实期望严格为 0
    # 所有策略的日收益就是同方差零均值噪声：没有任何人有 alpha
    rets = rng.normal(0.0, vol, (n_strat, n_days))
    sharpe = np.sqrt(252) * rets.mean(axis=1) / rets.std(axis=1, ddof=1)

    # 观测夏普的理论标准差 = 1/√T（T 以年计）
    theo_std = 1.0 / np.sqrt(years)
    best = float(sharpe.max())
    survived = sharpe[sharpe > 1.0]       # 「活下来并被大家看见的」那些

    sec.check(
        f"一个所有人真实期望都严格为 0 的策略池里，最好的那颗观测夏普能到 {best:.2f}；"
        f"观测夏普的标准差 {sharpe.std(ddof=1):.4f} 与理论值 1/√{years:.1f} = {theo_std:.4f} 吻合",
        ok=best > 2.0 and abs(sharpe.std(ddof=1) - theo_std) < 0.02 and abs(sharpe.mean()) < 0.02,
        detail=f"{n_strat} 个策略 × {n_days} 天，日收益 N(0, {vol}²) —— 没有任何 alpha，"
               f"均值确实只有 {sharpe.mean():.4f}。"
               f"但**观测**夏普的标准差是 {sharpe.std(ddof=1):.4f}，理论 1/√{years:.0f} = {theo_std:.4f}。"
               f"（推导：年化夏普 ≈Mean/σ·√252，其中分子是 T 年均值，"
               f"标准误 σ/√T，归一化后标准差正好是 1/√T。）"
               f"于是 {n_strat} 个纯运气策略里，最亮的那颗冲到了 {best:.2f}，"
               f"有 {len(survived)} 个（{len(survived) / n_strat:.2%}）超过夏普 1.0。"
               f"**这不是运气好，这是必然 —— 只要池子够大，一定有人中彩票。**",
        tolerance="max > 2.0 且 |样本std − 1/√T| < 0.02 且均值 < 0.02",
    )

    # 幸存者视角：只看 Sharpe > 1 的那些人，他们的「平均实力」看起来是多少
    sec.check(
        f"只看幸存者：{len(survived)} 个观测夏普超过 1.0 的策略，它们的平均观测夏普是 "
        f"{survived.mean():.3f}，而整个池子的真实期望是 0 —— 差距全部来自筛选本身",
        ok=survived.mean() > 1.2 and abs(sharpe.mean()) < 0.02,
        detail=f"幸存者的平均观测夏普 {survived.mean():.3f}，中位数 {np.median(survived):.3f}，"
               f"最好的 {survived.max():.3f}。"
               f"而这一切的来源是一条没有任何 Alpha 的噪声。"
               f"这正是 LinkedIn 那篇「30 家最强公司的 10 个特质」的问题："
               f"它采集了成功者，却**结构上无法**采集失败者——"
               f"失败的公司没有上市镁光灯，甚至没有公开的财务数据。"
               f"书里点得很准：这是第 8 章讲的**类别不平衡**的极端版本，"
               f"正例（失败）一个都采不到。"
               f"CTA 上的直接后果：任何「还在运行的策略排行榜」都自带这个偏差，"
               f"因为清盘的、停更的、亏光的早就自己下架了。",
        tolerance="幸存者均值 > 1.2 且全池均值 < 0.02",
    )
    return {"sharpe": sharpe, "best": best, "theo_std": theo_std,
            "survived": survived, "years": years}


# ------------------------------------------------- ④ 选择偏差：航空公司那份问卷
def experiment_selection_bias(sec: Section) -> dict:
    # 真实的市场情况（这也是「上帝视角」的答案）
    p_like = 0.30            # 全体旅客中真正最喜欢 A 航的比例
    p_fly_given_like = 0.70  # 喜欢 A 的人里有 70% 真的买了 A 的票
    p_fly_given_not = 0.10   # 不喜欢 A 的人里也有 10% 因为时刻/票价买了 A
    # 贝叶斯：P(喜欢A | 乘坐A)
    p_fly = p_fly_given_like * p_like + p_fly_given_not * (1 - p_like)
    p_like_given_fly = p_fly_given_like * p_like / p_fly

    # 蒙特卡洛复核
    rng = np.random.default_rng(SEED + 3)
    n = 200_000
    likes = rng.random(n) < p_like
    p_fly_vec = np.where(likes, p_fly_given_like, p_fly_given_not)
    flies = rng.random(n) < p_fly_vec
    mc = float(likes[flies].mean())
    truth = float(likes.mean())

    sec.check(
        f"只调查自己的乘客：会有 {p_like_given_fly:.2%} 的人说最喜欢这家航司，"
        f"而全市场的真实值只有 {p_like:.2%} —— 蒙特卡洛复核得 {mc:.4f}",
        ok=abs(mc - p_like_given_fly) < 0.01 and p_like_given_fly > 2 * p_like,
        detail=f"贝叶斯算一遍：P(喜欢|乘坐) = 0.70×0.30 / (0.70×0.30 + 0.10×0.70) "
               f"= {p_fly_given_like * p_like:.3f} / {p_fly:.3f} = {p_like_given_fly:.4f}。"
               f"{n // 1000} 人的模拟给出 {mc:.4f}，两者相差 {abs(mc - p_like_given_fly):.2e}。"
               f"（作为对照，随机抽全体旅客得到的是 {truth:.4f}，这才接近真实的 30%。）"
               f"注意这不是说谎：**每位乘客都真诚地回答了问题，"
               f"是抽样框自己把结论做歪了。** 书里另一个例子同理 —— "
               f"工作日白天打民调电话，接到的多半是退休人员。",
        tolerance="蒙特卡洛与公式相差 < 0.01，且结果 > 2 倍真实值",
    )
    return {"p_like_given_fly": p_like_given_fly, "mc": mc, "truth": truth,
            "p_like": p_like}


# ------------------------------------------------- 图
def draw(sec: Section, conf: dict, simp: dict, surv: dict, sel: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.scatter(conf["x"], conf["y"], s=5, alpha=0.25, color=plot.ACCENT)
    b = np.polyfit(conf["x"], conf["y"], 1)
    xs = np.linspace(conf["x"].min(), conf["x"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), lw=2.0, color=plot.ACCENT2,
            label=f"边际：r = {conf['r_xy']:.3f}")
    ax.set_xlabel("X（孩子出现在厨房）"); ax.set_ylabel("Y（饼干消失）")
    ax.set_title("只看这两个变量，关系强得不能再强", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    z = conf["z"]
    q = np.digitize(z, np.quantile(z, [1 / 3, 2 / 3]))
    for g, color in enumerate([plot.ACCENT, plot.ACCENT2, plot.MUTED]):
        m = q == g
        ax.scatter(conf["x"][m], conf["y"][m], s=5, alpha=0.3, color=color,
                   label=f"控制 Z 的第 {g + 1} 层")
    bb = np.polyfit(conf["x"], conf["y"], 1)
    ax.plot(xs, np.polyval(bb, xs), lw=2.0, color="k", ls="--", alpha=0.7)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"把 Z 切成三层：偏相关 {conf['r_partial']:.3f}，关系消失",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("confounding", fig,
               caption="同一份数据，左图忽略 Z、右图按 Z 分层 —— 这就是有没有采到 Z 的差别")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    ax = axes[0]
    ax.scatter(simp["x1"], simp["y1"], s=7, alpha=0.5, color=plot.ACCENT,
               label=f"组 1 斜率 {simp['s1']:.2f}")
    ax.scatter(simp["x2"], simp["y2"], s=7, alpha=0.5, color=plot.ACCENT2,
               label=f"组 2 斜率 {simp['s2']:.2f}")
    xx = np.concatenate([simp["x1"], simp["x2"]])
    b_all = np.polyfit(xx, np.concatenate([simp["y1"], simp["y2"]]), 1)
    xs2 = np.linspace(xx.min(), xx.max(), 50)
    ax.plot(xs2, np.polyval(b_all, xs2), "k-", lw=2.2,
            label=f"合并后斜率 {simp['s_all']:.2f}")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("辛普森悖论：组内全正，整体为负", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(labelsize=8)

    ax = axes[1]
    sh = surv["sharpe"]
    ax.hist(sh, bins=80, color=plot.MUTED, alpha=0.55, label="全部策略（真实夏普期望 = 0）")
    ax.hist(surv["survived"], bins=40, color=plot.ACCENT, alpha=0.85,
            label=f"幸存者：观测夏普 > 1.0")
    ax.axvline(sh.mean(), ls=":", lw=1.2, color="k")
    ax.axvline(surv["best"], ls="--", lw=1.4, color=plot.ACCENT2,
               label=f"最好的一颗：{surv['best']:.2f}")
    ax.set_xlabel("观测夏普（3 年）"); ax.set_ylabel("策略数")
    ax.set_title("没有人有 alpha，但照样有明星", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    sec.figure("simpson-and-survivorship", fig,
               caption="左：合并与分组给出相反的斜率；右：纯噪声策略池的观测夏普分布")


def run(sec: Section) -> None:
    conf = experiment_confounder(sec)
    simp = experiment_simpson(sec)
    surv = experiment_survivorship(sec)
    sel = experiment_selection_bias(sec)
    draw(sec, conf, simp, surv, sel)

    sec.observe(
        f"观测夏普的标准差 = 1/√T 这条公式是可以直接用在自评上的："
        f"3 年数据给自己一个标准差 ±{1 / np.sqrt(3):.2f}，10 年才收到 ±{1 / np.sqrt(10):.2f}。"
        f"所以一个跑了 3 年、观测夏普 0.6 的策略，**根本区分不了它是真 alpha 还是运气** —— "
        f"零假设下一次抽样就有这么大的波动。"
        f"第 7 章那句「样本量决定尺子的刻度」在这里换个形式又出现了一次。"
    )
    sec.observe(
        "这一节四个实验的共同结构是同一句话：**数据集里『没有出现』的东西，"
        "往往比『出现了』的东西更有决定性。** 混杂变量没被采集、"
        "失败的公司被排除在样本之外、不常旅的人永远不会出现在航空公司问卷里。"
        "CTA 上对应的动作是：看任何一个策略排行榜之前，先问一句"
        "『下架的那批在哪里，它们长什么样』。"
    )
    sec.pitfall(
        "书里把 16:8 禁食那项研究说得挺狠，但要注意它的弱点有层次："
        "自我报告属于回忆偏差，缺少饮食营养数据属于混杂变量，"
        "而「观察性」这件事本身只否定了因果、没否定相关。"
        "这三个不是同一个指控，混着说会让自己也失去说服力。"
    )
    sec.pitfall(
        "别把「剔除离群值」当自动的 p-hacking（书 9.2 节会展开）。"
        "这里真正危险的是**筛选发生在拿到结果之后**："
        f"策略池那 {len(surv['survived'])} 个夏普 > 1 的策略之所以虚高，"
        "是因为先跑了 10000 个再看谁好看；如果一开始就声明『只上夏普 > 1 的策略』，"
        "那倒算得上是个诚实的规则。"
    )


if __name__ == "__main__":
    run(SEC)
    print(SEC.summary())
