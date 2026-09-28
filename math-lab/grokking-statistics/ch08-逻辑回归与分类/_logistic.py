"""8.x 共享工具：跳过 sklearn，用 numpy 手写一套逻辑回归 + 分类评估指标。

为什么不用 sklearn：本机的 mathlab 环境没装它。更重要的是——照着书把
LogisticRegression / train_test_split / roc_curve 全黑箱调一遍，学不到任何东西。
这一章的核心恰恰是「0/1 标签 + 概率输出」这套新范式，尤其是
**阈值才决定混淆矩阵**。所以全部涉笔 operated by hand，每行都能照抄到别的语言里。

提供：
  · NAP25 / NAP100        书里两份幼儿午睡数据集（25 个 / 100 个样本）
  · fit_logistic()        Newton–Raphson（IRLS）最大似然，等价于 penalty=None 的 sklearn
  · predict_proba()       sigmoid(β₀ + β₁x)
  · confusion() / prf()   混淆矩阵、精确率 / 召回率 / F1
  · roc_prc()             ROC 与 PRC 的阈值扫描
  · auc_trapezoid()       梯形法面积
  · auc_mannwhitney()     AUC 的秩和定义（用来和梯形法互相验算）
  · average_precision()   PRC 下的面积
  · holdout_split()       随机训练/测试划分
  · kfold_split()         K 折划分

所有函数纯 numpy，无隐藏状态。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 数据集
# 图 8.1 / 8.2 的小数据集：25 天的随机观察
NAP25_X = np.array([
    0.1, 0.5, 0.7, 1.2, 1.2, 1.5, 1.6, 1.9, 2.5, 2.5,
    2.7, 3.2, 3.3, 3.7, 3.8, 4.1, 4.5, 4.5, 4.5, 5.0,
    5.1, 5.6, 5.7, 6.2, 6.3,
])
NAP25_Y = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0,
])

# 图 8.7 起用到的大数据集：扩到 100 个样本（46 个 1 / 54 个 0）
NAP100_X = np.array([
    0.1, 0.13, 0.22, 0.31, 0.38, 0.38, 0.46, 0.5, 0.5, 0.56,
    0.64, 0.7, 0.7, 0.85, 0.96, 0.97, 1.06, 1.06, 1.15, 1.2,
    1.2, 1.22, 1.23, 1.24, 1.31, 1.33, 1.33, 1.41, 1.5, 1.6,
    1.7, 1.78, 1.84, 1.9, 1.9, 1.91, 1.98, 1.98, 2.03, 2.11,
    2.31, 2.37, 2.42, 2.5, 2.5, 2.5, 2.7, 2.77, 2.82, 2.92,
    3.17, 3.2, 3.28, 3.3, 3.32, 3.35, 3.46, 3.48, 3.7, 3.77,
    3.8, 3.8, 3.81, 3.82, 3.86, 3.89, 4.1, 4.2, 4.34, 4.48,
    4.49, 4.5, 4.5, 4.5, 4.63, 4.88, 4.9, 4.96, 5.0, 5.07,
    5.1, 5.11, 5.15, 5.23, 5.26, 5.47, 5.6, 5.64, 5.7, 5.73,
    5.81, 5.92, 5.98, 5.99, 6.08, 6.11, 6.11, 6.2, 6.21, 6.3,
])
NAP100_Y = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
    0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
    1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
], dtype=float)

# ---------------------------------------------------------------- 模型
def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """σ(z) = 1 / (1 + e^{-z})，用稳定写法避免大 |z| 溢出。"""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out if out.ndim else float(out)


def fit_logistic(x: np.ndarray, y: np.ndarray, *, tol: float = 1e-12,
                 max_iter: int = 200, ridge: float = 0.0):
    """一元逻辑回归的最大似然估计（Newton–Raphson / IRLS）。

    对数似然 ℓ(β) = Σ [yᵢ·log pᵢ + (1−yᵢ)·log(1−pᵢ)]，pᵢ = σ(β₀ + β₁xᵢ)。
    梯度 X'(y−p)，Hessian −X'WX（W = diag(p(1−p))），所以更新是
        β ← β + (X'WX)^{-1} X'(y − p)
    这就是加权最小二乘迭代（IRLS）。返回 (β₀, β₁, 迭代次数)。
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for it in range(1, max_iter + 1):
        p = sigmoid(X @ beta)
        w = np.clip(p * (1.0 - p), 1e-12, None)
        grad = X.T @ (y - p)
        H = (X * w[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            return float(beta[0]), float(beta[1]), it
    return float(beta[0]), float(beta[1]), max_iter


def predict_proba(x: np.ndarray, b0: float, b1: float) -> np.ndarray:
    """给定小时数，返回「会崩溃」的概率。"""
    return np.asarray(sigmoid(b0 + b1 * np.asarray(x, dtype=float)), dtype=float)


def log_odds(x: np.ndarray | float, b0: float, b1: float) -> np.ndarray:
    """logit(p) = ln(p/(1−p)) = β₀ + β₁x —— 8.1 要验证的就是这个恒等式。"""
    return b0 + b1 * np.asarray(x, dtype=float)


# ---------------------------------------------------------------- 评估
def confusion(y: np.ndarray, yhat: np.ndarray) -> tuple[int, int, int, int]:
    """返回 (TP, FP, TN, FN)：预测 1 且真是 1 / 预测 1 但真是 0 / …"""
    y = np.asarray(y).ravel()
    yhat = np.asarray(yhat).ravel()
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    return tp, fp, tn, fn


def prf(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    """精确率 / 召回率 / F1 / 准确率。TN 不进前两个指标——这是本章的关键。"""
    tp, fp, tn, fn = confusion(y, yhat)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }


def at_threshold(y: np.ndarray, proba: np.ndarray, thr: float) -> dict[str, float]:
    """把概率按阈值切成 0/1 之后算整套指标。"""
    return prf(y, (np.asarray(proba) >= thr).astype(int))


def roc_prc(y: np.ndarray, proba: np.ndarray):
    """扫全部阈值，返回 (fpr, tpr, precision, recall, thresholds)。

    thresholds 从大到小排列（严格递增的预测概率），与 sklearn 的 roc_curve 一致。
    ROC：(0,0) → … → (1,1)；PRC：按 recall 递增补齐尾端点 (0, 1)。
    """
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(proba, dtype=float).ravel()
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    # 以每个不同的预测概率为候选阈值，从大到小
    thrs = np.unique(s)[::-1]
    tpr, fpr, prec, rec = [0.0], [0.0], [], []
    for t in thrs:
        yhat = (s >= t).astype(int)
        tp = int(np.sum((yhat == 1) & (y == 1)))
        fp = int(np.sum((yhat == 1) & (y == 0)))
        tpr.append(tp / n_pos if n_pos else 0.0)
        fpr.append(fp / n_neg if n_neg else 0.0)
        prec.append(tp / (tp + fp) if tp + fp else 1.0)
        rec.append(tp / n_pos if n_pos else 0.0)
    # 补上永远预测负的端点
    thrs_ext = np.append(thrs, thrs[-1] - 1.0) if len(thrs) else np.array([0.0])
    tpr_arr = np.array(tpr + [1.0])
    fpr_arr = np.array(fpr + [1.0])
    prec_arr = np.array(prec + [1.0])
    rec_arr = np.array(rec + [0.0])
    return fpr_arr, tpr_arr, prec_arr, rec_arr, thrs_ext


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """曲线下面积（梯形法）。x 必须单调不减。"""
    return float(np.trapezoid(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def auc_mannwhitney(y: np.ndarray, proba: np.ndarray) -> float:
    """AUC 的等价定义：随机抽一个正例、一个负例，正例得分更高的概率。

    AUC = (Σ_pos rank − n_pos(n_pos+1)/2) / (n_pos · n_neg)，rank 用平均秩处理并列。
    这个定义与 prevalence 无关，正好解释「为什么 ROC-AUC 不怕类别不平衡」。
    """
    y = np.asarray(y).ravel().astype(int)
    s = np.asarray(proba, dtype=float).ravel()
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, proba: np.ndarray) -> float:
    """PRC 下面积。用 recall 做横轴，precision 按后一个 recall 取值加权（阶梯的正确算法）。"""
    fpr, tpr, prec, rec, _ = roc_prc(y, proba)
    order = np.argsort(rec)
    r = rec[order]
    p = prec[order]
    return float(np.sum(np.diff(r) * p[1:]))


# ---------------------------------------------------------------- 划分
def holdout_split(n: int, test_size: float, rng: np.random.Generator):
    """随机训练/测试划分。test 取 ceil(n·test_size)，与 sklearn 的口径一致。"""
    idx = rng.permutation(n)
    n_test = int(np.ceil(n * test_size))
    test = idx[:n_test]
    train = idx[n_test:]
    return train, test


def kfold_split(n: int, k: int, rng: np.random.Generator):
    """K 折：先打乱再切成 k 段，每段轮流作测试集。"""
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    for f in range(k):
        test = folds[f]
        train = np.concatenate([folds[i] for i in range(k) if i != f])
        yield train, test
