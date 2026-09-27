"""数值实验常用判据。每个函数都返回能被 `sec.check()` 直接用的东西。"""
from __future__ import annotations

import numpy as np


def rel_error(approx: np.ndarray, exact: np.ndarray, ord=None) -> float:
    """相对误差 ‖approx-exact‖ / ‖exact‖。"""
    denom = np.linalg.norm(np.asarray(exact), ord=ord)
    if denom == 0:
        return float(np.linalg.norm(np.asarray(approx), ord=ord))
    return float(np.linalg.norm(np.asarray(approx) - np.asarray(exact), ord=ord) / denom)


def max_excess(values: np.ndarray, bound: float = 0.0) -> float:
    """values 里【超出上界 bound】的最大正偏差。用于 "|u·v| ≤ ‖u‖‖v‖" 这类不等式。"""
    return float(np.max(np.asarray(values) - bound))


def convergence_rate(hs: np.ndarray, errs: np.ndarray) -> float:
    """最小二乘拟合斜率 p，使 err ≈ C·h^p。返回 p。"""
    hs = np.asarray(hs, float)
    errs = np.asarray(errs, float)
    mask = (hs > 0) & (errs > 0)
    p = np.polyfit(np.log(hs[mask]), np.log(errs[mask]), 1)
    return float(p[0])


def numerical_rank(a: np.ndarray, tol: float | None = None) -> int:
    """按奇异值阈值判秩。默认阈值取 numpy 的相对口径。"""
    s = np.linalg.svd(np.asarray(a, float), compute_uv=False)
    if tol is None:
        tol = max(a.shape) * np.finfo(float).eps * (s[0] if s.size else 0.0)
    return int(np.sum(s > tol))


def perturb_relative(a: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    """给 a 施加相对量级为 level 的随机扰动（保持形状）。"""
    a = np.asarray(a, float)
    return a + level * np.linalg.norm(a) * rng.standard_normal(a.shape) / np.sqrt(a.size)


def hilbert(n: int) -> np.ndarray:
    """Hilbert 矩阵 H_ij = 1/(i+j-1)，教科书里著名的病态例子。"""
    i = np.arange(1, n + 1)
    return 1.0 / (i[:, None] + i[None, :] - 1.0)
