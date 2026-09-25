# Chapter 4: Testing and regularization with scikit-learn
# 对应笔记：01.欠拟合与过拟合、02.测试、03.验证集、04.模型复杂度图、05.正则化
# 说明：原书使用 Turi Create，此处改用 scikit-learn 实现。

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(0)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# 绘图：多项式回归曲线与数据点（可选测试集）
# ---------------------------------------------------------------------------
def plot_polynomial_regression(model, X, Y, degree, X_test=None, Y_test=None):
    """
    绘制原始数据点和多项式回归曲线，可选叠加测试集点。

    Args:
        model: 训练好的 scikit-learn 模型。
        X: 输入特征（list 或 numpy array）。
        Y: 标签（list 或 numpy array）。
        degree: 多项式阶数。
        X_test, Y_test: 可选，测试集特征与标签。
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    X_plot = np.linspace(np.min(X), np.max(X), 100).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_plot_poly = poly.fit_transform(X_plot)
    Y_plot_poly = model.predict(X_plot_poly)

    plt.scatter(X, Y, color="blue", label="Original Data")
    plt.plot(X_plot, Y_plot_poly, color="red", label=f"Polynomial Regression (degree {degree})")

    if X_test is not None and Y_test is not None:
        X_test = np.array(X_test).reshape(-1, 1)
        Y_test = np.array(Y_test)
        plt.scatter(X_test, Y_test, color="orange", marker="^", label="Test Data")

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Polynomial Regression (Degree {degree})")
    plt.legend()
    plt.grid(True)
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    plt.show()


# ---------------------------------------------------------------------------
# 数据集：围绕 -x^2 + 2 的 40 个带噪声点
# ---------------------------------------------------------------------------
def polynomial(coefs, x):
    n = len(coefs)
    return sum([coefs[i] * x**i for i in range(n)])


def draw_polynomial(coefs):
    n = len(coefs)
    x = np.linspace(-1, 1, 1000)
    plt.plot(x, sum([coefs[i] * x**i for i in range(n)]), linestyle="-", color="black")


# 真实多项式：-x^2 + 2
coefs = [2, 0, -1]

X = []
Y = []
for i in range(40):
    x = random.uniform(-1, 1)
    y = polynomial(coefs, x) + random.gauss(0, 0.1)
    X.append(x)
    Y.append(y)


# ---------------------------------------------------------------------------
# 训练：多项式回归（可选 L1/L2 正则化）
# ---------------------------------------------------------------------------
def train_polynomial_regression(X, Y, degree, regularization=None, alpha=1.0):
    """
    训练多项式回归，可选 L1（Lasso）或 L2（Ridge）正则化。

    Args:
        X, Y: 特征与标签。
        degree: 多项式阶数。
        regularization: 'L1', 'L2' 或 None。
        alpha: 正则化强度（对应笔记中的 λ）。

    Returns:
        训练好的模型。
    """
    X = np.array(X).reshape(-1, 1)
    Y = np.array(Y)

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    if regularization == "L1":
        model = Lasso(alpha=alpha)
    elif regularization == "L2":
        model = Ridge(alpha=alpha)
    else:
        model = LinearRegression()

    model.fit(X_poly, Y)
    return model


def evaluate_model(model, X_test, Y_test, degree):
    """在测试集上计算 RMSE。"""
    X_test = np.array(X_test).reshape(-1, 1)
    Y_test = np.array(Y_test)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_test_poly = poly.fit_transform(X_test)
    y_pred = model.predict(X_test_poly)
    return np.sqrt(mean_squared_error(Y_test, y_pred))


# ---------------------------------------------------------------------------
# 主流程：无正则 / L1 / L2 对比
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    degree_used = 20

    # 划分训练集与测试集（对应 02.测试、03.验证集）
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    print("Shape of X_train:", np.shape(X_train))
    print("Shape of X_test:", np.shape(X_test))

    # 无正则化：高阶多项式容易过拟合（对应 01.欠拟合与过拟合）
    model_no_reg = train_polynomial_regression(X_train, Y_train, degree_used)
    plot_polynomial_regression(model_no_reg, X, Y, degree_used)
    rmse_no = evaluate_model(model_no_reg, X_test, Y_test, degree_used)
    print(f"Test RMSE (degree {degree_used}, no reg): {rmse_no}")

    # L1 正则化（Lasso）：对应 05.正则化，部分系数被压成 0
    l1_penalty = 0.01
    model_L1 = train_polynomial_regression(X_train, Y_train, degree_used, "L1", l1_penalty)
    plot_polynomial_regression(model_L1, X_train, Y_train, degree_used, X_test, Y_test)
    rmse_L1 = evaluate_model(model_L1, X_test, Y_test, degree_used)
    print(f"Test RMSE (degree {degree_used}, L1): {rmse_L1}")

    # L2 正则化（Ridge）：对应 05.正则化，系数缩小但很少为 0
    l2_penalty = 0.01
    model_L2 = train_polynomial_regression(X_train, Y_train, degree_used, "L2", l2_penalty)
    plot_polynomial_regression(model_L2, X_train, Y_train, degree_used, X_test, Y_test)
    rmse_L2 = evaluate_model(model_L2, X_test, Y_test, degree_used)
    print(f"Test RMSE (degree {degree_used}, L2): {rmse_L2}")

    # 系数对比：无正则化系数很大且杂乱；L1 产生稀疏；L2 整体缩小
    print("\nCoefficients (no reg):", model_no_reg.intercept_, model_no_reg.coef_[:5], "...")
    print("Coefficients (L1):    ", model_L1.intercept_, model_L1.coef_[:5], "...")
    print("Coefficients (L2):    ", model_L2.intercept_, model_L2.coef_[:5], "...")
