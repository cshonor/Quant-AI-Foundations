# Chapter 3: Linear Regression for a housing dataset
# 对应笔记：01.例子、02.损失函数、03.梯度下降
# 说明：使用 scikit-learn 实现，原书为 Turi Create。

from matplotlib import pyplot as plt
import numpy as np
import random

# ---------------------------------------------------------------------------
# 绘图：直线与散点
# ---------------------------------------------------------------------------
# 画直线：斜率 slope = m（每间房价格），y 轴截距 y_intercept = b（基础价格），公式 p̂ = mr + b
def draw_line(slope, y_intercept, color="grey", linewidth=0.7, starting=0, ending=8):
    x = np.linspace(starting, ending, 1000)
    plt.plot(x, y_intercept + slope * x, linestyle="-", color=color, linewidth=linewidth)


# 画散点：横轴房间数 r，纵轴价格 p
def plot_points(features, labels):
    X = np.array(features)
    y = np.array(labels)
    plt.scatter(X, y)
    plt.xlabel("number of rooms")
    plt.ylabel("prices")


# ---------------------------------------------------------------------------
# 数据：表 3.2（房间数 r，价格 p）
# ---------------------------------------------------------------------------
features = np.array([1, 2, 3, 5, 6, 7])  # r：房间数
labels = np.array([155, 197, 244, 356, 407, 448])  # p：真实价格


# ---------------------------------------------------------------------------
# 简单技巧 / 绝对技巧 / 平方技巧（对应 01.例子 伪代码）
# ---------------------------------------------------------------------------
def simple_trick(base_price, price_per_room, num_rooms, price):
    small_random_1 = random.random() * 0.1
    small_random_2 = random.random() * 0.1
    predicted_price = base_price + price_per_room * num_rooms
    if price > predicted_price and num_rooms > 0:
        price_per_room += small_random_1
        base_price += small_random_2
    if price > predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    if price < predicted_price and num_rooms > 0:
        price_per_room -= small_random_1
        base_price -= small_random_2
    if price < predicted_price and num_rooms < 0:
        price_per_room -= small_random_1
        base_price += small_random_2
    return price_per_room, base_price


def absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + price_per_room * num_rooms
    if price > predicted_price:
        price_per_room += learning_rate * num_rooms
        base_price += learning_rate
    else:
        price_per_room -= learning_rate * num_rooms
        base_price -= learning_rate
    return price_per_room, base_price


def square_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicted_price = base_price + price_per_room * num_rooms
    price_per_room += learning_rate * num_rooms * (price - predicted_price)
    base_price += learning_rate * (price - predicted_price)
    return price_per_room, base_price


# ---------------------------------------------------------------------------
# 线性回归算法：随机初始 → 重复「选点 + square_trick 更新」→ 返回 m, b
# ---------------------------------------------------------------------------
def linear_regression(features, labels, learning_rate=0.01, epochs=1000):
    random.seed(0)
    price_per_room = random.random()
    base_price = random.random()
    for epoch in range(epochs):
        i = random.randint(0, len(features) - 1)
        num_rooms = features[i]
        price = labels[i]
        price_per_room, base_price = square_trick(
            base_price, price_per_room, num_rooms, price, learning_rate=learning_rate
        )
    draw_line(price_per_room, base_price, "black", starting=0, ending=8)
    plot_points(features, labels)
    print("Price per room:", price_per_room)
    print("Base price:", base_price)
    return price_per_room, base_price


# ---------------------------------------------------------------------------
# RMSE（对应 02.损失函数）
# ---------------------------------------------------------------------------
def rmse(labels, predictions):
    n = len(labels)
    differences = np.subtract(labels, predictions)
    return np.sqrt(1.0 / n * (np.dot(differences, differences)))


# ---------------------------------------------------------------------------
# 用 sklearn 拟合同一数据
# ---------------------------------------------------------------------------
def main():
    from sklearn.linear_model import LinearRegression

    print("Features:", features)
    print("Labels:", labels)

    plt.ylim(0, 500)
    linear_regression(features, labels, learning_rate=0.01, epochs=1000)
    plt.show()

    features_reshaped = features.reshape(-1, 1)
    model = LinearRegression()
    model.fit(features_reshaped, labels)
    print("\nSklearn fit:")
    print("Coefficient:", model.coef_)
    print("Intercept:", model.intercept_)

    new_point = np.array([[4]])
    predicted_label = model.predict(new_point)
    print("Predicted label for feature 4:", predicted_label)


if __name__ == "__main__":
    main()
