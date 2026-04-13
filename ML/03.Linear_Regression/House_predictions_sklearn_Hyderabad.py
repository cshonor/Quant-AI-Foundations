# House predictions with linear regression (Hyderabad dataset)
# 对应笔记：01.例子、02.损失函数（MSE/RMSE）、03.梯度下降
# 说明：使用 scikit-learn，原书为 Turi Create。数据可本地 Hyderabad.csv 或从 URL 加载。

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# 加载数据：优先本地 Hyderabad.csv，否则从网络下载
# ---------------------------------------------------------------------------
def load_data():
    local_path = "Hyderabad.csv"
    url = "https://raw.githubusercontent.com/luisguiserrano/manning/master/Chapter_03_Linear_Regression/Hyderabad.csv"
    if os.path.isfile(local_path):
        return pd.read_csv(local_path)
    return pd.read_csv(url)


# ---------------------------------------------------------------------------
# 单特征：面积 vs 价格，拟合直线
# ---------------------------------------------------------------------------
def run_simple_model(data):
    X = data[["Area"]]
    y = data["Price"]
    simple_model = LinearRegression()
    simple_model.fit(X, y)
    print(f"y-intercept: {simple_model.intercept_}")
    print(f"slope (coefficient of Area): {simple_model.coef_[0]}")

    area_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    predicted_prices = simple_model.predict(area_range)
    plt.scatter(X, y, color="blue", label="Data Points")
    plt.plot(area_range, predicted_prices, color="red", linewidth=2, label="Regression Line")
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Area vs. Price with Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.show()
    return simple_model


# ---------------------------------------------------------------------------
# 多特征：标准化 + one-hot，全特征线性回归
# ---------------------------------------------------------------------------
def run_full_model(data):
    data_truncated = data[:2434]
    data_scaled = data_truncated.copy()

    area_mean = data_scaled["Area"].mean()
    area_std = data_scaled["Area"].std()
    data_scaled["Area"] = (data_scaled["Area"] - area_mean) / area_std

    bedrooms_mean = data_scaled["No. of Bedrooms"].mean()
    bedrooms_std = data_scaled["No. of Bedrooms"].std()
    data_scaled["No. of Bedrooms"] = (
        data_scaled["No. of Bedrooms"] - bedrooms_mean
    ) / bedrooms_std

    data_scaled_encoded = pd.get_dummies(
        data_scaled, columns=["Location"], prefix="Location", dtype=int
    )

    X_full = data_scaled_encoded.drop("Price", axis=1)
    y_full = data_scaled_encoded["Price"]

    model_predict_all = LinearRegression()
    model_predict_all.fit(X_full, y_full)

    print("\nLinear Regression Model Coefficients (Predicting Price from all features):")
    print(f"Intercept: {model_predict_all.intercept_}")
    print("Coefficients for features:")
    for feature, coef in zip(X_full.columns, model_predict_all.coef_):
        print(f"  {feature}: {coef}")

    y_pred = model_predict_all.predict(X_full)
    mse = mean_squared_error(y_full, y_pred)
    rmse_val = np.sqrt(mse)
    print(f"\nRoot Mean Squared Error (RMSE) of the model: {rmse_val}")

    # 新样本预测：面积 1000、3 卧室、Location_Gachibowli，需做相同标准化与 one-hot
    new_house_data = pd.DataFrame({"Area": [1000], "No. of Bedrooms": [3]})
    new_house_data["Area"] = (new_house_data["Area"] - area_mean) / area_std
    new_house_data["No. of Bedrooms"] = (
        new_house_data["No. of Bedrooms"] - bedrooms_mean
    ) / bedrooms_std

    location_cols = [col for col in X_full.columns if col.startswith("Location_")]
    new_house_location_dummies = pd.DataFrame(
        0, index=new_house_data.index, columns=location_cols
    )
    if "Location_Gachibowli" in location_cols:
        new_house_location_dummies["Location_Gachibowli"] = 1

    new_house_processed = pd.concat([new_house_data, new_house_location_dummies], axis=1)
    for col in X_full.columns:
        if col not in new_house_processed.columns:
            new_house_processed[col] = 0
    new_house_processed = new_house_processed[X_full.columns]

    predicted_price = model_predict_all.predict(new_house_processed)
    print(
        f"\nPredicted price for a house with size 1000 and 3 bedrooms (Gachibowli): {predicted_price[0]:,.2f}"
    )

    return model_predict_all, X_full, area_mean, area_std, bedrooms_mean, bedrooms_std


def main():
    data = load_data()
    print(data.head())
    num_rows, num_cols = data.shape
    print(f"The dataset has {num_rows} rows and {num_cols} columns")

    # 单特征：面积 vs 价格
    plt.scatter(data["Area"], data["Price"])
    plt.xlabel("Area")
    plt.ylabel("Price")
    plt.title("Area vs Price")
    plt.show()

    run_simple_model(data)

    # 多特征模型
    run_full_model(data)


if __name__ == "__main__":
    main()
