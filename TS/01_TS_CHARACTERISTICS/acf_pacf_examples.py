import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess


def simulate_ar(ar_params, n=600, burnin=200, seed=42):
    np.random.seed(seed)
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.array([1.0])
    process = ArmaProcess(ar, ma)
    return process.generate_sample(nsample=n, burnin=burnin)


def simulate_ma(ma_params, n=600, burnin=200, seed=42):
    np.random.seed(seed)
    ar = np.array([1.0])
    ma = np.r_[1, np.array(ma_params)]
    process = ArmaProcess(ar, ma)
    return process.generate_sample(nsample=n, burnin=burnin)


def simulate_arma(ar_params, ma_params, n=600, burnin=200, seed=42):
    np.random.seed(seed)
    ar = np.r_[1, -np.array(ar_params)]
    ma = np.r_[1, np.array(ma_params)]
    process = ArmaProcess(ar, ma)
    return process.generate_sample(nsample=n, burnin=burnin)


def save_acf_pacf_figure(series, title, output_path, lags=30):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=160)
    plot_acf(series, lags=lags, alpha=0.05, ax=axes[0])
    axes[0].set_title(f"{title} - ACF")
    plot_pacf(series, lags=lags, alpha=0.05, method="ywm", ax=axes[1])
    axes[1].set_title(f"{title} - PACF")
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    base = "C:/Users/12392/Desktop/py/ML/TS/01_TS_CHARACTERISTICS"

    # AR(2): ACF 拖尾, PACF 2 阶截尾（理论特征）
    ar2 = simulate_ar([0.75, -0.25], seed=1)
    save_acf_pacf_figure(
        ar2,
        "AR(2) Example",
        f"{base}/acf_pacf_ar2_example.png",
        lags=30,
    )

    # MA(2): ACF 2 阶截尾, PACF 拖尾（理论特征）
    ma2 = simulate_ma([0.65, 0.25], seed=2)
    save_acf_pacf_figure(
        ma2,
        "MA(2) Example",
        f"{base}/acf_pacf_ma2_example.png",
        lags=30,
    )

    # ARMA(1,1): ACF/PACF 都拖尾（理论特征）
    arma11 = simulate_arma([0.6], [0.4], seed=3)
    save_acf_pacf_figure(
        arma11,
        "ARMA(1,1) Example",
        f"{base}/acf_pacf_arma11_example.png",
        lags=30,
    )

    print("Saved: acf_pacf_ar2_example.png")
    print("Saved: acf_pacf_ma2_example.png")
    print("Saved: acf_pacf_arma11_example.png")


if __name__ == "__main__":
    main()
