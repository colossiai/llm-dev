# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "statsmodels"]
# ///
"""
00 总览 —— 六个模型的"指纹"画在一张图里
========================================

一图看懂每个模型的招牌特征 (signature), 看完再去看 01~06 的细节。

  ┌─────────────┬─────────────┬─────────────┐
  │ AR: ACF拖尾  │ MA: ACF截断  │ ARMA: 两者拖尾│   ← 预测"值"
  ├─────────────┼─────────────┼─────────────┤
  │ ARIMA:差分平稳│ GARCH:波动聚集│ Kalman:噪声去噪│  ← 趋势/波动/状态
  └─────────────┴─────────────┴─────────────┘

口诀:
  AR 拖尾, MA 截断 (看 ACF 区分这对兄弟);
  ARIMA 先差分拍平, GARCH 盯波动, Kalman 在噪声里追真值。
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import acf

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
LAGS = 18
CI = 1.96 / np.sqrt(800)  # 近似 95% 置信带


def stem_acf(ax, y, title, note):
    a = acf(y, nlags=LAGS)
    x = np.arange(LAGS + 1)
    ax.bar(x, a, width=0.5, color="tab:blue", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(CI, color="red", ls="--", alpha=0.4)
    ax.axhline(-CI, color="red", ls="--", alpha=0.4)
    ax.set_xlabel("滞后 lag")
    ax.set_ylim(-0.4, 1.05)
    ax.set_title(title)
    ax.text(0.97, 0.92, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="tab:red", bbox=dict(boxstyle="round", fc="white", alpha=0.7))


def arma_sample(ar, ma, n):
    return ArmaProcess(np.r_[1, -np.array(ar)], np.r_[1, np.array(ma)]).generate_sample(
        nsample=n, distrvs=rng.standard_normal)


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ---- (0,0) AR: ACF 几何衰减拖尾 ----
    stem_acf(axes[0, 0], arma_sample([0.8], [], 800),
             "AR(1) φ=0.8 —— 预测'值'", "指纹: ACF 拖尾衰减")

    # ---- (0,1) MA: ACF 在 lag=q 后截断 ----
    ax = axes[0, 1]
    stem_acf(ax, arma_sample([], [0.7, 0.4], 800),
             "MA(2) —— 预测'值'", "指纹: ACF lag>2 截断")
    ax.axvline(2.5, color="green", ls=":", alpha=0.7)

    # ---- (0,2) ARMA: ACF 拖尾 (两成分都有) ----
    stem_acf(axes[0, 2], arma_sample([0.6], [0.5], 800),
             "ARMA(1,1) —— 惯性+冲击", "指纹: ACF/PACF 双拖尾→用AIC定阶")

    # ---- (1,0) ARIMA: 差分把不平稳变平稳 ----
    ax = axes[1, 0]
    shocks = rng.normal(0, 1, 400)
    price = 100 + np.cumsum(0.15 + shocks)  # 带漂移 → 不平稳
    ax.plot(price, color="tab:blue", label="原'价格'(不平稳)")
    ax.set_ylabel("价格", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(np.r_[np.nan, np.diff(price)], color="tab:orange", lw=0.7, alpha=0.6)
    ax2.set_ylabel("Δ价格(平稳)", color="tab:orange")
    ax.set_title("ARIMA —— 差分(I)去趋势")
    ax.set_xlabel("时间 t")
    ax.text(0.5, 0.08, "指纹: 差分后才平稳", transform=ax.transAxes, ha="center",
            fontsize=9, color="tab:red", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ---- (1,1) GARCH: 波动聚集 ----
    ax = axes[1, 1]
    n = 1200
    omega, alpha, beta = 0.05, 0.1, 0.88
    r = np.zeros(n)
    s2 = np.zeros(n)
    s2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        s2[t] = omega + alpha * r[t - 1] ** 2 + beta * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    ax.plot(r, lw=0.4, color="tab:blue", label="收益率")
    ax.plot(2 * np.sqrt(s2), color="tab:red", lw=1.0, label="±2σ_t 波动带")
    ax.plot(-2 * np.sqrt(s2), color="tab:red", lw=1.0)
    ax.set_title("GARCH —— 预测'波动率'")
    ax.set_xlabel("时间 t")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.5, 0.06, "指纹: 波动聚集(大波动扎堆)", transform=ax.transAxes, ha="center",
            fontsize=9, color="tab:red", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    # ---- (1,2) Kalman: 噪声去噪 ----
    ax = axes[1, 2]
    m = 200
    true = 100 + np.cumsum(rng.normal(0, 0.3, m))
    z = true + rng.normal(0, 2.0, m)
    xhat = np.zeros(m)
    x, p, Q, R = z[0], 1.0, 0.1, 4.0
    for t in range(m):
        p_pred = p + Q
        K = p_pred / (p_pred + R)
        x = x + K * (z[t] - x)
        p = (1 - K) * p_pred
        xhat[t] = x
    ax.plot(z, ".", ms=3, alpha=0.35, color="gray", label="噪声观测")
    ax.plot(true, color="tab:green", lw=1.6, label="真实状态")
    ax.plot(xhat, color="tab:red", lw=1.3, label="Kalman 估计")
    ax.set_title("Kalman —— 噪声里估'真值'")
    ax.set_xlabel("时间 t")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.5, 0.06, "指纹: 在线融合模型+观测", transform=ax.transAxes, ha="center",
            fontsize=9, color="tab:red", bbox=dict(boxstyle="round", fc="white", alpha=0.7))

    fig.suptitle("时间序列六模型总览: 上排预测'值'(看ACF指纹) | 下排管趋势/波动/状态",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print("口诀: AR拖尾, MA截断; ARIMA差分, GARCH盯波动, Kalman追真值。")


if __name__ == "__main__":
    main()
