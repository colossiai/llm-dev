# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "statsmodels"]
# ///
"""
01 AR (AutoRegressive, 自回归) —— "未来 = 过去值的加权和"
========================================================

一句话直觉:
    今天的值, 主要由"最近几天的值"决定 (惯性系统)。
    AR(p): y_t = c + φ_1·y_{t-1} + ... + φ_p·y_{t-p} + ε_t

类比:
    水温。现在水很热, 下一刻大概率还是热的 —— 它"记得"自己刚才的状态。
    φ 就是"记性有多好": φ 越接近 1, 惯性越强, 回到均值越慢。

金融场景:
    - 均值回复 (mean-reversion): 利差、配对交易的价差, 偏离后会慢慢被拉回。
      AR(1) 里 0<φ<1 正是"慢慢拉回均值"的数学形式。
    - 短期动量: φ>0 表示涨了还想涨 (但纯价格通常接近随机游走 φ≈1)。

关键性质 (以 AR(1): y_t = φ·y_{t-1} + ε_t 为例):
    | φ 的取值      | 行为                         |
    | ------------ | --------------------------- |
    | |φ| < 1      | 平稳, 围绕均值波动 (可建模)      |
    | φ = 1        | 随机游走 (不平稳, 就是股价)      |
    | |φ| > 1      | 爆炸发散 (现实里几乎不出现)      |
    长期均值 μ = c/(1-φ); 衰减"半衰期" ≈ ln(0.5)/ln(φ)

这张图做什么:
    图 1  不同 φ 的 AR(1) 路径    → 直观看"惯性强弱"
    图 2  ACF (自相关函数)        → AR 的指纹: 拖尾衰减 (geometric decay)
    图 3  用 statsmodels 拟合      → 从数据反推 φ, 看估计准不准
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)


def simulate_ar1(phi: float, n: int, sigma: float = 1.0, c: float = 0.0) -> np.ndarray:
    """模拟 AR(1): y_t = c + φ·y_{t-1} + ε_t, ε ~ N(0, σ²)。"""
    y = np.zeros(n)
    eps = rng.normal(0, sigma, size=n)
    for t in range(1, n):
        y[t] = c + phi * y[t - 1] + eps[t]
    return y


def main() -> None:
    n = 300
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: 不同 φ 的路径 ----
    ax = axes[0]
    for phi in (0.0, 0.5, 0.9, 0.99):
        ax.plot(simulate_ar1(phi, n), label=f"φ = {phi}", lw=1.1)
    ax.set_title("图1: AR(1) 路径 —— φ 越大惯性越强")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("y_t")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图 2: ACF, AR 的指纹是"拖尾衰减" ----
    y = simulate_ar1(0.8, 600)
    plot_acf(y, lags=30, ax=axes[1], title="图2: AR(1) φ=0.8 的 ACF (几何衰减拖尾)")
    axes[1].set_xlabel("滞后 lag")

    # ---- 图 3: 拟合, 从数据反推 φ ----
    true_phi = 0.7
    y = simulate_ar1(true_phi, 800)
    # ARIMA(p=1, d=0, q=0) 就是 AR(1)
    res = ARIMA(y, order=(1, 0, 0)).fit()
    est_phi = res.arparams[0]
    ax = axes[2]
    ax.plot(y[:150], label="数据", lw=1.0, alpha=0.7)
    fitted = res.fittedvalues[:150]
    ax.plot(fitted, label="AR(1) 拟合", lw=1.2)
    ax.set_title(f"图3: 拟合 φ → 真值={true_phi}, 估计={est_phi:.3f}")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("AR 自回归: 未来 = 过去值的加权和 (惯性 / 均值回复)", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"真实 φ = {true_phi}, 估计 φ = {est_phi:.4f}")
    print("一句话: AR 看'过去的值', ACF 拖尾衰减是它的指纹。")


if __name__ == "__main__":
    main()
