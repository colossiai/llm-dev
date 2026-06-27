# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "statsmodels"]
# ///
"""
02 MA (Moving Average, 移动平均) —— "未来 = 过去'冲击'的余波"
============================================================

注意: 这里的 MA 不是"移动平均线 (技术指标)", 而是时间序列里的
      "对随机冲击 ε 做加权" 的模型, 名字撞了, 含义完全不同。

一句话直觉:
    今天的值 = 今天的随机冲击 + 过去几次冲击的"余波"。
    MA(q): y_t = μ + ε_t + θ_1·ε_{t-1} + ... + θ_q·ε_{t-q}

类比:
    往平静的湖面扔石头 (ε = 冲击/意外消息)。
    水花不会瞬间消失, 会荡漾几圈再平息 —— θ 就是"余波还剩多少"。

AR vs MA 的本质区别 (最关键的对比):
    | 模型 | 看的是什么        | 记忆长度              |
    | --- | ---------------- | -------------------- |
    | AR  | 过去的"值" y_{t-k} | 无限长 (指数衰减拖尾)   |
    | MA  | 过去的"冲击" ε_{t-k}| 有限长 (q 步后彻底归零) |

金融场景:
    - 一条突发新闻 (ε) 砸下来, 价格不是一步到位, 而是几天内被逐步消化 → MA 余波。
    - 微观结构噪声 (bid-ask bounce): 报价在买卖价间来回弹, 表现为 MA(1) 的负相关。

关键指纹:
    MA(q) 的 ACF 在 lag=q 之后 "突然截断 (cut off) 归零";
    这正好和 AR 的"拖尾衰减"相反 —— 这是辨认两者的核心手段。

这张图做什么:
    图 1  MA(1) θ 不同 → 冲击余波的样子
    图 2  ACF 在 lag=q 后截断 → MA 的指纹
    图 3  拟合 θ → 从数据反推余波系数
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def simulate_ma(thetas: list[float], n: int, sigma: float = 1.0, mu: float = 0.0) -> np.ndarray:
    """模拟 MA(q): y_t = μ + ε_t + Σ θ_j·ε_{t-j}。"""
    q = len(thetas)
    eps = rng.normal(0, sigma, size=n + q)
    y = np.zeros(n)
    for t in range(n):
        val = mu + eps[t + q]
        for j, th in enumerate(thetas, start=1):
            val += th * eps[t + q - j]
        y[t] = val
    return y


def main() -> None:
    n = 300
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: MA(1) 不同 θ ----
    ax = axes[0]
    for theta in (0.0, 0.6, -0.6, 0.95):
        ax.plot(simulate_ma([theta], n), label=f"θ = {theta}", lw=1.0, alpha=0.85)
    ax.set_title("图1: MA(1) 路径 —— θ 控制冲击余波")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("y_t")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图 2: ACF 在 lag=q 后截断 (这里 q=2) ----
    y = simulate_ma([0.7, 0.4], 800)
    plot_acf(y, lags=20, ax=axes[1], title="图2: MA(2) 的 ACF —— lag>2 后截断归零")
    axes[1].set_xlabel("滞后 lag")
    axes[1].axvline(2.5, color="red", ls="--", alpha=0.5)

    # ---- 图 3: 拟合 θ ----
    true_theta = [0.8, -0.3]
    y = simulate_ma(true_theta, 1000)
    res = ARIMA(y, order=(0, 0, 2)).fit()  # MA(2)
    est = res.maparams
    ax = axes[2]
    ax.plot(y[:150], label="数据", lw=1.0, alpha=0.7)
    ax.plot(res.fittedvalues[:150], label="MA(2) 拟合", lw=1.2)
    ax.set_title(f"图3: 拟合 θ → 真值={true_theta}, 估计={np.round(est,2).tolist()}")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("MA 移动平均: 未来 = 过去随机冲击 ε 的余波 (有限记忆, ACF 截断)", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"真实 θ = {true_theta}, 估计 θ = {np.round(est, 4).tolist()}")
    print("一句话: MA 看'过去的冲击', ACF 在 lag=q 后截断是它的指纹 (与 AR 拖尾相反)。")


if __name__ == "__main__":
    main()
