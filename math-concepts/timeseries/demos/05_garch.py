# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "arch", "scipy"]
# ///
"""
05 GARCH —— 不预测"值", 而是预测"波动有多大 (风险)"
==================================================

一句话直觉:
    前面 AR/MA/ARIMA 预测的是 y 本身 (均值)。
    GARCH 预测的是 y 的"波动率 σ_t" —— 也就是风险大小。

为什么金融必须有它? —— 两个铁律事实:
    1. 波动聚集 (volatility clustering): 大涨大跌扎堆出现, 平静期也扎堆。
       "今天暴动 → 明天大概率还暴动。"
    2. 收益率分布是"尖峰厚尾": 极端行情比正态分布预测的频繁得多。
    普通模型假设方差恒定 (同方差), 完全抓不住这两点 → 风控会严重低估尾部风险。

模型 (GARCH(1,1), 最常用):
    σ_t² = ω + α·ε_{t-1}² + β·σ_{t-1}²
    | 项          | 含义                              |
    | ----------- | ------------------------------- |
    | ω           | 长期平均方差水平 (基线)             |
    | α·ε_{t-1}²  | 昨天"震得猛不猛" (新消息冲击)        |
    | β·σ_{t-1}²  | 昨天的波动水平 (波动的惯性/记忆)      |
    持续性 = α+β, 越接近 1, 波动冲击衰减越慢 (金融里常 ~0.95+)。

类比:
    α = "今天被吓到的程度", β = "还没缓过来的余悸"。
    余悸 + 新惊吓 = 明天的紧张程度 (波动率)。

GARCH vs AR (别搞混):
    AR:    σ_t² = ω + α·ε_{t-1}²            ← 只看冲击, 没有"波动惯性"
    GARCH: 多了 β·σ_{t-1}² ← 加入波动自身的记忆, 用更少参数拟合长记忆波动。

这张图做什么:
    图 1  模拟收益率 → 肉眼可见"波动聚集"
    图 2  拟合出的条件波动率 σ_t 随时间变化
    图 3  收益率分布 vs 正态 → 看"厚尾"
"""

import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(99)


def simulate_garch(n: int, omega=0.05, alpha=0.1, beta=0.88) -> tuple[np.ndarray, np.ndarray]:
    """手写模拟 GARCH(1,1), 返回 (收益率 r, 真实波动率 σ)。"""
    r = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)  # 无条件方差作初值
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return r, np.sqrt(sigma2)


def main() -> None:
    n = 1500
    r, sigma_true = simulate_garch(n)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: 收益率, 看波动聚集 ----
    axes[0].plot(r, lw=0.5, color="tab:blue")
    axes[0].set_title("图1: 模拟收益率 —— 波动聚集 (大波动扎堆)")
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel("收益率 r_t")
    axes[0].grid(alpha=0.3)

    # ---- 图 2: 拟合 GARCH(1,1), 画条件波动率 ----
    # arch 库习惯用百分比尺度, 放大 100 倍数值更稳
    am = arch_model(r * 100, mean="Zero", vol="GARCH", p=1, q=1)
    res = am.fit(disp="off")
    cond_vol = res.conditional_volatility / 100  # 还原回原尺度
    ax = axes[1]
    ax.plot(sigma_true, label="真实 σ_t", lw=1.0, color="tab:green", alpha=0.7)
    ax.plot(cond_vol, label="GARCH 估计 σ_t", lw=0.9, color="tab:red", alpha=0.8)
    ax.set_title("图2: 条件波动率 σ_t —— GARCH 抓住了波动的起伏")
    ax.set_xlabel("时间 t")
    ax.set_ylabel("σ_t")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图 3: 厚尾 ----
    ax = axes[2]
    z = r / r.std()
    ax.hist(z, bins=80, density=True, alpha=0.6, label="收益率(标准化)")
    xs = np.linspace(-6, 6, 200)
    ax.plot(xs, stats.norm.pdf(xs), "r--", lw=1.5, label="正态分布")
    ax.set_title("图3: 厚尾 —— 极端行情比正态频繁 (尾部更高)")
    ax.set_xlabel("标准化收益率")
    ax.set_yscale("log")  # 对数坐标更易看尾部
    ax.legend()
    ax.grid(alpha=0.3)

    # 读取拟合参数
    p = res.params
    omega_hat, alpha_hat, beta_hat = p["omega"], p["alpha[1]"], p["beta[1]"]

    fig.suptitle("GARCH: 预测'波动率/风险' (波动聚集 + 厚尾), 风控与定价的核心", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"真实参数:  α=0.10, β=0.88, 持续性 α+β=0.98")
    print(f"估计参数:  α={alpha_hat:.3f}, β={beta_hat:.3f}, 持续性={alpha_hat+beta_hat:.3f}")
    print(f"超额峰度 (>0 即厚尾): {stats.kurtosis(r):.2f}  (正态=0)")
    print("一句话: GARCH = 长期基线 + 新冲击(α) + 波动惯性(β), 专门刻画风险随时间的起伏。")


if __name__ == "__main__":
    main()
