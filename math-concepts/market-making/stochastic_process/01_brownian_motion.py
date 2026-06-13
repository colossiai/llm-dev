# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
01 布朗运动 (Brownian Motion) 与 几何布朗运动 (GBM)
====================================================

画三张图, 把"价格为什么用布朗运动建模"看清楚:

  图 1  标准布朗运动 W_t :        多条样本路径
                                 → 看清"随机游走"的样子
  图 2  几何布朗运动 GBM :        S_t = S_0 · exp((μ - σ²/2)t + σ W_t)
                                 → mid-price 的标准建模, 保证恒正
  图 3  收益率分布检验 :          GBM 的 log-收益率 应该是正态
                                 → 这是 GBM 假设是否成立的检验方式

直觉:
  - 布朗运动 = "每一步都是独立同分布的微小高斯噪声叠加" 的极限
  - 关键性质:
        E[W_t]      = 0
        Var[W_t]    = t              ← 方差线性增长, 标准差按 √t 增长
        W_t - W_s  ~ N(0, t - s)     ← 增量是独立的正态
  - 为什么用 GBM 而不直接用 BM 建模 mid-price?
        BM 可能跑到负数, 价格不能为负
        GBM = exp(BM), 永远 > 0, 而且 log-收益率是正态 (经典金融假设)

做市里怎么用:
  - mid-price S_t 用 GBM 模拟 → 给定 σ, 你能算 "未来 Δt 内价格走多远"
  - 这直接决定了 spread 应该开多大 (波动越大, 价差越宽)
  - Avellaneda-Stoikov 的核心假设之一就是 mid-price 服从布朗运动
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)


def simulate_brownian(T: float, n_steps: int, n_paths: int) -> tuple[np.ndarray, np.ndarray]:
    """模拟标准布朗运动 W_t, 返回 (时间网格, 路径矩阵 shape=[n_paths, n_steps+1])。

    关键: 增量 ΔW ~ N(0, Δt), 然后累加。
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    dW = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))   # 关键: 标准差是 √dt 不是 dt
    W = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(dW, axis=1)], axis=1)
    return t, W


def simulate_gbm(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int):
    """模拟几何布朗运动 GBM。

    精确解:  S_t = S_0 · exp((μ - σ²/2) t + σ W_t)
    比起直接对 dS = μS dt + σS dW 数值离散, 这个解析公式没有离散误差。
    """
    t, W = simulate_brownian(T, n_steps, n_paths)
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S


def draw_brownian(ax):
    """图 1: 标准布朗运动多条路径 + 理论 ±σ 包络。"""
    T, n_steps, n_paths = 1.0, 500, 8
    t, W = simulate_brownian(T, n_steps, n_paths)

    for i in range(n_paths):
        ax.plot(t, W[i], lw=1.2, alpha=0.8)

    # 理论 ±√t 包络 (1 个标准差)
    ax.plot(t, np.sqrt(t), color="black", lw=1.5, linestyle="--", label=r"±1σ 包络: $\pm\sqrt{t}$")
    ax.plot(t, -np.sqrt(t), color="black", lw=1.5, linestyle="--")

    ax.axhline(0, color="gray", lw=0.6, linestyle=":")
    ax.set_xlabel("时间 t")
    ax.set_ylabel(r"$W_t$")
    ax.set_title("图 1: 标准布朗运动 8 条样本路径\n"
                 "方差随 t 线性增长 → 1σ 包络按 √t 张开")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def draw_gbm(ax):
    """图 2: 几何布朗运动 (mid-price 模型)。"""
    S0, mu, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps, n_paths = 500, 20

    t, S = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)

    for i in range(n_paths):
        ax.plot(t, S[i], lw=1.0, alpha=0.6)

    # 理论均值线 E[S_t] = S0 · e^(μt)
    ax.plot(t, S0 * np.exp(mu * t), color="black", lw=2.5, label=r"理论均值 $S_0 e^{\mu t}$")
    ax.axhline(S0, color="gray", lw=0.6, linestyle=":")

    ax.set_xlabel("时间 t (年)")
    ax.set_ylabel(r"价格 $S_t$")
    ax.set_title(f"图 2: GBM (mid-price 模型) — $S_0$={S0}, μ={mu}, σ={sigma}\n"
                 "永远恒正 → 适合建模价格")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def draw_returns_distribution(ax):
    """图 3: GBM 的 log-收益率 是否正态? (检验 GBM 假设)"""
    S0, mu, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps, n_paths = 1, 50_000  # 一步即可, 看 log(S_T / S_0)

    t, S = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)
    log_returns = np.log(S[:, -1] / S0)

    ax.hist(log_returns, bins=80, density=True, color="#1f77b4", alpha=0.6,
            edgecolor="white", label="模拟 log-收益率直方图")

    # 理论正态曲线 N((μ - σ²/2)T, σ²T)
    x = np.linspace(log_returns.min(), log_returns.max(), 400)
    mean_theo = (mu - 0.5 * sigma**2) * T
    std_theo = sigma * np.sqrt(T)
    pdf_theo = np.exp(-0.5 * ((x - mean_theo) / std_theo) ** 2) / (std_theo * np.sqrt(2 * np.pi))
    ax.plot(x, pdf_theo, color="#d62728", lw=2.5,
            label=fr"理论 $N({mean_theo:.3f}, {std_theo:.3f}^2)$")

    ax.set_xlabel(r"log-收益率 $\ln(S_T / S_0)$")
    ax.set_ylabel("密度")
    ax.set_title("图 3: GBM 的 log-收益率 服从正态分布\n"
                 "现实里这个假设常被违反 → 引出跳跃扩散 (见 04)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    draw_brownian(axes[0])
    draw_gbm(axes[1])
    draw_returns_distribution(axes[2])

    plt.tight_layout()
    savepath = "01_brownian_motion.png"
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {savepath}")


if __name__ == "__main__":
    main()
