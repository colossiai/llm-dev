# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
手写 MA —— 为什么 MA 比 AR 难估? 自己算一遍就懂
================================================

02_ma.py 用 statsmodels 一行拟合了 MA。这里拆开看本质, 重点回答一个问题:

    为什么 AR 有闭式解, MA 却要"猜+优化"?

核心原因 (一句话):
    AR 的回归变量 y_{t-1} 是"看得见"的 → 线性回归闭式解。
    MA 的变量 ε_{t-1} 是"看不见"的随机冲击 → 没法直接回归, 只能反推。

以 MA(1): y_t = ε_t + θ·ε_{t-1} 为例, 两种手写解法 (只用 numpy):

【解法一】矩估计 (Method of Moments) —— 闭式但粗糙
    MA(1) 的理论一阶自相关: ρ₁ = θ / (1 + θ²)
    用样本 ρ̂₁ 反解这个二次方程: θ²·ρ̂₁ − θ + ρ̂₁ = 0
    → 取 |θ|<1 的那个根 (可逆条件)。一步到位, 但只用了 ρ₁ 一个信息, 不够准。

【解法二】条件最小二乘 CSS (Conditional Sum of Squares) —— 准, 要搜索
    关键技巧: 给定一个 θ, 假设 ε₀=0, 就能"递推还原"所有冲击:
        ε_t = y_t − θ·ε_{t-1}
    再算残差平方和 SSE(θ)=Σ ε_t²。让 SSE 最小的 θ 就是估计值。
    这就是 MLE 在高斯下的近似。这里用一维网格搜索把 SSE(θ) 曲线整条画出来, 看最小点。

类比:
    AR 像"解方程"(已知条件直接算); MA 像"试钥匙"(挨个 θ 试, 哪个让残差最小用哪个)。

这张图做什么:
    图 1  MA(1) 数据
    图 2  CSS 的 SSE(θ) 曲线 + 最小点, 对比矩估计与真值 ← 核心
    图 3  用估计的 θ 递推还原的冲击 ε vs 真实冲击
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def simulate_ma1(theta: float, n: int, sigma: float = 1.0):
    """模拟 MA(1): y_t = ε_t + θ·ε_{t-1}。返回 (y, 真实冲击 ε)。"""
    eps = rng.normal(0, sigma, size=n + 1)
    y = eps[1:] + theta * eps[:-1]
    return y, eps[1:]


def recover_eps(y: np.ndarray, theta: float) -> np.ndarray:
    """给定 θ, 假设 ε₀=0, 递推还原冲击: ε_t = y_t − θ·ε_{t-1}。"""
    eps = np.zeros(len(y))
    prev = 0.0
    for t in range(len(y)):
        prev = y[t] - theta * prev
        eps[t] = prev
    return eps


def css(y: np.ndarray, theta: float) -> float:
    """条件残差平方和 SSE(θ)。"""
    eps = recover_eps(y, theta)
    return float((eps ** 2).sum())


def fit_ma1_moments(y: np.ndarray) -> float:
    """矩估计: 用样本一阶自相关解二次方程 ρ₁θ²−θ+ρ₁=0, 取可逆根。"""
    yc = y - y.mean()
    rho1 = (yc[:-1] * yc[1:]).sum() / (yc ** 2).sum()
    if abs(rho1) >= 0.5:  # |ρ₁|<0.5 才有实根 (MA(1) 的理论上限)
        rho1 = np.sign(rho1) * 0.499
    disc = np.sqrt(1 - 4 * rho1 ** 2)
    roots = [(1 + disc) / (2 * rho1), (1 - disc) / (2 * rho1)]
    return min(roots, key=abs)  # 取 |θ|<1 的可逆根


def fit_ma1_css(y: np.ndarray, grid: np.ndarray):
    """CSS: 在网格上找让 SSE 最小的 θ。返回 (θ̂, SSE 曲线)。"""
    sse = np.array([css(y, th) for th in grid])
    return grid[np.argmin(sse)], sse


def main() -> None:
    true_theta = 0.7
    n = 1000
    y, eps_true = simulate_ma1(true_theta, n)

    grid = np.linspace(-0.99, 0.99, 400)
    theta_mm = fit_ma1_moments(y)
    theta_css, sse = fit_ma1_css(y, grid)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: 数据 ----
    axes[0].plot(y[:150], lw=1.0, color="tab:blue")
    axes[0].set_title(f"图1: MA(1) 数据 θ={true_theta}")
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel("y_t")
    axes[0].grid(alpha=0.3)

    # ---- 图 2: CSS 的 SSE(θ) 曲线 (核心) ----
    ax = axes[1]
    ax.plot(grid, sse, color="tab:blue", label="SSE(θ) 曲线")
    ax.axvline(theta_css, color="tab:red", ls="--", label=f"CSS 最小点 θ={theta_css:.3f}")
    ax.axvline(theta_mm, color="tab:orange", ls=":", label=f"矩估计 θ={theta_mm:.3f}")
    ax.axvline(true_theta, color="tab:green", ls="-", alpha=0.6, label=f"真值 θ={true_theta}")
    ax.set_title("图2: CSS 在网格上'试 θ', 谷底就是估计值")
    ax.set_xlabel("候选 θ")
    ax.set_ylabel("残差平方和 SSE")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图 3: 还原的冲击 vs 真实冲击 ----
    eps_hat = recover_eps(y, theta_css)
    ax = axes[2]
    ax.plot(eps_true[:120], label="真实冲击 ε", lw=1.0, color="tab:green", alpha=0.7)
    ax.plot(eps_hat[:120], label="递推还原 ε̂", lw=1.0, color="tab:red", ls="--", alpha=0.8)
    corr = np.corrcoef(eps_true, eps_hat)[0, 1]
    ax.set_title(f"图3: 用 θ̂ 还原的冲击 (相关={corr:.3f})")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("手写 MA: 冲击 ε 看不见 → 没闭式解, 靠'试 θ + 残差最小'(CSS) 反推", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"真值 θ      = {true_theta}")
    print(f"矩估计 θ    = {theta_mm:.4f}  (只用了 ρ₁, 较粗)")
    print(f"CSS 估计 θ  = {theta_css:.4f}  (用了全部残差, 更准)")
    print("一句话: AR 解方程, MA 试钥匙 —— 因为 MA 的回归变量 ε 看不见。")


if __name__ == "__main__":
    main()
