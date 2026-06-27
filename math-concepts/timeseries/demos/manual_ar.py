# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
手写 AR —— 不调库, 自己把"参数怎么估出来"算一遍
================================================

01_ar.py 用 statsmodels 一行就拟合了。这里把那行拆开, 看清 AR 估计的本质:

一句话直觉:
    AR(p) 就是"把 y_t 对它自己的前 p 个值做线性回归"。
    所以 估 φ = 解一个线性方程组, 有闭式解, 根本不用迭代。

两种等价的手写解法 (都只用 numpy.linalg):
    | 方法           | 思路                              | 怎么算            |
    | ------------- | -------------------------------- | ---------------- |
    | OLS 最小二乘    | 把 y_t = φ·[y_{t-1..t-p}] 当回归   | 解正规方程 XᵀX φ=Xᵀy |
    | Yule-Walker   | 让"模型自相关"对上"样本自相关"        | 解 Toeplitz 方程 R φ=r |
    两者在大样本下几乎一样; OLS 更直观, YW 更能体现"AR 由自相关结构决定"。

预测 = 递推:
    估出 φ 后, ŷ_{t+1}=φ·过去值; 多步预测就把预测值当真值继续往前滚 (会向均值收敛)。

这张图做什么:
    图 1  数据 + 一步预测拟合
    图 2  真值 φ vs OLS vs Yule-Walker (三者对齐)
    图 3  多步向前预测 vs 真实未来
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)


def simulate_ar(phi: list[float], n: int, sigma: float = 1.0) -> np.ndarray:
    """模拟 AR(p): y_t = Σ φ_i·y_{t-i} + ε_t。"""
    p = len(phi)
    phi = np.asarray(phi)
    y = np.zeros(n)
    eps = rng.normal(0, sigma, size=n)
    for t in range(p, n):
        y[t] = phi @ y[t - p:t][::-1] + eps[t]
    return y


def fit_ar_ols(y: np.ndarray, p: int):
    """OLS 估计: 把 AR 当成线性回归, 解正规方程。返回 (常数项, φ)。"""
    n = len(y)
    # 设计矩阵: 每行 = [1, y_{t-1}, y_{t-2}, ..., y_{t-p}]
    X = np.column_stack([np.ones(n - p)] + [y[p - k - 1:n - k - 1] for k in range(p)])
    target = y[p:]
    # 闭式解 β = (XᵀX)⁻¹ Xᵀy  (用 solve 比直接求逆更稳)
    beta = np.linalg.solve(X.T @ X, X.T @ target)
    return beta[0], beta[1:]


def fit_ar_yule_walker(y: np.ndarray, p: int):
    """Yule-Walker 估计: 用样本自协方差解 Toeplitz 方程。返回 φ。"""
    y = y - y.mean()
    n = len(y)
    # 样本自协方差 γ_k
    gamma = np.array([(y[: n - k] * y[k:]).sum() / n for k in range(p + 1)])
    R = np.array([[gamma[abs(i - j)] for j in range(p)] for i in range(p)])  # Toeplitz
    r = gamma[1:p + 1]
    return np.linalg.solve(R, r)


def forecast_ar(y: np.ndarray, const: float, phi: np.ndarray, steps: int) -> np.ndarray:
    """递推多步预测: 把预测值当真值继续往前滚。"""
    p = len(phi)
    hist = list(y[-p:])
    out = []
    for _ in range(steps):
        nxt = const + phi @ np.array(hist[-p:][::-1])
        out.append(nxt)
        hist.append(nxt)
    return np.array(out)


def main() -> None:
    true_phi = [0.5, 0.3]  # AR(2)
    p = len(true_phi)
    y_full = simulate_ar(true_phi, 800)
    y, future = y_full[:-30], y_full[-30:]

    const, phi_ols = fit_ar_ols(y, p)
    phi_yw = fit_ar_yule_walker(y, p)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: 一步预测拟合 ----
    fitted = np.array([const + phi_ols @ y[t - p:t][::-1] for t in range(p, len(y))])
    axes[0].plot(y[p:p + 150], label="数据", lw=1.0, alpha=0.7)
    axes[0].plot(fitted[:150], label="一步预测 (OLS)", lw=1.2)
    axes[0].set_title("图1: 手写 AR(2) 一步预测拟合")
    axes[0].set_xlabel("时间 t")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ---- 图 2: 三种 φ 对比 ----
    ax = axes[1]
    x = np.arange(p)
    w = 0.25
    ax.bar(x - w, true_phi, w, label="真值", alpha=0.85)
    ax.bar(x, phi_ols, w, label="OLS", alpha=0.85)
    ax.bar(x + w, phi_yw, w, label="Yule-Walker", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"φ{i+1}" for i in range(p)])
    ax.set_title("图2: 真值 vs OLS vs Yule-Walker (几乎重合)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # ---- 图 3: 多步预测 ----
    fc = forecast_ar(y, const, phi_ols, steps=30)
    ax = axes[2]
    ax.plot(range(-40, 0), y[-40:], label="历史", lw=1.0)
    ax.plot(range(30), future, label="真实未来", lw=1.2, color="tab:green")
    ax.plot(range(30), fc, label="递推预测", lw=1.5, color="tab:red", ls="--")
    ax.axhline(y.mean(), color="gray", ls=":", alpha=0.6, label="均值")
    ax.set_title("图3: 多步预测向均值收敛")
    ax.set_xlabel("相对时间")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("手写 AR: 估 φ = 解线性方程组 (OLS / Yule-Walker), 无需迭代", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"真值 φ        = {true_phi}")
    print(f"OLS 估计 φ    = {np.round(phi_ols, 4).tolist()}  (常数项={const:.4f})")
    print(f"Yule-Walker φ = {np.round(phi_yw, 4).tolist()}")
    print("一句话: AR 是对自身滞后做线性回归, 有闭式解 → 解方程就出 φ。")


if __name__ == "__main__":
    main()
