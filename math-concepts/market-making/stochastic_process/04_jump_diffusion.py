# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "scipy"]
# ///
"""
04 跳跃扩散 (Jump-Diffusion / Merton 模型)
==========================================

画三张图, 把"为什么纯 GBM 不够, 必须加跳跃"看清楚:

  图 1  路径对比 :              纯 GBM vs Jump-Diffusion 同一随机种子
                               → 直接看清"跳跃"是什么
  图 2  收益率分布对比 :         GBM 是正态, JD 显著肥尾 (尾巴更厚)
                               → 这是真实金融数据的特征
  图 3  QQ 图 :                把模拟数据分位数 vs 正态分位数对比
                               → 偏离 45° 线 = 肥尾的可视化证据

直觉:
  - 纯 GBM 假设: log-收益率 ~ 正态  → 3σ 以上极端事件几乎不可能
  - 现实里: 闪崩 / 新闻冲击 / 流动性挤兑 → "黑天鹅"远比正态预测的频繁
  - 解决: Merton (1976) 加一个独立的"跳跃" Poisson 过程
        dS/S = μ dt + σ dW + (J - 1) dN
        - dW: 正常布朗扩散 (日常波动)
        - dN: 跳跃事件 (Poisson 强度 λ_J)
        - J : 每次跳的乘数 (常用 J = exp(Y), Y ~ N(m, v²))

做市里怎么用:
  - 估计 λ_J 让你知道"突发风险"的频率
  - 跳跃风险 → 你的库存可能在一瞬间被甩在错误一侧
  - 防御做市: 在新闻窗口前主动撤单 / 扩大 spread
  - 注意: 跳跃扩散是基础模型, 更现实的做法是把跳跃方向用 Hawkes (见 03) 替代 Poisson
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(123)


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, brownian=None):
    """普通 GBM。允许传入预先生成的布朗增量, 用于和 JD 共享同一份随机源。"""
    dt = T / n_steps
    if brownian is None:
        brownian = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))
    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * brownian, axis=1)
    return np.exp(log_S)


def simulate_jump_diffusion(S0, mu, sigma, T, n_steps, n_paths,
                             lam_jump, jump_mean, jump_std, brownian=None):
    """Merton jump-diffusion。

    每个 dt 内: 是否跳? 跳几次 ~ Poisson(λ_jump · dt)
    跳的幅度: J = exp(Y), Y ~ N(jump_mean, jump_std²)
    """
    dt = T / n_steps
    if brownian is None:
        brownian = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

    # 每个时间步的跳跃次数
    n_jumps = rng.poisson(lam_jump * dt, size=(n_paths, n_steps))
    # 总跳跃量 = 该步内所有跳跃的 log-J 之和
    # 一次跳的 log-J ~ N(jump_mean, jump_std²), n_jumps 次跳的和 ~ N(n*m, n*v²)
    jump_sum = np.where(
        n_jumps > 0,
        rng.normal(n_jumps * jump_mean, np.sqrt(n_jumps) * jump_std),
        0.0,
    )

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * brownian
    log_increments = drift + diffusion + jump_sum

    log_S = np.zeros((n_paths, n_steps + 1))
    log_S[:, 0] = np.log(S0)
    log_S[:, 1:] = np.log(S0) + np.cumsum(log_increments, axis=1)
    return np.exp(log_S), n_jumps


def draw_path_compare(ax):
    """图 1: 同一布朗增量下, GBM vs JD 的路径对比。"""
    S0, mu, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps = 500
    n_paths = 1

    # 关键: 两个模型共享同一份布朗源, 这样差异 100% 来自跳跃
    dt = T / n_steps
    shared_brownian = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

    S_gbm = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, brownian=shared_brownian)
    S_jd, n_jumps = simulate_jump_diffusion(
        S0, mu, sigma, T, n_steps, n_paths,
        lam_jump=8.0,       # 平均每年 8 次跳
        jump_mean=-0.02,    # 跳跃略偏负 (崩盘风险大于暴涨)
        jump_std=0.08,      # 跳跃方差
        brownian=shared_brownian,
    )

    t = np.linspace(0, T, n_steps + 1)
    ax.plot(t, S_gbm[0], color="#1f77b4", lw=1.5, label="纯 GBM")
    ax.plot(t, S_jd[0], color="#d62728", lw=1.5, label="Jump-Diffusion (Merton)")

    # 标记跳跃时刻
    jump_times = t[1:][n_jumps[0] > 0]
    for jt in jump_times:
        ax.axvline(jt, color="orange", lw=0.5, linestyle=":", alpha=0.7)
    ax.scatter(jump_times, S_jd[0][1:][n_jumps[0] > 0], color="orange",
               s=50, zorder=5, label=f"跳跃时刻 ({len(jump_times)} 次)")

    ax.set_xlabel("时间 t (年)")
    ax.set_ylabel("价格")
    ax.set_title(f"图 1: 同布朗源下 GBM vs JD\n"
                 "黄色竖线 = 跳跃时刻 → JD 在这些点突然偏离 GBM")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def draw_return_distributions(ax):
    """图 2: log-收益率分布对比, 看肥尾。"""
    S0, mu, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps = 252  # 每年 252 个交易日
    n_paths = 5000

    dt = T / n_steps
    brownian = rng.normal(0, np.sqrt(dt), size=(n_paths, n_steps))

    S_gbm = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, brownian=brownian)
    S_jd, _ = simulate_jump_diffusion(
        S0, mu, sigma, T, n_steps, n_paths,
        lam_jump=8.0, jump_mean=-0.02, jump_std=0.08, brownian=brownian,
    )

    # 日 log-收益率
    ret_gbm = np.diff(np.log(S_gbm), axis=1).flatten()
    ret_jd = np.diff(np.log(S_jd), axis=1).flatten()

    bins = np.linspace(-0.10, 0.10, 100)
    ax.hist(ret_gbm, bins=bins, density=True, alpha=0.5,
            color="#1f77b4", label=f"GBM (峰度={stats.kurtosis(ret_gbm):.2f})")
    ax.hist(ret_jd, bins=bins, density=True, alpha=0.5,
            color="#d62728", label=f"JD (峰度={stats.kurtosis(ret_jd):.2f})")

    ax.set_xlabel("日 log-收益率")
    ax.set_ylabel("密度")
    ax.set_title("图 2: 日收益率分布\n"
                 "JD 在尾部 (|return|>3%) 显著厚于 GBM → 肥尾")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.01)


def draw_qq_plot(ax):
    """图 3: QQ 图 — JD 收益率 vs 标准正态分位数, 偏离 45° 线 = 肥尾证据。"""
    S0, mu, sigma, T = 100.0, 0.05, 0.20, 1.0
    n_steps = 252
    n_paths = 2000

    S_jd, _ = simulate_jump_diffusion(
        S0, mu, sigma, T, n_steps, n_paths,
        lam_jump=8.0, jump_mean=-0.02, jump_std=0.08,
    )
    ret_jd = np.diff(np.log(S_jd), axis=1).flatten()

    # 标准化
    ret_std = (ret_jd - ret_jd.mean()) / ret_jd.std()

    # QQ 图
    stats.probplot(ret_std, dist="norm", plot=ax)

    ax.set_xlabel("理论正态分位数")
    ax.set_ylabel("样本分位数 (JD 收益率)")
    ax.set_title("图 3: QQ 图\n"
                 "两端明显偏离红线 (45°) → 极端事件比正态预测的多得多")
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    draw_path_compare(axes[0])
    draw_return_distributions(axes[1])
    draw_qq_plot(axes[2])

    plt.tight_layout()
    savepath = "04_jump_diffusion.png"
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {savepath}")


if __name__ == "__main__":
    main()
