# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
02 泊松过程 (Poisson Process) 与 复合泊松 (Compound Poisson)
============================================================

画三张图, 把"订单为什么用泊松过程建模"看清楚:

  图 1  泊松过程 N_t :          阶梯状的事件计数, 多条路径
                               → 看清"离散事件随机到达"的样子
  图 2  到达间隔分布 :          相邻事件的时间间隔 → 指数分布
                               → 这是泊松过程的"特征签名"
  图 3  复合泊松 :              不仅算"几笔单", 还累加每笔的"成交量"
                               → 这才是真实的成交额过程

直觉:
  - 泊松过程回答的问题: "下一个事件什么时候来? 单位时间内来几个?"
  - 关键性质 (强度 λ):
        N(t)         ~ Poisson(λt)                          ← t 时刻已发生的事件数
        E[N(t)]      = λt,  Var[N(t)] = λt                   ← 均值 = 方差 (特征)
        间隔 τ        ~ Exponential(λ)                       ← 无记忆性
  - 无记忆性 (memoryless) 直觉: "已经等了 5 秒还没来, 不影响下一秒来的概率"
        现实订单流当然不完全无记忆 (订单会聚集) → 引出 Hawkes 过程 (见 03)

做市里怎么用:
  - 买单到达 / 卖单到达 / 成交事件 通常先用泊松建模 (一阶近似)
  - 关键参数: 到达率 λ(δ) 是"价差 δ"的函数
        你报得离 mid 越远 (δ 大), 成交概率越低
        Avellaneda-Stoikov 假设: λ(δ) = A · exp(-k δ)   (距离越远指数衰减)
  - 复合泊松: 累计成交额 = Σ V_i,  N(t) 笔成交各成交 V_i 单位
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)


def simulate_poisson_path(rate: float, T: float) -> np.ndarray:
    """模拟一条泊松过程的事件时间。

    方法: 间隔 τ_i ~ Exp(λ) 独立同分布, 累加得到事件时间。
    """
    arrivals = []
    t = 0.0
    while True:
        tau = rng.exponential(1.0 / rate)
        t += tau
        if t > T:
            break
        arrivals.append(t)
    return np.array(arrivals)


def draw_poisson_paths(ax):
    """图 1: 多条泊松路径的阶梯计数。"""
    rate, T = 5.0, 10.0
    n_paths = 6

    for i in range(n_paths):
        arrivals = simulate_poisson_path(rate, T)
        # 阶梯线: 在每个 arrival 处计数 +1
        t_plot = np.concatenate([[0], np.repeat(arrivals, 2), [T]])
        n_plot = np.concatenate([[0, 0], np.repeat(np.arange(1, len(arrivals) + 1), 2)])
        ax.plot(t_plot, n_plot, lw=1.5, alpha=0.8, label=f"路径 {i+1}")

    # 理论均值线
    t_grid = np.linspace(0, T, 200)
    ax.plot(t_grid, rate * t_grid, color="black", lw=2.5, linestyle="--",
            label=fr"理论均值 $E[N(t)] = \lambda t = {rate} t$")

    ax.set_xlabel("时间 t (秒)")
    ax.set_ylabel(r"事件计数 $N(t)$")
    ax.set_title(f"图 1: 泊松过程 6 条样本路径 (λ={rate})\n"
                 "阶梯跳变 → 每跳 +1 代表一个事件 (一笔订单到达)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_interarrival_distribution(ax):
    """图 2: 到达间隔分布 (应该是指数分布)。"""
    rate, T = 5.0, 1000.0  # 长时间窗口拿足够样本
    arrivals = simulate_poisson_path(rate, T)
    intervals = np.diff(arrivals)

    ax.hist(intervals, bins=60, density=True, color="#1f77b4", alpha=0.6,
            edgecolor="white", label="模拟间隔直方图")

    # 理论指数分布密度 λe^(-λτ)
    x = np.linspace(0, intervals.max(), 400)
    ax.plot(x, rate * np.exp(-rate * x), color="#d62728", lw=2.5,
            label=fr"理论 $\lambda e^{{-\lambda \tau}}$ (λ={rate})")

    ax.set_xlabel("到达间隔 τ (秒)")
    ax.set_ylabel("密度")
    ax.set_title("图 2: 到达间隔服从指数分布\n"
                 "无记忆性: 已经等 5 秒不会让下一秒到达概率变高")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


def draw_compound_poisson(ax):
    """图 3: 复合泊松 (订单数 × 订单量 = 累计成交额)。"""
    rate, T = 5.0, 10.0
    n_paths = 4

    for i in range(n_paths):
        arrivals = simulate_poisson_path(rate, T)
        # 每笔订单量: 假设服从对数正态 (模拟"大单偶发, 小单常见")
        volumes = rng.lognormal(mean=0.0, sigma=0.8, size=len(arrivals))
        cumulative = np.cumsum(volumes)

        t_plot = np.concatenate([[0], np.repeat(arrivals, 2), [T]])
        v_plot = np.concatenate([[0, 0], np.repeat(cumulative, 2)])
        ax.plot(t_plot, v_plot, lw=1.5, alpha=0.8, label=f"路径 {i+1}")

    ax.set_xlabel("时间 t (秒)")
    ax.set_ylabel(r"累计成交量 $\sum_{i=1}^{N(t)} V_i$")
    ax.set_title("图 3: 复合泊松过程\n"
                 "阶梯跳变高度 ≠ 1, 而是随机订单量 V_i (对数正态)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    draw_poisson_paths(axes[0])
    draw_interarrival_distribution(axes[1])
    draw_compound_poisson(axes[2])

    plt.tight_layout()
    savepath = "02_poisson_process.png"
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {savepath}")


if __name__ == "__main__":
    main()
