# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
03 Hawkes 过程 (自激发过程) — 现代做市必备
=============================================

画三张图, 把"为什么 Poisson 不够, 必须用 Hawkes"看清楚:

  图 1  Poisson vs Hawkes 事件流对比 :
                               → 直接看出 Hawkes 的"事件聚集 (clustering)"
  图 2  Hawkes 强度函数 λ(t) :
                               → 看清每次事件后 λ 跳升 + 指数衰减
  图 3  事件间隔分布对比 :
                               → Hawkes 的间隔分布是"重尾"(短间隔变多)

直觉:
  - Poisson 假设事件独立 → 强度 λ 是常数
  - Hawkes 假设事件自激发 (self-exciting) → 每发生一次, 强度跳升一段, 然后指数衰减:
        λ(t) = λ_0 + Σ_{t_i < t} α · exp(-β (t - t_i))
        - λ_0 :  基础强度 (没事件时的常规到达率)
        - α   :  跳升幅度 (一次事件让 λ 涨多少)
        - β   :  衰减速度 (跳升的影响多快回归)
  - 稳定性条件: α / β < 1 (否则强度爆炸)

为什么真实订单流是 Hawkes 而不是 Poisson?
  - 一笔大单成交 → 触发其他人跟风 / 算法平仓 → 短时间内更多成交
  - 高频实证: 订单间隔分布远比指数分布更"重尾", Poisson 完全拟合不上

做市里怎么用:
  - 用 Hawkes 估计"真实"到达率 → 在簇内瞬间扩大 spread (避免被一波吃光库存)
  - 区分"知情订单簇" vs "随机订单流" → 防逆向选择的关键

数值实现 (Ogata's thinning algorithm):
  - 在当前最大可能强度 λ_max 下生成候选事件
  - 用概率 λ(t)/λ_max 接受 (拒绝采样)
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)


def hawkes_intensity(t: float, history: np.ndarray, lam0: float, alpha: float, beta: float) -> float:
    """计算 t 时刻的 Hawkes 强度 λ(t), history 是 < t 的事件时间。"""
    if len(history) == 0:
        return lam0
    return lam0 + alpha * np.sum(np.exp(-beta * (t - history)))


def simulate_hawkes(lam0: float, alpha: float, beta: float, T: float) -> np.ndarray:
    """Ogata 拒绝采样模拟 Hawkes 过程。"""
    events = []
    t = 0.0
    while t < T:
        # 上界: 当前所有项都贡献最大值时的 λ
        lam_max = hawkes_intensity(t, np.array(events), lam0, alpha, beta) + 1e-9
        # 候选事件 (按上界泊松采样)
        t += rng.exponential(1.0 / lam_max)
        if t >= T:
            break
        # 拒绝采样
        u = rng.uniform()
        lam_t = hawkes_intensity(t, np.array(events), lam0, alpha, beta)
        if u <= lam_t / lam_max:
            events.append(t)
    return np.array(events)


def simulate_poisson_path(rate: float, T: float) -> np.ndarray:
    arrivals = []
    t = 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t > T:
            break
        arrivals.append(t)
    return np.array(arrivals)


def draw_compare_paths(ax):
    """图 1: 直观对比 Poisson 路径 vs Hawkes 路径 (相同平均强度)。"""
    T = 30.0
    lam0, alpha, beta = 0.5, 1.2, 1.5  # α/β = 0.8 < 1, 稳定
    hawkes_events = simulate_hawkes(lam0, alpha, beta, T)

    # 选一个等价平均强度的 Poisson 做公平对比
    # Hawkes 稳态均值: lam_bar = lam0 / (1 - alpha/beta)
    lam_bar = lam0 / (1 - alpha / beta)
    poisson_events = simulate_poisson_path(lam_bar, T)

    # 用 raster (eventplot) 画事件位置
    ax.eventplot([poisson_events, hawkes_events],
                 colors=["#1f77b4", "#d62728"],
                 lineoffsets=[1.0, 0.0],
                 linelengths=0.6)

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels([f"Hawkes\n({len(hawkes_events)} 个事件)",
                        f"Poisson\n({len(poisson_events)} 个事件)"])
    ax.set_xlabel("时间 t (秒)")
    ax.set_title(f"图 1: 相同平均强度下的事件流对比 (λ_bar={lam_bar:.2f})\n"
                 "Hawkes 明显成簇 (聚集), Poisson 均匀散布")
    ax.set_xlim(0, T)
    ax.grid(True, alpha=0.3, axis="x")


def draw_intensity(ax):
    """图 2: Hawkes 强度函数 λ(t) 的形状, 配合事件标记。"""
    T = 20.0
    lam0, alpha, beta = 0.3, 1.0, 1.2
    events = simulate_hawkes(lam0, alpha, beta, T)

    # 在密集网格上计算 λ(t)
    t_grid = np.linspace(0, T, 2000)
    lam_grid = np.array([hawkes_intensity(t, events[events < t], lam0, alpha, beta)
                         for t in t_grid])

    ax.plot(t_grid, lam_grid, color="#d62728", lw=1.5, label=r"Hawkes 强度 $\lambda(t)$")
    ax.axhline(lam0, color="#1f77b4", lw=1.5, linestyle="--",
               label=fr"基础强度 $\lambda_0$ = {lam0}")

    # 事件标记
    ax.scatter(events, np.full_like(events, -0.05), marker="|", s=200,
               color="black", label="事件发生时刻")

    ax.set_xlabel("时间 t (秒)")
    ax.set_ylabel(r"强度 $\lambda(t)$")
    ax.set_title(f"图 2: Hawkes 强度动态 (λ₀={lam0}, α={alpha}, β={beta})\n"
                 "每事件后 λ 跳升 α, 然后以速率 β 指数衰减回 λ₀")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.15)


def draw_interval_distributions(ax):
    """图 3: Hawkes vs Poisson 的事件间隔分布对比 (重尾性)。"""
    T = 5000.0  # 长窗口
    lam0, alpha, beta = 0.5, 1.2, 1.5
    hawkes_events = simulate_hawkes(lam0, alpha, beta, T)
    lam_bar = lam0 / (1 - alpha / beta)
    poisson_events = simulate_poisson_path(lam_bar, T)

    hawkes_intervals = np.diff(hawkes_events)
    poisson_intervals = np.diff(poisson_events)

    # 用对数纵轴看尾巴
    bins = np.linspace(0, 4, 80)
    ax.hist(poisson_intervals, bins=bins, density=True, alpha=0.5,
            color="#1f77b4", label="Poisson (理论指数)")
    ax.hist(hawkes_intervals, bins=bins, density=True, alpha=0.5,
            color="#d62728", label="Hawkes (短间隔显著更多)")

    # 理论指数曲线
    x = np.linspace(0, 4, 200)
    ax.plot(x, lam_bar * np.exp(-lam_bar * x), color="#1f77b4", lw=2,
            linestyle="--", label=fr"理论 $\lambda e^{{-\lambda \tau}}$, λ={lam_bar:.2f}")

    ax.set_xlabel("事件间隔 τ (秒)")
    ax.set_ylabel("密度")
    ax.set_title("图 3: 间隔分布对比\n"
                 "Hawkes 在 τ→0 处密度远高于 Poisson → 聚集的数学体现")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.001)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    draw_compare_paths(axes[0])
    draw_intensity(axes[1])
    draw_interval_distributions(axes[2])

    plt.tight_layout()
    savepath = "03_hawkes_process.png"
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {savepath}")


if __name__ == "__main__":
    main()
