# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
05 Avellaneda-Stoikov 做市策略 — 把前面 4 个过程组装成完整 demo
================================================================

把前面四个文件学到的东西组合起来:
  - mid-price S_t :  GBM (见 01)
  - 订单到达 :       Poisson, 到达率随价差衰减 λ(δ) = A·e^(-k δ) (见 02)
  - 我的库存 q_t :   被随机成交流"推"着走
  - 报价决策 :       根据库存 q 动态调整 bid/ask → 这是 AS 模型的核心贡献

画四张图对比"朴素做市 vs AS 做市":
  图 1  价格 + 报价 :        看清两策略报价怎么挂
  图 2  库存对比 :           看 AS 怎么把库存拉回 0 (mean-reverting)
  图 3  单次 PnL 曲线 :      AS 不一定每次都赢, 但更稳
  图 4  多次蒙卡终端 PnL 分布: 关键! AS 终端 PnL 的方差更小 (低风险)

直觉:
  - 朴素策略: 始终对称报价 mid ± δ_fixed, 不考虑库存
        风险: 一旦库存堆积到正/负方向, 价格反向变动 → 大亏
  - AS 策略: 库存正 → 主动降 bid + 抬 ask, 引导市场把库存吃回去
        本质: 在"价差利润"和"库存风险"之间动态平衡

AS 公式 (核心三行, 简化常数风险厌恶):
        r(s, q, t)   =  s - q · γ · σ² · (T - t)          # 库存调整后的"参考价"
        δ_total      =  γ · σ² · (T - t) + (2/γ)·ln(1 + γ/k)   # 总半价差
        bid = r - δ_total/2,   ask = r + δ_total/2
  其中
        γ : 风险厌恶系数 (大 → 更怕库存)
        σ : mid-price 波动率
        k : 订单到达对价差的敏感度 λ(δ) = A·e^(-k δ)
        T : 时间窗口结束 (做市时段终点, 回到收盘库存 = 0)

参考: Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 市场 / 策略 参数
# ============================================================
S0     = 100.0   # 初始 mid-price
sigma  = 2.0     # mid-price 波动率 (绝对值)
T      = 1.0     # 做市时段长度
n_steps = 1000   # 时间步数
dt     = T / n_steps

# 订单到达模型: λ(δ) = A · exp(-k δ)
A      = 140.0   # 价差为 0 时的到达率上限
k      = 1.5     # 价差敏感度: 报远一点, 成交率掉得有多快

# 策略参数
gamma  = 0.1     # AS 的风险厌恶系数
fixed_delta = 1.0  # 朴素策略的固定半价差


def simulate_mid_price(seed: int) -> np.ndarray:
    """用算术布朗运动模拟 mid-price (AS 原文也是用算术 BM, 不是 GBM)。"""
    rng = np.random.default_rng(seed)
    dW = rng.normal(0, np.sqrt(dt), n_steps)
    S = np.empty(n_steps + 1)
    S[0] = S0
    S[1:] = S0 + np.cumsum(sigma * dW)
    return S


def step_orders(delta_bid: float, delta_ask: float, rng) -> tuple[bool, bool]:
    """单步内是否成交。

    成交 = 这一步内出现"对手单到达"事件 (Poisson)。
    概率 ≈ λ(δ) · dt = A · exp(-k δ) · dt
    """
    p_buy_filled  = A * np.exp(-k * delta_bid)  * dt   # 我的买单成交 (有人卖给我)
    p_sell_filled = A * np.exp(-k * delta_ask)  * dt   # 我的卖单成交 (有人买走)
    buy_filled  = rng.uniform() < p_buy_filled
    sell_filled = rng.uniform() < p_sell_filled
    return buy_filled, sell_filled


def run_strategy(S: np.ndarray, strategy: str, seed: int) -> dict:
    """跑一遍策略, 返回 (库存, 现金, 报价时间序列)。"""
    rng = np.random.default_rng(seed)
    q = 0          # 库存 (持仓数量)
    cash = 0.0     # 累计现金

    q_path = np.zeros(n_steps + 1)
    bid_path = np.zeros(n_steps + 1)
    ask_path = np.zeros(n_steps + 1)
    pnl_path = np.zeros(n_steps + 1)

    for i in range(n_steps):
        t = i * dt
        time_left = T - t
        s = S[i]

        if strategy == "naive":
            # 朴素: 对称固定价差, 不管库存
            r = s
            delta_b = delta_a = fixed_delta
        elif strategy == "AS":
            # Avellaneda-Stoikov 闭式解
            r = s - q * gamma * sigma**2 * time_left                    # 参考价
            total_spread = gamma * sigma**2 * time_left + (2 / gamma) * np.log(1 + gamma / k)
            half = total_spread / 2
            # bid/ask 围绕 r 对称, 而 r 已经因库存偏移
            bid = r - half
            ask = r + half
            delta_b = s - bid       # 我的 bid 相对 mid 的距离
            delta_a = ask - s       # 我的 ask 相对 mid 的距离
            # 兜底: 不允许负价差 (穿越 mid)
            delta_b = max(delta_b, 0.01)
            delta_a = max(delta_a, 0.01)
        else:
            raise ValueError(strategy)

        bid_price = s - delta_b
        ask_price = s + delta_a
        bid_path[i] = bid_price
        ask_path[i] = ask_price

        buy_filled, sell_filled = step_orders(delta_b, delta_a, rng)
        if buy_filled:
            q    += 1
            cash -= bid_price   # 花钱买入
        if sell_filled:
            q    -= 1
            cash += ask_price   # 卖出收钱

        # 标记到市值的 PnL = 现金 + 库存 × mid
        pnl_path[i] = cash + q * s
        q_path[i] = q

    # 收盘: 强制 q=0 平仓 (按最终 mid 平)
    cash += q * S[-1]
    pnl_path[-1] = cash
    q_path[-1] = 0
    bid_path[-1] = bid_path[-2]
    ask_path[-1] = ask_path[-2]

    return dict(q=q_path, pnl=pnl_path, bid=bid_path, ask=ask_path, final_pnl=cash)


def draw_quotes_and_price(ax, S, naive, AS):
    """图 1: mid-price + 两策略的 bid/ask (只画前 200 步避免太密)。"""
    n_show = 200
    t = np.arange(n_show) * dt

    ax.plot(t, S[:n_show], color="black", lw=1.5, label="mid-price S_t", zorder=3)
    ax.plot(t, naive["bid"][:n_show], color="#1f77b4", lw=0.8, alpha=0.7, label="朴素 bid")
    ax.plot(t, naive["ask"][:n_show], color="#1f77b4", lw=0.8, alpha=0.7, label="朴素 ask")
    ax.plot(t, AS["bid"][:n_show], color="#d62728", lw=0.8, alpha=0.7, label="AS bid")
    ax.plot(t, AS["ask"][:n_show], color="#d62728", lw=0.8, alpha=0.7, label="AS ask")

    ax.set_xlabel("时间 t")
    ax.set_ylabel("价格")
    ax.set_title("图 1: 前 200 步报价对比\n"
                 "蓝 (朴素) 始终对称, 红 (AS) 会随库存偏移")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_inventory(ax, naive, AS):
    """图 2: 库存路径对比 (AS 应该把库存压回 0 附近)。"""
    t = np.arange(n_steps + 1) * dt
    ax.plot(t, naive["q"], color="#1f77b4", lw=1.2, label="朴素策略")
    ax.plot(t, AS["q"], color="#d62728", lw=1.2, label="AS 策略")
    ax.axhline(0, color="gray", lw=0.6, linestyle=":")

    ax.set_xlabel("时间 t")
    ax.set_ylabel("库存 q")
    ax.set_title("图 2: 库存路径\n"
                 "AS 主动把库存推回 0, 朴素策略放任漂移")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def draw_pnl_path(ax, naive, AS):
    """图 3: 单次 PnL 时间序列。"""
    t = np.arange(n_steps + 1) * dt
    ax.plot(t, naive["pnl"], color="#1f77b4", lw=1.2,
            label=f"朴素 (终端={naive['final_pnl']:.1f})")
    ax.plot(t, AS["pnl"], color="#d62728", lw=1.2,
            label=f"AS (终端={AS['final_pnl']:.1f})")
    ax.axhline(0, color="gray", lw=0.6, linestyle=":")

    ax.set_xlabel("时间 t")
    ax.set_ylabel("Mark-to-Market PnL")
    ax.set_title("图 3: 单次回测 PnL 曲线\n"
                 "单次结果不一定 AS 赢, 看图 4 的分布才公平")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)


def draw_monte_carlo(ax, n_runs: int = 500):
    """图 4: 多次蒙特卡洛, 比较两策略终端 PnL 的分布 (均值 vs 方差)。"""
    naive_finals = []
    AS_finals    = []
    for seed in range(n_runs):
        S = simulate_mid_price(seed=seed)
        naive_finals.append(run_strategy(S, "naive", seed=seed + 10_000)["final_pnl"])
        AS_finals.append(run_strategy(S, "AS",      seed=seed + 10_000)["final_pnl"])

    naive_finals = np.array(naive_finals)
    AS_finals = np.array(AS_finals)

    bins = np.linspace(min(naive_finals.min(), AS_finals.min()),
                       max(naive_finals.max(), AS_finals.max()), 50)
    ax.hist(naive_finals, bins=bins, alpha=0.5, color="#1f77b4",
            label=f"朴素: mean={naive_finals.mean():.1f}, std={naive_finals.std():.1f}")
    ax.hist(AS_finals, bins=bins, alpha=0.5, color="#d62728",
            label=f"AS:   mean={AS_finals.mean():.1f}, std={AS_finals.std():.1f}")
    ax.axvline(0, color="black", lw=0.8, linestyle=":")

    ax.set_xlabel("终端 PnL")
    ax.set_ylabel("频次")
    ax.set_title(f"图 4: {n_runs} 次蒙特卡洛终端 PnL 分布\n"
                 "重点不是均值 — 看 AS 的标准差小很多 (风险更低)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    # 跑一条样本路径用于图 1-3
    S = simulate_mid_price(seed=42)
    naive = run_strategy(S, "naive", seed=999)
    AS = run_strategy(S, "AS", seed=999)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    draw_quotes_and_price(axes[0, 0], S, naive, AS)
    draw_inventory(axes[0, 1], naive, AS)
    draw_pnl_path(axes[1, 0], naive, AS)
    draw_monte_carlo(axes[1, 1], n_runs=500)

    plt.tight_layout()
    savepath = "05_market_making_demo.png"
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {savepath}")


if __name__ == "__main__":
    main()
