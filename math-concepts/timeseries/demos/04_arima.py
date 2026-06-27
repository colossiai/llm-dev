# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "statsmodels"]
# ///
"""
04 ARIMA (AR + I + MA) —— "先去趋势, 再做 ARMA"
==============================================

一句话直觉:
    很多数据 (股价、GDP、人口) 一直涨/跌, 是"不平稳"的, ARMA 直接用会失效。
    ARIMA 多了一个 I (Integration, 差分): 先把数据差分成平稳的, 再套 ARMA。

ARIMA(p, d, q) 三个数字:
    | 字母 | 名字       | 作用                         |
    | --- | --------- | --------------------------- |
    | p   | AR 阶      | 用过去值 (惯性)                |
    | d   | 差分次数    | 去掉趋势, 让数据平稳 ← 新增的    |
    | q   | MA 阶      | 用过去冲击 (余波)              |

差分是什么?
    Δy_t = y_t - y_{t-1}   (一阶差分 = 相邻两点之差)
    类比: 你看里程表 (一直增) 没法建模, 但看"每秒走了多远 (速度)"就平稳了。
    股价不平稳, 但"对数收益率 = Δlog(价格)"近似平稳 —— 这就是 d=1 在干的事。

怎么定 d?
    用 ADF 单位根检验: p 值 < 0.05 才算平稳。差到平稳为止 (通常 d=1 就够)。
    经验: d 不要过度差分 (over-differencing 会引入虚假的 MA 结构)。

金融场景:
    经典预测流程: 价格 →(取 log 差分)→ 平稳收益率 → ARIMA 拟合 → 预测 → 还原回价格。
    注意: 真实股价极接近随机游走 (ARIMA(0,1,0)), 短期方向几乎不可预测 ——
          这正是"为什么纯靠 ARIMA 炒股很难"的统计学解释。

这张图做什么:
    图 1  带趋势的"价格" + 一阶差分后变平稳 → 看懂 I 的作用
    图 2  ADF 检验前后 p 值对比
    图 3  ARIMA 拟合 + 向前预测 (带置信区间)
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(11)


def simulate_price(n: int) -> np.ndarray:
    """构造一个'带漂移 + ARMA 噪声'的不平稳价格序列 (像股价)。"""
    drift = 0.15
    shocks = rng.normal(0, 1.0, size=n)
    # 收益率本身是平稳 ARMA, 价格 = 漂移 + 累积 → 不平稳
    ret = drift + shocks + 0.4 * np.r_[0, shocks[:-1]]
    price = 100 + np.cumsum(ret)
    return price


def main() -> None:
    n = 400
    price = simulate_price(n)
    diff = np.diff(price)  # 一阶差分 → 收益率

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: 原序列(不平稳) vs 差分后(平稳) ----
    ax = axes[0]
    ax.plot(price, color="tab:blue", label="原'价格' (不平稳, 有趋势)")
    ax.set_ylabel("价格", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(np.r_[np.nan, diff], color="tab:orange", lw=0.8, alpha=0.7, label="一阶差分 (平稳)")
    ax2.set_ylabel("Δ价格", color="tab:orange")
    ax.set_title("图1: 差分 I —— 把'有趋势'变成'平稳'")
    ax.set_xlabel("时间 t")
    ax.grid(alpha=0.3)

    # ---- 图 2: ADF 检验前后对比 ----
    p_raw = adfuller(price)[1]
    p_diff = adfuller(diff)[1]
    ax = axes[1]
    bars = ax.bar(["原序列", "差分后"], [p_raw, p_diff], color=["tab:red", "tab:green"], alpha=0.8)
    ax.axhline(0.05, color="k", ls="--", label="平稳门槛 p=0.05")
    ax.set_title("图2: ADF 检验 p 值 (越低越平稳)")
    ax.set_ylabel("ADF p-value")
    for b, v in zip(bars, [p_raw, p_diff]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    ax.legend()

    # ---- 图 3: ARIMA 拟合 + 预测 ----
    train = price[:-30]
    res = ARIMA(train, order=(1, 1, 1)).fit()  # p=1, d=1, q=1
    fc = res.get_forecast(steps=30)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.2)  # 80% 区间
    idx_future = np.arange(len(train), len(train) + 30)
    ax = axes[2]
    ax.plot(np.arange(len(train)), train, label="训练数据", lw=1.0)
    ax.plot(np.arange(len(price) - 30, len(price)), price[-30:], label="真实未来", lw=1.2, color="tab:green")
    ax.plot(idx_future, mean, label="ARIMA(1,1,1) 预测", lw=1.5, color="tab:red")
    ax.fill_between(idx_future, ci[:, 0], ci[:, 1], color="tab:red", alpha=0.2, label="80% 区间")
    ax.set_title("图3: ARIMA 预测 (注意区间越来越宽 = 越远越不确定)")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle("ARIMA = 先差分(I)去趋势, 再用 ARMA 预测 (金融: 价格→收益率→预测)", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"ADF p值: 原序列={p_raw:.3f} (不平稳), 差分后={p_diff:.4f} (平稳)")
    print("一句话: I 用差分把'有趋势'拍平成平稳, 然后才轮到 ARMA 上场。")


if __name__ == "__main__":
    main()
