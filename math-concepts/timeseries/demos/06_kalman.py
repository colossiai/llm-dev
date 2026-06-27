# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
06 Kalman Filter (卡尔曼滤波) —— 在噪声里估计"真实状态"
=====================================================

一句话直觉:
    你看不到真实值, 只能看到"带噪声的观测"。
    Kalman 把"模型的预测"和"新观测"按各自的可信度加权融合, 估出真实状态。

类比 (核心!):
    你在估自己体重。
      - 模型预测: "昨天 70kg, 我没乱吃, 今天估计还是 70kg" (但不完全确定)
      - 体重秤读数: "71.5kg" (秤也会抖, 不完全可信)
    最终答案 = 在 70 和 71.5 之间, 偏向"更可信的那个"。
    Kalman 增益 K 就是这个"偏向权重": 观测越准 → K 越大 → 越信秤。

两步循环 (每来一个新数据就走一遍):
    | 步骤        | 干什么                          | 直觉            |
    | ---------- | ------------------------------ | -------------- |
    | 预测 Predict | 用模型推下一状态, 不确定性变大     | "先猜, 心里没底"  |
    | 更新 Update  | 用新观测修正, 不确定性变小        | "看到数据, 更有谱" |

数学骨架 (一维随机游走 + 噪声观测):
    预测:  x̂⁻ = x̂        ;  P⁻ = P + Q          (Q=过程噪声, 模型有多不靠谱)
    更新:  K  = P⁻/(P⁻+R)                        (R=观测噪声, 秤有多不靠谱)
           x̂  = x̂⁻ + K·(z − x̂⁻)                 (z=新观测)
           P  = (1−K)·P⁻

GARCH/ARIMA 是"拟合一段历史"; Kalman 是"在线 (online) 逐点更新", 天生适合实时系统。

金融场景:
    - 从抖动的成交价里, 实时估计"真实 mid-price / 公允价值"。
    - 动态对冲比 (pairs trading 的 β 会漂移) → 用 Kalman 实时跟踪。
    - GPS / 机器人定位也是同一套数学。

这张图做什么:
    图 1  真实状态 vs 噪声观测 vs Kalman 估计 → 看"去噪"效果
    图 2  Kalman 增益 K 与不确定性 P 随时间收敛
    图 3  不同 Q/R 比值 → "更信模型 vs 更信数据"的对比
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(3)


def kalman_1d(z: np.ndarray, Q: float, R: float, x0: float, P0: float):
    """一维 Kalman 滤波 (状态模型 = 随机游走)。

    返回 (估计 x̂, 不确定性 P, 增益 K)。
    """
    n = len(z)
    xhat = np.zeros(n)
    P = np.zeros(n)
    K = np.zeros(n)
    x, p = x0, P0
    for t in range(n):
        # --- 预测 ---
        x_pred = x          # 随机游走: 下一状态 = 当前状态
        p_pred = p + Q
        # --- 更新 ---
        K[t] = p_pred / (p_pred + R)
        x = x_pred + K[t] * (z[t] - x_pred)
        p = (1 - K[t]) * p_pred
        xhat[t], P[t] = x, p
    return xhat, P, K


def main() -> None:
    n = 200
    # 真实状态: 缓慢漂移的"公允价值" (随机游走)
    true = 100 + np.cumsum(rng.normal(0, 0.3, n))
    R_true = 4.0  # 观测噪声方差 (秤/成交价的抖动)
    z = true + rng.normal(0, np.sqrt(R_true), n)  # 带噪声观测

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: 去噪效果 ----
    xhat, P, K = kalman_1d(z, Q=0.1, R=R_true, x0=z[0], P0=1.0)
    ax = axes[0]
    ax.plot(z, ".", ms=3, alpha=0.4, color="gray", label="噪声观测 (如成交价)")
    ax.plot(true, lw=1.6, color="tab:green", label="真实状态 (公允价值)")
    ax.plot(xhat, lw=1.4, color="tab:red", label="Kalman 估计")
    ax.set_title("图1: Kalman 去噪 —— 从抖动观测里恢复真实状态")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图 2: 增益 K 与不确定性 P 收敛 ----
    ax = axes[1]
    ax.plot(K, color="tab:purple", label="Kalman 增益 K (越大越信观测)")
    ax.plot(P, color="tab:orange", label="不确定性 P (越来越小=越有谱)")
    ax.set_title("图2: K 与 P 快速收敛到稳态")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 图 3: Q/R 比值 → 信模型 vs 信数据 ----
    ax = axes[2]
    ax.plot(true, lw=1.6, color="tab:green", label="真实状态", zorder=5)
    ax.plot(z, ".", ms=2, alpha=0.25, color="gray")
    for Q, label, color in [
        (0.001, "Q/R 小: 更信模型 (平滑但滞后)", "tab:blue"),
        (5.0, "Q/R 大: 更信数据 (灵敏但抖)", "tab:red"),
    ]:
        xh, _, _ = kalman_1d(z, Q=Q, R=R_true, x0=z[0], P0=1.0)
        ax.plot(xh, lw=1.2, color=color, label=label)
    ax.set_title("图3: 调 Q/R —— 信模型(平滑) vs 信数据(灵敏)")
    ax.set_xlabel("时间 t")
    ax.legend()
    ax.grid(alpha=0.3)

    rmse_obs = np.sqrt(np.mean((z - true) ** 2))
    rmse_kf = np.sqrt(np.mean((xhat - true) ** 2))

    fig.suptitle("Kalman 滤波: 预测+更新两步循环, 按不确定性融合'模型与观测'(在线实时)", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print(f"RMSE  原始观测={rmse_obs:.3f}  →  Kalman 估计={rmse_kf:.3f}  (降噪 {(1-rmse_kf/rmse_obs)*100:.0f}%)")
    print("一句话: 信模型 + 信数据, 按不确定性加权融合 → 在噪声里估出真实状态。")


if __name__ == "__main__":
    main()
