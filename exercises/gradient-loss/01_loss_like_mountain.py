"""
01 - 为什么说 loss 像一座山?

直觉:
  - 模型参数 (w, b, ...) 是地图上的坐标 (经度, 纬度)
  - loss(参数) 是该坐标点的 "海拔高度"
  - 所有 (参数 → loss) 的组合, 在空间里画出一个"地形"——就是 loss surface
  - 这个地形通常有山峰 (大 loss = 模型很烂)、山谷 (小 loss = 模型很好)
  - 训练 = 在这片山地里找最低的山谷

下面用一个最简单的线性回归 loss 把这座 "山" 画出来。
真实数据由 y = 2x + 1 生成, 我们去搜参数 w, b。
loss(w, b) = mean( (w·x + b - y)² )  ← 一个二元函数, 正好画成 3D 曲面。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # =========================================================
    # 1. 造数据: y = 2x + 1 + 一点噪声
    #    真实最优参数: w=2, b=1, 此处 loss 接近 0 (山谷底)
    # =========================================================
    torch.manual_seed(0)
    x_data = torch.linspace(-2, 2, 50)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn_like(x_data)

    def loss_fn(w, b):
        y_pred = w * x_data + b
        return ((y_pred - y_data) ** 2).mean()

    # =========================================================
    # 2. 在 (w, b) 平面上扫一遍, 算出每个点的 loss → 得到"地形"
    # =========================================================
    ws = torch.linspace(-2, 6, 80)
    bs = torch.linspace(-3, 5, 80)
    W, B = torch.meshgrid(ws, bs, indexing="ij")
    L = torch.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            L[i, j] = loss_fn(W[i, j], B[i, j])

    print("--- loss 地形抽样 ---")
    print(f"loss(w=2,  b=1 )  = {loss_fn(torch.tensor(2.0), torch.tensor(1.0)).item():.4f}  ← 山谷底")
    print(f"loss(w=0,  b=0 )  = {loss_fn(torch.tensor(0.0), torch.tensor(0.0)).item():.4f}  ← 山腰")
    print(f"loss(w=-2, b=-3)  = {loss_fn(torch.tensor(-2.0), torch.tensor(-3.0)).item():.4f}  ← 山顶")
    print("→ 改变 (w, b), loss 上下起伏, 这就是一座'山'")

    # =========================================================
    # 3. 画 3D 曲面 + 2D 等高线, 同一座山的两种视角
    # =========================================================
    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(W.numpy(), B.numpy(), L.numpy(), cmap="terrain", alpha=0.85)
    ax1.scatter([2], [1], [loss_fn(torch.tensor(2.0), torch.tensor(1.0)).item()],
                color="red", s=80, label="山谷底 (w=2, b=1)")
    ax1.set_xlabel("w")
    ax1.set_ylabel("b")
    ax1.set_zlabel("loss")
    ax1.set_title("3D 视角: loss 是一座山")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(W.numpy(), B.numpy(), L.numpy(), levels=25, cmap="terrain")
    ax2.clabel(cs, inline=True, fontsize=7)
    ax2.plot(2, 1, "r*", markersize=18, label="山谷底 (w=2, b=1)")
    ax2.set_xlabel("w")
    ax2.set_ylabel("b")
    ax2.set_title("俯视图 (等高线): 圈越密 = 坡越陡")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("01_loss_like_mountain.png", dpi=120)
    print("\n图已保存到 01_loss_like_mountain.png")
    plt.show()


if __name__ == "__main__":
    main()
