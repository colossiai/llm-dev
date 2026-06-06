"""
04 - 导数 · 梯度 (PyTorch autograd 入门)

导数 (Derivative): 一元函数 f(x) 在某点的瞬时变化率, 即 f'(x)
梯度 (Gradient)  : 多元函数的偏导数组成的向量
                   ∇f = [∂f/∂x, ∂f/∂y, ...]
                   指向"函数值上升最快"的方向 (反方向 = 下降最快)

PyTorch 的 autograd 能自动求这些导数 —— 这正是 LLM 训练的核心:
    每一步沿着 -∇loss 的方向走一小步, loss 就在变小。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

# matplotlib 中文显示 (macOS)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    # =========================================================
    # 1. 单变量导数: f(x) = x², 求 f'(2)
    #    手算: f'(x) = 2x → f'(2) = 4
    # =========================================================
    x = torch.tensor(2.0, requires_grad=True)   # 告诉 PyTorch: 请追踪 x 的梯度
    y = x ** 2
    y.backward()                                 # 反向传播, 算出 dy/dx
    print("--- 单变量导数 ---")
    print(f"f(x) = x², 在 x=2 处: f'(x) = {x.grad.item()}  (手算: 4)")

    # =========================================================
    # 2. 多变量梯度: f(x, y) = x² + y², 求 ∇f(1, 2)
    #    手算: ∂f/∂x = 2x, ∂f/∂y = 2y → ∇f(1, 2) = [2, 4]
    # =========================================================
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    f = x ** 2 + y ** 2
    f.backward()
    print("\n--- 多变量梯度 ---")
    print(f"f = x² + y², ∇f(1, 2) = [{x.grad.item()}, {y.grad.item()}]  (手算: [2, 4])")

    # =========================================================
    # 3. 梯度下降: 找 f(x, y) = (x-3)² + (y+1)² 的最小值
    #    理论最小点: (3, -1), 最小值: 0
    #    LLM 训练的本质 = 这个循环, 只是参数从 2 个变成几十亿个。
    # =========================================================
    x = torch.tensor(0.0, requires_grad=True)
    y = torch.tensor(0.0, requires_grad=True)
    lr = 0.1                                     # 学习率 learning rate
    trajectory = [(x.item(), y.item())]

    for step in range(50):
        loss = (x - 3) ** 2 + (y + 1) ** 2
        loss.backward()
        with torch.no_grad():                    # 更新参数时不追踪计算图
            x -= lr * x.grad
            y -= lr * y.grad
        x.grad.zero_()                           # 梯度会累加, 用完必须清零
        y.grad.zero_()
        trajectory.append((x.item(), y.item()))

    print("\n--- 梯度下降 ---")
    print(f"50 步后 (x, y) = ({x.item():.4f}, {y.item():.4f})  (目标: (3, -1))")
    print("→ LLM 训练就是这个套路, 只是参数从 2 个变成了几十亿个")

    # =========================================================
    # 4. 可视化: 损失函数等高线 + 梯度下降轨迹
    # =========================================================
    xs = np.linspace(-1, 5, 100)
    ys = np.linspace(-4, 3, 100)
    X, Y = np.meshgrid(xs, ys)
    Z = (X - 3) ** 2 + (Y + 1) ** 2

    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(X, Y, Z, levels=20, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=7)

    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], "ro-", markersize=4, lw=1, label="梯度下降轨迹")
    ax.plot(3, -1, "g*", markersize=18, label="真实最小点 (3, -1)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("梯度下降: 从 (0, 0) 一步步走向最小点")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gradient_descent.png", dpi=120)
    print("\n图已保存到 gradient_descent.png")
    plt.show()


if __name__ == "__main__":
    main()
