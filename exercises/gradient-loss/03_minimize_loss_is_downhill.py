"""
03 - 为什么求最小 loss 就像下山?

上一节我们看到: ∇loss 永远指向"上山最陡"。
那"下山最陡"是哪? —— 就是它的反方向: -∇loss

所以最朴素的下山策略 (= 梯度下降 gradient descent):
    新位置 = 旧位置 - 学习率 · ∇loss
                              ↑ 减号 = 朝下山方向走
                              ↑ 学习率 lr = 每步走多大

每走一步, 我们都站在新位置上, 重新看脚下哪边是下坡, 再迈下一步。
迭代足够多步后, 就会落进山谷底 = loss 最小处 = 模型训练好了。

这正是 LLM 训练在做的事, 只是参数从 2 个 (w, b) 变成几十亿个。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    torch.manual_seed(0)
    x_data = torch.linspace(-2, 2, 50)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn_like(x_data)

    def loss_fn(w, b):
        y_pred = w * x_data + b
        return ((y_pred - y_data) ** 2).mean()

    # =========================================================
    # 1. 从一个"山顶"位置出发, 一步步下山
    # =========================================================
    w = torch.tensor(-1.5, requires_grad=True)
    b = torch.tensor(-2.5, requires_grad=True)
    lr = 0.1

    trajectory = [(w.item(), b.item(), loss_fn(w, b).item())]
    print("--- 下山过程 (前 10 步) ---")
    print(f"{'step':>4} | {'w':>7} | {'b':>7} | {'loss':>8}")
    print(f"{0:>4} | {w.item():>7.3f} | {b.item():>7.3f} | {trajectory[0][2]:>8.4f}")

    for step in range(1, 81):
        loss = loss_fn(w, b)
        loss.backward()                          # 算 ∇loss
        with torch.no_grad():
            w -= lr * w.grad                     # 沿 -∇loss 走一步 → 下山
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        trajectory.append((w.item(), b.item(), loss.item()))
        if step <= 10 or step % 20 == 0:
            print(f"{step:>4} | {w.item():>7.3f} | {b.item():>7.3f} | {loss.item():>8.4f}")

    print(f"\n终点 (w, b) = ({w.item():.3f}, {b.item():.3f})  (真实最优: (2, 1))")
    print("→ 从山顶滚到了山谷底, loss 从 ~17 降到 ~0.01")

    # =========================================================
    # 2. 对比: 如果学习率太大, 会"跨过山谷"在两壁之间反复横跳
    # =========================================================
    w2 = torch.tensor(-1.5, requires_grad=True)
    b2 = torch.tensor(-2.5, requires_grad=True)
    big_lr = 1.05
    traj_big = [(w2.item(), b2.item())]
    for _ in range(40):
        loss = loss_fn(w2, b2)
        loss.backward()
        with torch.no_grad():
            w2 -= big_lr * w2.grad
            b2 -= big_lr * b2.grad
        w2.grad.zero_()
        b2.grad.zero_()
        traj_big.append((w2.item(), b2.item()))
    print(f"\n学习率过大 (lr={big_lr}) 时终点: ({w2.item():.2f}, {b2.item():.2f}) → 发散了!")

    # =========================================================
    # 3. 画 loss 等高线 + 下山轨迹
    # =========================================================
    ws = torch.linspace(-3, 6, 100)
    bs = torch.linspace(-4, 5, 100)
    W, B = torch.meshgrid(ws, bs, indexing="ij")
    L = torch.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            L[i, j] = loss_fn(W[i, j], B[i, j])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 合适学习率的下山轨迹
    ax = axes[0]
    cs = ax.contour(W.numpy(), B.numpy(), L.numpy(), levels=25, cmap="terrain")
    ax.clabel(cs, inline=True, fontsize=7)
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], "ro-", markersize=3, lw=1, alpha=0.8,
            label=f"下山轨迹 (lr={lr})")
    ax.plot(traj[0, 0], traj[0, 1], "b^", markersize=14, label="起点 (山顶)")
    ax.plot(2, 1, "g*", markersize=20, label="山谷底 (2, 1)")
    ax.set_xlabel("w"); ax.set_ylabel("b")
    ax.set_title("合适的学习率: 稳稳走到谷底")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    # 右: 学习率过大, 反复横跳
    ax = axes[1]
    cs = ax.contour(W.numpy(), B.numpy(), L.numpy(), levels=25, cmap="terrain")
    ax.clabel(cs, inline=True, fontsize=7)
    tb = np.array(traj_big)
    # 截断, 不然画面被甩到天边
    tb_clip = tb[np.all(np.abs(tb) < 50, axis=1)]
    ax.plot(tb_clip[:, 0], tb_clip[:, 1], "mo-", markersize=3, lw=1, alpha=0.8,
            label=f"轨迹 (lr={big_lr})")
    ax.plot(tb[0, 0], tb[0, 1], "b^", markersize=14, label="起点")
    ax.plot(2, 1, "g*", markersize=20, label="山谷底 (2, 1)")
    ax.set_xlabel("w"); ax.set_ylabel("b")
    ax.set_title("学习率过大: 跨过山谷, 越跳越远 → 发散")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("03_minimize_loss_is_downhill.png", dpi=120)
    print("\n图已保存到 03_minimize_loss_is_downhill.png")
    plt.show()


if __name__ == "__main__":
    main()
