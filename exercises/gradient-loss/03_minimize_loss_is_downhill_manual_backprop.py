"""
03 (手动反向传播版) - 把 loss.backward() 拆开, 自己用链式法则求 ∇loss

上一版用了 loss.backward(), PyTorch 在背后帮我们算梯度。
这一版我们不再依赖 autograd, 自己一步步算, 看清楚"反向传播"到底在干啥。

直觉:
  前向 (forward): 从参数 w, b 出发, 一层层算到最后的标量 loss。
  反向 (backward): 从 loss 反着走回去, 用链式法则把"loss 对每个中间量的依赖"
                  一层层传回参数头上, 得到 ∂loss/∂w 和 ∂loss/∂b。

本例的计算图非常短, 把每一层都列出来:

    forward                                  backward (链式法则)
    --------                                 -------------------
    (1) y_pred[i] = w · x[i] + b             ← dL/dw   = Σ dL/dy_pred[i] · x[i]
                                             ← dL/db   = Σ dL/dy_pred[i] · 1
    (2) r[i]      = y_pred[i] - y[i]         ← dL/dy_pred[i] = dL/dr[i] · 1
    (3) sq[i]     = r[i]²                    ← dL/dr[i]      = dL/dsq[i] · 2·r[i]
    (4) loss      = (1/N) · Σ sq[i]          ← dL/dsq[i]     = 1/N

把 (4)→(3)→(2)→(1) 串起来化简, 就得到我们之前那两条手算公式:
    ∂loss/∂w = (2/N) · Σ x[i] · r[i]  = 2 · mean(x · r)
    ∂loss/∂b = (2/N) · Σ      r[i]    = 2 · mean(r)
其中 r = y_pred - y_data 是残差。

"反向传播" 这个词听起来玄, 但本质就是: 链式法则 + 沿计算图反向走一遍。
LLM 的反向传播也是同一回事, 只是计算图有几百层, 而不是 4 步。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def forward(w, b, x_data, y_data):
    """前向: 一路算到 loss, 同时把中间量 (y_pred, r) 记下来, 反向时要用。"""
    y_pred = w * x_data + b           # (1)
    r = y_pred - y_data               # (2) 残差
    sq = r * r                        # (3)
    loss = sq.mean()                  # (4)
    cache = (y_pred, r)               # 反向传播的"工作台" (类似深度框架里的 ctx)
    return loss, cache


def backward(cache, x_data):
    """反向: 从 dL/dloss=1 出发, 沿计算图 (4)→(3)→(2)→(1) 反着走, 算到 dL/dw, dL/db。"""
    y_pred, r = cache
    N = x_data.numel()

    # (4) loss = mean(sq)         →  dL/dsq[i] = 1/N
    dsq = torch.full_like(r, 1.0 / N)

    # (3) sq[i] = r[i]²           →  dL/dr[i] = dL/dsq[i] · 2·r[i]
    dr = dsq * 2.0 * r

    # (2) r[i] = y_pred[i] - y[i] →  dL/dy_pred[i] = dL/dr[i] · 1
    dy_pred = dr

    # (1) y_pred[i] = w·x[i] + b  →  对 w 累加 x[i], 对 b 累加 1
    dw = (dy_pred * x_data).sum()
    db = dy_pred.sum()
    return dw, db


def main():
    torch.manual_seed(0)
    x_data = torch.linspace(-2, 2, 50)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn_like(x_data)

    # =========================================================
    # 0. 先验证一下: 手算 backward 和 autograd 的结果是否一致
    # =========================================================
    w_check = torch.tensor(-1.5, requires_grad=True)
    b_check = torch.tensor(-2.5, requires_grad=True)
    loss_check, cache = forward(w_check, b_check, x_data, y_data)
    loss_check.backward()
    with torch.no_grad():
        dw_manual, db_manual = backward(cache, x_data)

    print("--- 验证: 在 (w=-1.5, b=-2.5) 处 ---")
    print(f"autograd:    ∇loss = [{w_check.grad.item():.6f}, {b_check.grad.item():.6f}]")
    print(f"手动反向传播: ∇loss = [{dw_manual.item():.6f}, {db_manual.item():.6f}]")
    print("→ 完全一致, 说明我们沿着计算图反着走一遍, 算出的就是 autograd 算的东西\n")

    # =========================================================
    # 1. 用手算梯度做梯度下降 (不再调用 .backward())
    # =========================================================
    w = torch.tensor(-1.5)   # 注意: 不需要 requires_grad, 因为我们不用 autograd 了
    b = torch.tensor(-2.5)
    lr = 0.1

    loss0, _ = forward(w, b, x_data, y_data)
    trajectory = [(w.item(), b.item(), loss0.item())]
    print("--- 下山过程 (前 10 步, 全程纯手动反向传播) ---")
    print(f"{'step':>4} | {'w':>7} | {'b':>7} | {'loss':>8}")
    print(f"{0:>4} | {w.item():>7.3f} | {b.item():>7.3f} | {trajectory[0][2]:>8.4f}")

    for step in range(1, 81):
        loss, cache = forward(w, b, x_data, y_data)
        dw, db = backward(cache, x_data)
        w = w - lr * dw                      # 沿 -∇loss 走一步 → 下山
        b = b - lr * db
        trajectory.append((w.item(), b.item(), loss.item()))
        if step <= 10 or step % 20 == 0:
            print(f"{step:>4} | {w.item():>7.3f} | {b.item():>7.3f} | {loss.item():>8.4f}")

    print(f"\n终点 (w, b) = ({w.item():.3f}, {b.item():.3f})  (真实最优: (2, 1))")
    print("→ 跟 autograd 版本走出的轨迹应该一模一样, 因为算的是同一个梯度")

    # =========================================================
    # 2. 对比: 学习率过大, 会"跨过山谷"在两壁之间反复横跳 (照样手算梯度)
    # =========================================================
    w2 = torch.tensor(-1.5)
    b2 = torch.tensor(-2.5)
    big_lr = 1.05
    traj_big = [(w2.item(), b2.item())]
    for _ in range(40):
        _, cache = forward(w2, b2, x_data, y_data)
        dw, db = backward(cache, x_data)
        w2 = w2 - big_lr * dw
        b2 = b2 - big_lr * db
        traj_big.append((w2.item(), b2.item()))
    print(f"\n学习率过大 (lr={big_lr}) 时终点: ({w2.item():.2f}, {b2.item():.2f}) → 发散了!")

    # =========================================================
    # 3. 画 loss 等高线 + 下山轨迹 (和 autograd 版本对比图一致)
    # =========================================================
    ws = torch.linspace(-3, 6, 100)
    bs = torch.linspace(-4, 5, 100)
    W, B = torch.meshgrid(ws, bs, indexing="ij")
    L = torch.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            L[i, j], _ = forward(W[i, j], B[i, j], x_data, y_data)

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
    ax.set_title("合适的学习率: 稳稳走到谷底\n(梯度全部由手动反向传播算出)")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    # 右: 学习率过大, 反复横跳
    ax = axes[1]
    cs = ax.contour(W.numpy(), B.numpy(), L.numpy(), levels=25, cmap="terrain")
    ax.clabel(cs, inline=True, fontsize=7)
    tb = np.array(traj_big)
    tb_clip = tb[np.all(np.abs(tb) < 50, axis=1)]
    ax.plot(tb_clip[:, 0], tb_clip[:, 1], "mo-", markersize=3, lw=1, alpha=0.8,
            label=f"轨迹 (lr={big_lr})")
    ax.plot(tb[0, 0], tb[0, 1], "b^", markersize=14, label="起点")
    ax.plot(2, 1, "g*", markersize=20, label="山谷底 (2, 1)")
    ax.set_xlabel("w"); ax.set_ylabel("b")
    ax.set_title("学习率过大: 跨过山谷, 越跳越远 → 发散")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("03_minimize_loss_is_downhill_manual_backprop.png", dpi=120)
    print("\n图已保存到 03_minimize_loss_is_downhill_manual_backprop.png")
    plt.show()


if __name__ == "__main__":
    main()
