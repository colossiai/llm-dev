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
    (1) y_pred[i] = w · x[i] + b             ← dL/dw   = Σ ( dL/d(y_pred[i]) ·  d(y_pred[i])/d(w) ) = Σ dL/dy_pred[i] · x[i]
                                             ← dL/d(b)   = Σ ( dL/d(y_pred[i]) ·  d(y_pred[i])/d(b) ) = Σ dL/d(y_pred[i]) · 1
    (2) r[i]      = y_pred[i] - y[i]         ← dL/d(y_pred[i]) = dL/d(r[i]) · d(r[i])/d(y_pred[i]) = dL/d(r[i]) · 1
    (3) sq[i]     = r[i]²                    ← dL/d(r[i])      = dL/d(sq[i]) · d(sp[i])/d(r[i]) = dL/dsq[i] · 2·r[i]
    (4) loss      = (1/N) · Σ sq[i]          ← dL/d(sq[i])     = 1/N

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


def plot_computation_graph(savepath="03_computation_graph_forward_backward.png"):
    """画 forward/backward 计算图: 上排蓝色 = 前向; 下排红色 = 反向 (链式法则)。
    两排节点一一对应: 上面的 y_pred 对应下面的 dL/dy_pred, 一目了然。
    """
    fig, ax = plt.subplots(figsize=(16, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    BLUE = "#1f77b4"
    RED = "#d62728"

    # ---- 上排 forward 节点 (y=5.8) ----
    y_f = 5.8
    fwd_nodes = [
        (1.5,  "w, b",   "lightblue"),
        (5,    "y_pred", "white"),
        (8,    "r",      "white"),
        (11,   "sq",     "white"),
        (14,   "loss",   "lightgreen"),
    ]
    fwd_edges = [  # (x1, x2, 操作说明)
        (1.5, 5,    "(1) y_pred = w·x + b"),
        (5,   8,    "(2) r = y_pred - y"),
        (8,   11,   "(3) sq = r·r"),
        (11,  14,   "(4) loss = mean(sq)"),
    ]

    # ---- 下排 backward 节点 (y=2.2), 与上排一一对应 ----
    y_b = 2.2
    bwd_nodes = [
        (1.5,  "dL/dw\ndL/db",  "lightcoral"),
        (5,    "dL/dy_pred",    "mistyrose"),
        (8,    "dL/dr",         "mistyrose"),
        (11,   "dL/dsq",        "mistyrose"),
        (14,   "dL/dloss = 1",  "lightyellow"),
    ]
    bwd_edges = [  # (x1, x2, 链式法则的局部导数)
        (14, 11, "· 1/N"),
        (11, 8,  "· 2·r"),
        (8,  5,  "· 1"),
        (5,  1.5, "对 w: · x[i] 后求和\n对 b: · 1     后求和"),
    ]

    def draw_node(x, y, label, color):
        n_lines = label.count("\n") + 1
        h = 0.55 + 0.32 * (n_lines - 1)
        ax.add_patch(plt.Rectangle((x - 1, y - h / 2), 2, h,
                                   facecolor=color, edgecolor="black", lw=1.5, zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=4)

    for x, lbl, col in fwd_nodes:
        draw_node(x, y_f, lbl, col)
    for x, lbl, col in bwd_nodes:
        draw_node(x, y_b, lbl, col)

    # forward 箭头: 向右, 蓝色
    for x1, x2, lbl in fwd_edges:
        ax.annotate("", xy=(x2 - 1.05, y_f), xytext=(x1 + 1.05, y_f),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.5))
        ax.text((x1 + x2) / 2, y_f + 0.55, lbl, color=BLUE, fontsize=10.5,
                ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=BLUE))

    # backward 箭头: 向左, 红色
    for x1, x2, lbl in bwd_edges:
        ax.annotate("", xy=(x2 + 1.05, y_b), xytext=(x1 - 1.05, y_b),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2.5))
        ax.text((x1 + x2) / 2, y_b - 0.75, lbl, color=RED, fontsize=10,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=RED))

    # forward → backward 的"种子": loss 这一列向下接 dL/dloss=1
    ax.annotate("", xy=(14, y_b + 0.45), xytext=(14, y_f - 0.45),
                arrowprops=dict(arrowstyle="->", color="black", lw=2, linestyle="dashed"))
    ax.text(14.7, (y_f + y_b) / 2, "种子\ndL/dloss = 1",
            fontsize=10, va="center", ha="left")

    # 上下层的"对照虚线" — 提示同一个变量, 前向值 vs 反向梯度
    for (x, _, _), (xb, _, _) in zip(fwd_nodes, bwd_nodes):
        if x == 14:  # loss 这列已经画过实线了
            continue
        ax.plot([x, xb], [y_f - 0.45, y_b + 0.45], color="gray",
                lw=0.8, linestyle=":", alpha=0.5, zorder=1)

    # 标题 + 说明
    ax.text(8, 7.9, "Forward (前向): 顺着蓝箭头, 从 (w, b) 一路算到 loss",
            fontsize=14, ha="center", color=BLUE, fontweight="bold")
    ax.text(8, 0.5,
            "Backward (反向): 从 dL/dloss=1 出发, 沿红箭头依次乘上每一步的局部导数, 把梯度传回每个变量",
            fontsize=12, ha="center", color=RED, fontweight="bold")

    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"计算图已保存到 {savepath}")


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
    dsq = torch.full_like(r, 1.0 / N) # full_like 创建一个与已有 Tensor 形状相同的新 Tensor，并用指定值填充。

    # (3) sq[i] = r[i]²           →  dL/dr[i] = dL/dsq[i] · 2·r[i]
    dr = dsq * 2.0 * r

    # (2) r[i] = y_pred[i] - y[i] →  dL/dy_pred[i] = dL/dr[i] · 1
    dy_pred = dr

# 把 (4)→(3)→(2)→(1) 串起来化简, 就得到我们之前那两条手算公式:
#     ∂loss/∂w = (2/N) · Σ x[i] · r[i]  = 2 · mean(x · r)
#     ∂loss/∂b = (2/N) · Σ      r[i]    = 2 · mean(r)
# 其中 r = y_pred - y_data 是残差。


    # (1) y_pred[i] = w·x[i] + b  →  对 w 累加 x[i], 对 b 累加 1
    dw = (dy_pred * x_data).sum()
    db = dy_pred.sum()
    return dw, db


def main():
    torch.manual_seed(0)
    x_data = torch.linspace(-2, 2, 50)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn_like(x_data)

    # =========================================================
    # 0a. 先把 forward / backward 的计算图画出来, 对照后面的代码看
    # =========================================================
    plot_computation_graph()

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
