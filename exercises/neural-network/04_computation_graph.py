"""
画 04_backprop_numpy.py 这张 2 层 XOR 网络的 forward/backward 计算图。

  · 蓝色链 = forward       (h_pre → h → z → y_hat → L)
  · 红色链 = backward      (从 dL/dL=1 沿链式法则反推回每个中间量)
  · 橙色分支 = 参数梯度    (dW1, db1, dW2, db2 — 训练时 W -= lr · dW 用这四个)
  · 黑色虚线 = 反向传播种子 dL/dL = 1, 把 forward 和 backward 接起来

跑法:
  python 04_computation_graph.py            # 仅显示
  python 04_computation_graph.py --save     # 保存到 plots/
"""

import matplotlib.pyplot as plt

import common


def plot_graph(ax):
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")

    BLUE = "#1f77b4"
    RED = "#d62728"
    ORANGE = "#ff7f0e"
    GRAY = "#7f7f7f"

    # ============ Forward chain (上排) ============
    y_f = 7.5
    fwd_nodes = [
        (3.5,  "h_pre",  "white"),
        (6.7,  "h",      "white"),
        (10,   "z",      "white"),
        (13,   "y_hat",  "white"),
        (16,   "L",      "lightgreen"),
    ]
    # (x1, x2, 操作标签); 最左边一根 stub 是 "Linear1"
    fwd_edges = [
        (1.0,  3.5,  "Linear1\nX · W1.T + b1"),
        (3.5,  6.7,  "ReLU"),
        (6.7,  10,   "Linear2\nh · W2.T + b2"),
        (10,   13,   "sigmoid"),
        (13,   16,   "BCE\n(y_hat, y)"),
    ]

    # ============ Backward chain (下排) ============
    y_b = 4.8
    bwd_nodes = [
        (3.5,  "dL/dh_pre",  "mistyrose"),
        (6.7,  "dL/dh",      "mistyrose"),
        (10,   "dL/dz",      "lightcoral"),    # 高亮: sigmoid+BCE 在这里化简成 y_hat-y
        (13,   "dL/dy_hat",  "mistyrose"),
        (16,   "dL/dL = 1",  "lightyellow"),
    ]
    # (x1, x2, 局部导数 = 这一步要乘的东西); 注意箭头方向是 x1 → x2 (右→左)
    bwd_edges = [
        (16, 13,  "BCE 的导数"),
        (13, 10,  "化简: dL/dz\n= (y_hat - y) / N"),
        (10, 6.7, "· W2\n(把误差传回 h)"),
        (6.7, 3.5, "⊙ relu'(h_pre)\n(穿过激活)"),
        (3.5, 1.0, "(传到 X, 通常不要)"),
    ]

    # ============ Parameter gradients (最下排) ============
    y_g = 1.5
    # (x, label, 从 backward chain 哪个节点分支下来)
    grads = [
        (2.0,  "dL/dW1\n= dh_pre.T @ X",   3.5),
        (5.0,  "dL/db1\n= sum(dh_pre)",    3.5),
        (8.5,  "dL/dW2\n= dz.T @ h",       10),
        (11.5, "dL/db2\n= sum(dz)",        10),
    ]

    # ============ 画节点的小工具 ============
    def draw_node(x, y, label, color, w=2.2, extra_h=0.0):
        n_lines = label.count("\n") + 1
        h = 0.55 + 0.32 * (n_lines - 1) + extra_h
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h,
                                   facecolor=color, edgecolor="black",
                                   lw=1.5, zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", zorder=4)

    for x, lbl, col in fwd_nodes:
        draw_node(x, y_f, lbl, col)
    for x, lbl, col in bwd_nodes:
        draw_node(x, y_b, lbl, col)
    for x, lbl, _ in grads:
        draw_node(x, y_g, lbl, "sandybrown", w=2.5)

    # ============ Forward 箭头 (蓝色, 向右) ============
    HALF = 1.1
    for x1, x2, lbl in fwd_edges:
        # 最左边 stub: 没有起点节点, 短一点
        start_off = 0.0 if x1 == 1.0 else HALF
        ax.annotate("", xy=(x2 - HALF, y_f), xytext=(x1 + start_off, y_f),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.5))
        ax.text((x1 + x2) / 2, y_f + 0.7, lbl, color=BLUE, fontsize=9.5,
                ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=BLUE))

    # ============ Backward 箭头 (红色, 向左) ============
    for x1, x2, lbl in bwd_edges:
        end_off = 0.0 if x2 == 1.0 else HALF
        ax.annotate("", xy=(x2 + end_off, y_b), xytext=(x1 - HALF, y_b),
                    arrowprops=dict(arrowstyle="->", color=RED, lw=2.5))
        ax.text((x1 + x2) / 2, y_b - 0.75, lbl, color=RED, fontsize=9,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=RED))

    # ============ Forward → Backward 的种子 (L 接到 dL/dL=1) ============
    ax.annotate("", xy=(16, y_b + 0.4), xytext=(16, y_f - 0.4),
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=2, linestyle="dashed"))
    ax.text(16.5, (y_f + y_b) / 2, "种子\ndL/dL = 1",
            fontsize=10, va="center", ha="left")

    # ============ 上下对齐的灰色虚线 (同一个变量的前向值 / 反向梯度) ============
    for (xf, _, _), (xb, _, _) in zip(fwd_nodes[:-1], bwd_nodes[:-1]):
        ax.plot([xf, xb], [y_f - 0.4, y_b + 0.4],
                color=GRAY, lw=0.8, linestyle=":", alpha=0.5, zorder=1)

    # ============ 参数梯度分支 (橙色, 从 backward 节点向下) ============
    for x_g, lbl, x_src in grads:
        ax.annotate("", xy=(x_g, y_g + 0.4), xytext=(x_src, y_b - 0.4),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
    # 用一个文字标签解释分支用到了哪个 forward 缓存
    ax.text(3.5, 3.2, "(用到缓存 X)", fontsize=8.5, color=GRAY, ha="center", style="italic")
    ax.text(10,  3.2, "(用到缓存 h)", fontsize=8.5, color=GRAY, ha="center", style="italic")

    # ============ 标题 / 分层说明 ============
    ax.text(9, 10.4, "Forward (蓝): 数据从左到右, 一层层算到 BCE loss",
            fontsize=13, ha="center", color=BLUE, fontweight="bold")
    ax.text(9, 6.3,
            "Backward (红): 从 dL/dL=1 出发, 沿链式法则反向传梯度。"
            "化简: dL/dz = (y_hat - y) / N  ← sigmoid+BCE 组合的'魔法'",
            fontsize=11, ha="center", color=RED, fontweight="bold")
    ax.text(9, 0.3,
            "Parameter gradients (橙): 真正用来更新参数的 4 个梯度。"
            "训练时 W -= lr · dW",
            fontsize=12, ha="center", color=ORANGE, fontweight="bold")


def main():
    args = common.parse_args()
    fig, ax = plt.subplots(figsize=(18, 10))
    plot_graph(ax)
    plt.tight_layout()
    if not args.draw:
        # 默认就把图保存出来 (这个脚本主要目的就是出图)
        args.save = True
        args.draw = True
    common.finalize(args, "04_computation_graph", bbox_inches="tight")


if __name__ == "__main__":
    main()
