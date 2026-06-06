"""
04 - 手写反向传播 (纯 numpy, 不用 autograd)

================ 给零基础读者的 5 分钟讲解 ================

【这个脚本的目的】
  前面 03 用了 PyTorch 的 loss.backward(), 一行调用就自动算好了所有梯度。
  这里把"那一行"展开, 用纯 numpy 一步步亲手算, 让你看清楚 backward 究竟在做什么。
  这是理解神经网络的关键一步: 真正"看穿"训练机制。

【网络结构 (和上一个脚本几乎一样, 但隐藏层 4 个神经元而非 8 个)】
    x (2) ──Linear──> h_pre (4) ──ReLU──> h (4) ──Linear──> z (1) ──Sigmoid──> y_hat
                                                                                  │
                                                                                  ▼
                                                                              BCE Loss

【参数】
    W1 (4×2), b1 (4)      第 1 层
    W2 (1×4), b2 (1)      第 2 层

【前向 (forward) — 数据从左走到右, 算损失】
    h_pre = W1 · x + b1        ← 第 1 层的线性变换
    h     = ReLU(h_pre)        ← 第 1 层的激活
    z     = W2 · h + b2        ← 第 2 层的线性变换
    y_hat = sigmoid(z)         ← 把任意 z 压到 (0, 1) 当概率
    L     = -[y·log(y_hat) + (1-y)·log(1-y_hat)]   ← BCE 交叉熵

【反向 (backward) — 损失反着传, 算每个参数的"梯度"】
  "梯度" = 损失 L 对该参数的偏导数, 告诉我们"这个参数往哪边动一点 L 会变小"。
  规则: 链式法则 (chain rule)
      dL/dW = dL/d(输出) · d(输出)/dW

  从后往前一层一层算:
    dL/dz   = y_hat - y                  # (sigmoid + BCE 化简的"魔法"结果, 干净到不像话)
    dL/dW2  = (dL/dz) · h.T              # 链式: L → z → W2
    dL/db2  = dL/dz                      # 链式: L → z → b2
    dL/dh   = W2.T · (dL/dz)             # 把误差"反传"到隐藏层
    dL/dh_pre = dL/dh * relu'(h_pre)     # 穿过 ReLU 激活 (逐元素相乘)
    dL/dW1  = (dL/dh_pre) · x.T          # 链式: 一路传到 W1
    dL/db1  = dL/dh_pre

【为什么 dL/dz = y_hat - y 这么简单?】
  数学上, sigmoid + BCE 这个组合的导数刚好把分母分子都消掉, 得到这个极其干净的结果。
  这就是为什么二分类总是用 sigmoid + BCE 这个组合 — 不只是数学习惯, 也是为了数值稳定。

【为什么很多地方有 .T (转置)?】
  矩阵乘法对维度敏感, 反传时需要让形状对得上, 经常要把矩阵"翻一下"。
  你不需要硬记规则, 只要看每一行的 shape 注释就能验证对错。
"""

import matplotlib.pyplot as plt
import numpy as np

import common


def sigmoid(z):
    # 把任意实数压到 (0, 1) 区间
    return 1 / (1 + np.exp(-z))


def relu(x):
    # 负数变 0, 正数不变
    return np.maximum(0, x)


def relu_grad(x):
    # ReLU 的导数: 正数处为 1, 负数处为 0
    return (x > 0).astype(np.float64)


def forward(X, W1, b1, W2, b2):
    """
    前向传播: 给定输入 X 和参数, 算出预测 y_hat。
    同时返回 cache (一堆中间结果), backward 时要用 — 这是省力的关键!
    重新算一遍代价高, 缓存起来直接用更快。

    X: shape (N, 2) — N 个样本, 每个 2 维输入
    """
    h_pre = X @ W1.T + b1          # (N, 2) @ (2, 4) + (4,) → (N, 4) — 广播加 b1
    h     = relu(h_pre)            # (N, 4)
    z     = h @ W2.T + b2          # (N, 4) @ (4, 1) + (1,) → (N, 1)
    y_hat = sigmoid(z)             # (N, 1) — 每行一个概率
    cache = (X, h_pre, h, z, y_hat)
    return y_hat, cache


def bce_loss(y_hat, y):
    """
    Binary Cross Entropy 二分类交叉熵损失。
    eps = 1e-9 是为了防止 log(0) 出现 -inf。
    """
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


def backward(cache, y, W2):
    """
    手写反向传播: 用链式法则一步步算出每个参数的梯度。
    返回 dW1, db1, dW2, db2 — 训练时用 W -= lr * dW 来更新参数。
    """
    X, h_pre, h, z, y_hat = cache
    N = X.shape[0]   # 样本数, 用来求平均

    # ====== 从最后一层往回算 ======

    # dL/dz: sigmoid + BCE 的化简结果, 除以 N 是因为 loss 取了 mean
    dz = (y_hat - y) / N            # (N, 1)

    # ====== 第二层 (W2, b2) 的梯度 ======
    # 链式: L → z → W2.  推导: z = h @ W2.T, 所以 dz/dW2 = h
    dW2 = dz.T @ h                  # (1, N) @ (N, 4) = (1, 4)
    # 偏置 b2 直接是 dz 按样本维度加起来
    db2 = dz.sum(axis=0)            # 沿 batch 维度求和, shape (1,)

    # ====== 把误差反传到隐藏层 ======
    # 链式: L → z → h.  推导: z = h @ W2.T, 所以 dz/dh = W2
    dh = dz @ W2                    # (N, 1) @ (1, 4) = (N, 4)
    # 穿过 ReLU 激活: 用 relu 的导数 (元素级相乘)
    # 直觉: ReLU 在负半轴梯度为 0, 所以负半轴的位置"断路", 反传不过去
    dh_pre = dh * relu_grad(h_pre)  # (N, 4) * (N, 4) = (N, 4)

    # ====== 第一层 (W1, b1) 的梯度 ======
    # 链式: L → ... → h_pre → W1.  推导: h_pre = X @ W1.T, 所以 dh_pre/dW1 = X
    dW1 = dh_pre.T @ X              # (4, N) @ (N, 2) = (4, 2)
    db1 = dh_pre.sum(axis=0)        # (4,)

    return dW1, db1, dW2, db2


def main():
    args = common.parse_args()
    np.random.seed(7)   # 固定随机种子, 训练过程可复现

    # =========================================================
    # 1. XOR 数据 (4 个点, 经典的"线性不可分"问题)
    # =========================================================
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # =========================================================
    # 2. 随机初始化参数 (小的随机数, 不能全 0 否则所有神经元学一样)
    # =========================================================
    W1 = np.random.randn(4, 2) * 0.7   # 第 1 层权重
    b1 = np.zeros(4)                   # 第 1 层偏置 (0 初始化 OK)
    W2 = np.random.randn(1, 4) * 0.7   # 第 2 层权重
    b2 = np.zeros(1)                   # 第 2 层偏置

    lr = 0.5          # 学习率: 每步走多远 (太大震荡, 太小太慢)
    epochs = 5000     # 训练轮数 (每轮把 4 个 XOR 样本喂一遍)
    losses = []

    print(f"{'epoch':>5} | {'loss':>10} | {'||dW1||':>10} | {'||dW2||':>10}")
    print("-" * 50)

    # =========================================================
    # 3. 训练循环: forward → loss → backward → 更新参数
    # =========================================================
    for epoch in range(epochs):
        # 前向: 算预测和损失
        y_hat, cache = forward(X, W1, b1, W2, b2)
        loss = bce_loss(y_hat, y)
        losses.append(loss)

        # 反向: 算所有参数的梯度
        dW1, db1, dW2, db2 = backward(cache, y, W2)

        # 梯度下降: 朝梯度反方向走一小步 (lr 控制步长)
        # 减号是因为梯度指向"loss 增大的方向", 我们要 loss 变小所以反着走
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        # 每 500 轮打印一次进度
        # ||dW|| 是梯度的范数 (大小), 训练到位时会越来越小 (接近收敛)
        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"{epoch:>5} | {loss:>10.6f} | "
                  f"{np.linalg.norm(dW1):>10.4f} | {np.linalg.norm(dW2):>10.4f}")

    # =========================================================
    # 4. 检验最终预测: 看 4 个 XOR 点是否都猜对了
    # =========================================================
    y_hat, _ = forward(X, W1, b1, W2, b2)
    print("\n最终预测:")
    for xi, yi, yh in zip(X, y, y_hat):
        print(f"  {xi} → y_true={yi[0]:.0f}, y_hat={yh[0]:.4f}, "
              f"pred={'✓' if (yh[0] > 0.5) == bool(yi[0]) else '✗'}")

    # =========================================================
    # 5. 可视化: loss 曲线 + 决策边界
    # =========================================================
    if args.draw:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        # Loss 曲线: 用对数刻度 (因为 loss 跨好几个数量级)
        ax1.plot(losses, color="tab:blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("BCE loss")
        ax1.set_title("Loss 曲线 (手写 backprop)")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # 决策边界: 在平面上每个位置都预测一下, 看模型把空间怎么"切"
        xs = np.linspace(-0.3, 1.3, 200)
        ys = np.linspace(-0.3, 1.3, 200)
        XX, YY = np.meshgrid(xs, ys)
        grid = np.c_[XX.ravel(), YY.ravel()]
        yh, _ = forward(grid, W1, b1, W2, b2)
        ZZ = yh.reshape(XX.shape)

        ax2.contourf(XX, YY, ZZ, levels=20, cmap="RdBu", alpha=0.7)
        ax2.contour(XX, YY, ZZ, levels=[0.5], colors="k", linewidths=2)
        ax2.scatter(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], c="blue", s=180, edgecolors="k", label="类 0")
        ax2.scatter(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], c="red", s=180, edgecolors="k", label="类 1")
        ax2.set_title("学到的决策边界")
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        common.finalize(args, "04_backprop_numpy")
    else:
        print("\n(未画图。加 --plot 显示图, --save 保存到 plots/)")


if __name__ == "__main__":
    main()
