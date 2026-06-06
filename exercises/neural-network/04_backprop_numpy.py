"""
04 - 手写反向传播 (纯 numpy, 不用 autograd)

目标:理解 backward 到底在算什么。
网络结构:
    x (2)  ──Linear──>  h_pre (4)  ──ReLU──>  h (4)  ──Linear──>  z (1)  ──Sigmoid──>  y_hat
                                                                                          │
                                                                                          ▼
                                                                                       BCE Loss

参数: W1 (4×2), b1 (4),  W2 (1×4), b2 (1)

前向 forward:
    h_pre = W1 · x + b1
    h     = ReLU(h_pre)
    z     = W2 · h + b2
    y_hat = sigmoid(z)
    L     = -[y log y_hat + (1-y) log(1-y_hat)]

反向 backward (链式法则一步步推):
    dL/dz   = y_hat - y                         # (sigmoid + BCE 的经典化简结果)
    dL/dW2  = (dL/dz) · h.T
    dL/db2  = dL/dz
    dL/dh   = W2.T · (dL/dz)
    dL/dh_pre = dL/dh * relu'(h_pre)            # element-wise
    dL/dW1  = (dL/dh_pre) · x.T
    dL/db1  = dL/dh_pre
"""

import matplotlib.pyplot as plt
import numpy as np

import common


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


def forward(X, W1, b1, W2, b2):
    """X: (N, 2). 返回所有中间量, backward 时要用。"""
    h_pre = X @ W1.T + b1          # (N, 4)
    h     = relu(h_pre)            # (N, 4)
    z     = h @ W2.T + b2          # (N, 1)
    y_hat = sigmoid(z)             # (N, 1)
    cache = (X, h_pre, h, z, y_hat)
    return y_hat, cache


def bce_loss(y_hat, y):
    eps = 1e-9
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


def backward(cache, y, W2):
    """手写反向传播, 返回各参数的梯度。"""
    X, h_pre, h, z, y_hat = cache
    N = X.shape[0]

    # dL/dz = y_hat - y  (BCE + sigmoid 化简)
    dz = (y_hat - y) / N           # (N, 1)

    # 第二层
    dW2 = dz.T @ h                  # (1, 4)
    db2 = dz.sum(axis=0)            # (1,)

    # 反传到 hidden
    dh = dz @ W2                    # (N, 4)
    dh_pre = dh * relu_grad(h_pre)  # (N, 4)

    # 第一层
    dW1 = dh_pre.T @ X              # (4, 2)
    db1 = dh_pre.sum(axis=0)        # (4,)

    return dW1, db1, dW2, db2


def main():
    args = common.parse_args()
    np.random.seed(7)

    # =========================================================
    # 1. XOR 数据
    # =========================================================
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    # =========================================================
    # 2. 初始化参数 (Xavier-ish)
    # =========================================================
    W1 = np.random.randn(4, 2) * 0.7
    b1 = np.zeros(4)
    W2 = np.random.randn(1, 4) * 0.7
    b2 = np.zeros(1)

    lr = 0.5
    epochs = 5000
    losses = []

    print(f"{'epoch':>5} | {'loss':>10} | {'||dW1||':>10} | {'||dW2||':>10}")
    print("-" * 50)

    # =========================================================
    # 3. 训练循环
    # =========================================================
    for epoch in range(epochs):
        y_hat, cache = forward(X, W1, b1, W2, b2)
        loss = bce_loss(y_hat, y)
        losses.append(loss)

        dW1, db1, dW2, db2 = backward(cache, y, W2)

        # 梯度下降
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 500 == 0 or epoch == epochs - 1:
            print(f"{epoch:>5} | {loss:>10.6f} | "
                  f"{np.linalg.norm(dW1):>10.4f} | {np.linalg.norm(dW2):>10.4f}")

    # =========================================================
    # 4. 检验最终预测
    # =========================================================
    y_hat, _ = forward(X, W1, b1, W2, b2)
    print("\n最终预测:")
    for xi, yi, yh in zip(X, y, y_hat):
        print(f"  {xi} → y_true={yi[0]:.0f}, y_hat={yh[0]:.4f}, "
              f"pred={'✓' if (yh[0] > 0.5) == bool(yi[0]) else '✗'}")

    # =========================================================
    # 5. 可视化: loss 曲线 + 决策边界
    # =========================================================
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

        ax1.plot(losses, color="tab:blue")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("BCE loss")
        ax1.set_title("Loss 曲线 (手写 backprop)")
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # 决策边界
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
        path = common.save_fig("04_backprop_numpy")
        print(f"\n图已保存到 {path}")
    else:
        print("\n(未画图。加 --plot 生成 plots/04_backprop_numpy.png)")


if __name__ == "__main__":
    main()
