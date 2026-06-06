"""
01 - 单个神经元(Perceptron)

直觉:
  - 一个神经元 = 一条直线把平面切两半
  - 公式: y = step(W·x + b)
      * W·x + b > 0  → 输出 1 (正类)
      * W·x + b ≤ 0  → 输出 0 (负类)
  - 训练规则(感知机更新): 预测错了就把 W 向正确方向挪一点
      W ← W + lr * (y_true - y_pred) * x
      b ← b + lr * (y_true - y_pred)

这是最小可训练单元, 后面所有神经网络都是它的堆叠 + 非线性激活。
"""

import matplotlib.pyplot as plt
import numpy as np

import common


def step(z):
    return (z > 0).astype(np.float64)


def main():
    args = common.parse_args()
    np.random.seed(42)

    # =========================================================
    # 1. 造数据: 两堆 2D 点, 线性可分 (反对角线方向)
    #    类 0: 中心 (-2,  2), 类 1: 中心 (2, -2)
    #    最优 W 方向是 (1, -1), 初始随机 W 很可能错, 必须真正学
    # =========================================================
    n_per_class = 50
    X0 = np.random.randn(n_per_class, 2) * 0.8 + np.array([-2,  2])
    X1 = np.random.randn(n_per_class, 2) * 0.8 + np.array([ 2, -2])
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

    # print('X0')
    # print(X0)
    # print('X1')
    # print(X1)

    # =========================================================
    # 2. 初始化神经元参数 (故意让初始方向偏离最优, 看训练过程)
    # =========================================================
    W = np.array([-0.5, -0.5])  # 初始时和最优 (1, -1) 几乎垂直, 错分一半
    b = 0.0
    lr = 0.05
    epochs = 30

    print(f"初始 W = {W}, b = {b:.4f}")
    print("-" * 50)

    # =========================================================
    # 3. 训练: 感知机更新规则
    # =========================================================
    for epoch in range(epochs):
        errors = 0
        for i in range(len(X)):
            z = W @ X[i] + b
            y_pred = step(z)
            err = y[i] - y_pred
            if err != 0:
                W += lr * err * X[i]
                b += lr * err
                errors += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            preds = step(X @ W + b)
            acc = (preds == y).mean()
            print(f"Epoch {epoch:3d}: 错分 {errors:3d} 次, accuracy = {acc:.3f}")

    print("-" * 50)
    print(f"学到的 W = {W}, b = {b:.4f}")
    print(f"决策边界方程: {W[0]:.3f} * x1 + {W[1]:.3f} * x2 + {b:.3f} = 0")

    # =========================================================
    # 4. 可视化: 数据 + 决策边界
    # =========================================================
    if args.plot:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(X0[:, 0], X0[:, 1], c="tab:blue", label="类 0", alpha=0.7)
        ax.scatter(X1[:, 0], X1[:, 1], c="tab:red", label="类 1", alpha=0.7)

        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        # W[0]*x + W[1]*y + b = 0  =>  y = -(W[0]*x + b) / W[1]
        y_line = -(W[0] * x_line + b) / W[1]
        ax.plot(x_line, y_line, "g--", linewidth=2, label="决策边界")

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("单个神经元 = 一条直线分两类")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.show()
    else:
        print("\n(未画图。加 --plot 生成 plots/01_perceptron.png)")


if __name__ == "__main__":
    main()
