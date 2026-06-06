"""
05 - PyTorch Autograd (与 04 数值对比, 验证手写 backprop 是对的)

直觉:
  - autograd = PyTorch 自动帮你做 04 里那一堆链式法则求导
  - 同样的网络、同样的初始化、同样的数据, 手写 backward 和 autograd 算出来的梯度应该一致
  - 这个脚本就用同样初值跑一步 forward+backward, 对比两套梯度

为什么这样验证有意义:
  自己实现的反传如果有 bug, 数值会和 autograd 不一致 → 能立刻发现错误。
  这是工业界写新算子时必做的"梯度检查"。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import common


def numpy_forward_backward(X, y, W1, b1, W2, b2):
    """复用 04 的手写实现 (省去 import, 直接重写一遍, 保持脚本独立可跑)"""
    N = X.shape[0]
    h_pre = X @ W1.T + b1
    h = np.maximum(0, h_pre)
    z = h @ W2.T + b2
    y_hat = 1 / (1 + np.exp(-z))

    eps = 1e-9
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    dz = (y_hat - y) / N
    dW2 = dz.T @ h
    db2 = dz.sum(axis=0)
    dh = dz @ W2
    dh_pre = dh * (h_pre > 0)
    dW1 = dh_pre.T @ X
    db1 = dh_pre.sum(axis=0)
    return loss, dW1, db1, dW2, db2


def torch_forward_backward(X_t, y_t, W1_t, b1_t, W2_t, b2_t):
    """同一计算, 用 PyTorch tensor + autograd"""
    h_pre = X_t @ W1_t.T + b1_t
    h = F.relu(h_pre)
    z = h @ W2_t.T + b2_t

    # 用 binary_cross_entropy_with_logits 数值更稳, 和手写 BCE(sigmoid(z), y) 等价
    loss = F.binary_cross_entropy_with_logits(z, y_t, reduction="mean")
    loss.backward()
    return loss.item()


def main():
    args = common.parse_args()
    np.random.seed(7)
    torch.manual_seed(0)

    # =========================================================
    # 1. 同一份数据 + 同一份初始参数 (从 numpy 同步到 torch)
    # =========================================================
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y = np.array([[0], [1], [1], [0]], dtype=np.float64)

    W1 = np.random.randn(4, 2) * 0.7
    b1 = np.zeros(4)
    W2 = np.random.randn(1, 4) * 0.7
    b2 = np.zeros(1)

    # =========================================================
    # 2. numpy 路径: 一次 forward + 手写 backward
    # =========================================================
    loss_np, dW1_np, db1_np, dW2_np, db2_np = numpy_forward_backward(
        X, y, W1, b1, W2, b2
    )

    # =========================================================
    # 3. torch 路径: 同一份参数和数据, autograd
    # =========================================================
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    loss_t = torch_forward_backward(X_t, y_t, W1_t, b1_t, W2_t, b2_t)

    # =========================================================
    # 4. 对比 loss 和各梯度
    # =========================================================
    print("=" * 60)
    print(f"{'量':<10} | {'numpy(手写)':<16} | {'PyTorch(autograd)':<18} | diff")
    print("-" * 60)
    print(f"{'loss':<10} | {loss_np:<16.10f} | {loss_t:<18.10f} | {abs(loss_np - loss_t):.2e}")

    pairs = [
        ("dW1", dW1_np, W1_t.grad.numpy()),
        ("db1", db1_np, b1_t.grad.numpy()),
        ("dW2", dW2_np, W2_t.grad.numpy()),
        ("db2", db2_np, b2_t.grad.numpy()),
    ]
    max_diff = 0.0
    for name, g_np, g_t in pairs:
        d = np.max(np.abs(g_np - g_t))
        max_diff = max(max_diff, d)
        print(f"{name:<10} | shape={str(g_np.shape):<10}      | shape={str(g_t.shape):<12}        | max |diff| = {d:.2e}")

    print("-" * 60)
    if max_diff < 1e-9:
        print(f"✅ 一致! 最大 diff = {max_diff:.2e}  →  手写 backprop 推导正确")
    else:
        print(f"❌ 不一致! 最大 diff = {max_diff:.2e}")

    # =========================================================
    # 5. 顺便用 autograd 把 XOR 训完 (即使不画图也跑, 验证能收敛)
    # =========================================================
    # 重新初始化用同样的种子, 训 5000 epoch
    np.random.seed(7)
    W1 = np.random.randn(4, 2) * 0.7
    b1 = np.zeros(4)
    W2 = np.random.randn(1, 4) * 0.7
    b2 = np.zeros(1)

    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    lr = 0.5
    losses = []
    for _ in range(5000):
        for p in (W1_t, b1_t, W2_t, b2_t):
            if p.grad is not None:
                p.grad.zero_()

        h_pre = X_t @ W1_t.T + b1_t
        h = F.relu(h_pre)
        z = h @ W2_t.T + b2_t
        loss = F.binary_cross_entropy_with_logits(z, y_t, reduction="mean")
        loss.backward()
        losses.append(loss.item())

        with torch.no_grad():
            for p in (W1_t, b1_t, W2_t, b2_t):
                p -= lr * p.grad

    print(f"\nautograd 训练 5000 epoch 后 loss = {losses[-1]:.6f}")

    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(losses, color="tab:orange", label="autograd 训练 (5000 epoch)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.set_yscale("log")
        ax.set_title("PyTorch Autograd 训练 XOR\n(同初值, 同 lr, 曲线应与 04 几乎一致)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        path = common.save_fig("05_autograd_pytorch")
        print(f"图已保存到 {path}")
    else:
        print("(未画图。加 --plot 生成 plots/05_autograd_pytorch.png)")


if __name__ == "__main__":
    main()
