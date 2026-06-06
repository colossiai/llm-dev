"""
05 - PyTorch Autograd (与 04 数值对比, 验证手写 backprop 是对的)

================ 给零基础读者的 5 分钟讲解 ================

【上一脚本回顾】
  04 里我们手写了 backward, 用 numpy 一步步把链式法则跑了一遍。
  写过的人都担心一个问题: "我推的导数到底对不对?"

【这个脚本要做什么】
  让 PyTorch 帮我们独立算一遍, 再和 04 的手写结果对比:
    - 同一份数据
    - 同一份初始参数
    - 同一份网络结构
  如果两边算出的梯度数值完全一样 → 04 的推导正确 ✓

【PyTorch Autograd 是什么?】
  autograd = "automatic differentiation", 自动微分。
  你只要写 forward (前向计算), PyTorch 会自动算出 backward (梯度)。
  原理: PyTorch 记录每一步操作组成的"计算图", 然后从 loss 反着遍历图、套链式法则。
  你不用推任何导数公式, 也不用写 backward — 直接 loss.backward() 一句话搞定。

【三个关键 PyTorch 概念】
  1. requires_grad=True
     创建 tensor 时加这个参数, 告诉 PyTorch "请追踪这个张量的所有运算, 我后面要求它的梯度"
     一般只有"参数" (W, b) 需要, 数据 (X, y) 不需要。

  2. loss.backward()
     从 loss 开始反向遍历计算图, 把每个 requires_grad=True 的 tensor 的梯度
     算出来存到它的 .grad 属性里。

  3. tensor.grad
     存储该 tensor 的梯度 (shape 和 tensor 本身一样)。
     例如 W1_t.grad 就是 dL/dW1, 和我们手写的 dW1 应该完全一样。

【为什么要 zero_grad?】
  PyTorch 默认会"累加"梯度 — 你跑一次 backward 它把梯度加到 .grad 里, 不会清零。
  所以训练循环里每轮都要先 zero_grad(), 否则梯度会越积越大、训练崩盘。

【为什么 with torch.no_grad():?】
  做参数更新 (p -= lr * p.grad) 时, 这本身也是一次张量运算。
  如果不用 no_grad() 包住, PyTorch 会把它也加入计算图, 浪费内存且语义错误。
  no_grad() = "暂时关掉 autograd 的记录功能"。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import common


def numpy_forward_backward(X, y, W1, b1, W2, b2):
    """
    复用 04 的手写实现 — 一次 forward + 手写 backward, 算出 loss 和所有梯度。
    这里写完整一遍, 让本脚本独立可读不需要回看 04。
    """
    N = X.shape[0]

    # 前向
    h_pre = X @ W1.T + b1
    h = np.maximum(0, h_pre)
    z = h @ W2.T + b2
    y_hat = 1 / (1 + np.exp(-z))

    # 损失 (BCE)
    eps = 1e-9
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    # 反向 (手写链式法则, 详见 04 的注释)
    dz = (y_hat - y) / N
    dW2 = dz.T @ h
    db2 = dz.sum(axis=0)
    dh = dz @ W2
    dh_pre = dh * (h_pre > 0)
    dW1 = dh_pre.T @ X
    db1 = dh_pre.sum(axis=0)
    return loss, dW1, db1, dW2, db2


def torch_forward_backward(X_t, y_t, W1_t, b1_t, W2_t, b2_t):
    """
    同样的计算, 用 PyTorch + autograd。
    注意: 这里完全没写 backward 代码 — 一句 loss.backward() 就搞定所有梯度!
    """
    # 前向 — 写法和 numpy 几乎一样, 只是用了 torch tensor
    h_pre = X_t @ W1_t.T + b1_t
    h = F.relu(h_pre)
    z = h @ W2_t.T + b2_t

    # 用 binary_cross_entropy_with_logits 而非 BCE(sigmoid(z), y):
    # 后者 = 先 sigmoid 再算 log, 数值上容易 overflow/underflow
    # 前者把这两步合并成一个数值稳定的公式 (内部用 log-sum-exp 技巧)
    loss = F.binary_cross_entropy_with_logits(z, y_t, reduction="mean")

    # 这一句完成所有 backward — PyTorch 自动套链式法则
    # 之后 W1_t.grad, b1_t.grad, W2_t.grad, b2_t.grad 都会被填上
    loss.backward()
    return loss.item()   # .item() 把 0 维 tensor 转成 Python float


def main():
    args = common.parse_args()
    np.random.seed(7)
    torch.manual_seed(0)

    # =========================================================
    # 1. 准备数据 + 同一份初始参数 (numpy 端先生成, torch 端复制过去)
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
    # 3. torch 路径: 同一份数据和参数, 但走 autograd
    #    requires_grad=True 才会被 autograd 追踪
    # =========================================================
    X_t = torch.tensor(X, dtype=torch.float64)                       # 数据, 不追踪梯度
    y_t = torch.tensor(y, dtype=torch.float64)                       # 标签, 不追踪梯度
    W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True) # 参数, 追踪!
    b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
    b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

    loss_t = torch_forward_backward(X_t, y_t, W1_t, b1_t, W2_t, b2_t)

    # =========================================================
    # 4. 对比 loss 和 4 组梯度: 手写 vs autograd
    #    diff 应该接近机器精度 (1e-15 量级)
    # =========================================================
    print("=" * 60)
    print(f"{'量':<10} | {'numpy(手写)':<16} | {'PyTorch(autograd)':<18} | diff")
    print("-" * 60)
    print(f"{'loss':<10} | {loss_np:<16.10f} | {loss_t:<18.10f} | {abs(loss_np - loss_t):.2e}")

    pairs = [
        ("dW1", dW1_np, W1_t.grad.numpy()),   # .grad 取出 autograd 算的梯度
        ("db1", db1_np, b1_t.grad.numpy()),
        ("dW2", dW2_np, W2_t.grad.numpy()),
        ("db2", db2_np, b2_t.grad.numpy()),
    ]
    max_diff = 0.0
    for name, g_np, g_t in pairs:
        d = np.max(np.abs(g_np - g_t))   # 取所有元素差的最大绝对值
        max_diff = max(max_diff, d)
        print(f"{name:<10} | shape={str(g_np.shape):<10}      | shape={str(g_t.shape):<12}        | max |diff| = {d:.2e}")

    print("-" * 60)
    if max_diff < 1e-9:
        print(f"✅ 一致! 最大 diff = {max_diff:.2e}  →  手写 backprop 推导正确")
    else:
        print(f"❌ 不一致! 最大 diff = {max_diff:.2e}")

    # =========================================================
    # 5. 顺便用 autograd 完整训练 XOR 一遍 (5000 epoch)
    #    展示工业代码长什么样: 只写 forward, backward 一行调用
    # =========================================================
    # 重新初始化 (用同样的种子, 和 04 的训练完全可比)
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
        # ==== 步骤 1: 清空上一轮的梯度 ====
        # PyTorch 的 .grad 默认会"累加", 不清零会越积越大
        for p in (W1_t, b1_t, W2_t, b2_t):
            if p.grad is not None:
                p.grad.zero_()

        # ==== 步骤 2: 前向, 算 loss ====
        h_pre = X_t @ W1_t.T + b1_t
        h = F.relu(h_pre)
        z = h @ W2_t.T + b2_t
        loss = F.binary_cross_entropy_with_logits(z, y_t, reduction="mean")
        losses.append(loss.item())

        # ==== 步骤 3: 反向, autograd 自动填 .grad ====
        loss.backward()

        # ==== 步骤 4: 更新参数 ====
        # torch.no_grad() = 这块代码不要被 autograd 追踪
        # 否则参数更新本身也会被记入计算图, 造成内存泄漏和语义错误
        with torch.no_grad():
            for p in (W1_t, b1_t, W2_t, b2_t):
                p -= lr * p.grad   # 朝梯度反方向走 lr 那么远 — 经典梯度下降

    print(f"\nautograd 训练 5000 epoch 后 loss = {losses[-1]:.6f}")

    if args.draw:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(losses, color="tab:orange", label="autograd 训练 (5000 epoch)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.set_yscale("log")
        ax.set_title("PyTorch Autograd 训练 XOR\n(同初值, 同 lr, 曲线应与 04 几乎一致)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        common.finalize(args, "05_autograd_pytorch")
    else:
        print("(未画图。加 --plot 显示图, --save 保存到 plots/)")


if __name__ == "__main__":
    main()
