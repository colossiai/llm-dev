"""
03 - 为什么需要非线性激活? (XOR 实验)

直觉:
  - 没有非线性, 多层网络等价于一层
      Linear ∘ Linear = Linear  (W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1+b2))
  - XOR 是线性不可分的: 没有一条直线能把 (0,0)/(1,1) 和 (0,1)/(1,0) 分开
  - 用线性"激活" (恒等映射 y=x) 的 MLP → 学不会 XOR
  - 用 ReLU 的 MLP → 能学会

本脚本同结构、同初始化、同优化器, 唯一差别是激活函数, 看决策边界对比。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import common


class TinyMLP(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 1)
        self.act = activation

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def train(model, X, y, epochs=3000, lr=0.05):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(epochs):
        logits = model(X).squeeze(-1)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def accuracy(model, X, y):
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)).squeeze(-1) > 0.5).float()
        return (preds == y).float().mean().item()


def plot_boundary(ax, model, X, y, title):
    xs = np.linspace(-0.5, 1.5, 200)
    ys = np.linspace(-0.5, 1.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)
    with torch.no_grad():
        ZZ = torch.sigmoid(model(grid)).numpy().reshape(XX.shape)

    ax.contourf(XX, YY, ZZ, levels=20, cmap="RdBu", alpha=0.6)
    ax.contour(XX, YY, ZZ, levels=[0.5], colors="k", linewidths=2)
    Xn = X.numpy()
    yn = y.numpy()
    ax.scatter(Xn[yn == 0, 0], Xn[yn == 0, 1], c="blue", s=120, edgecolors="k", label="类 0")
    ax.scatter(Xn[yn == 1, 0], Xn[yn == 1, 1], c="red", s=120, edgecolors="k", label="类 1")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    args = common.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    # =========================================================
    # 1. XOR 数据 (4 个点)
    # =========================================================
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    print("XOR 数据:")
    print("  (0,0) -> 0")
    print("  (0,1) -> 1")
    print("  (1,0) -> 1")
    print("  (1,1) -> 0")
    print("  → 无法用一条直线分开\n")

    # =========================================================
    # 2. 两个网络: 线性激活 vs ReLU 激活
    # =========================================================
    torch.manual_seed(0)
    model_linear = TinyMLP(activation=nn.Identity())
    torch.manual_seed(0)
    model_relu = TinyMLP(activation=nn.ReLU())

    print("训练线性激活的网络 (相当于一层 Linear)...")
    loss_lin = train(model_linear, X, y)
    acc_lin = accuracy(model_linear, X, y)

    print("训练 ReLU 激活的网络...")
    loss_relu = train(model_relu, X, y)
    acc_relu = accuracy(model_relu, X, y)

    print(f"\n线性激活:  最终 loss = {loss_lin[-1]:.4f}, accuracy = {acc_lin:.3f}")
    print(f"ReLU 激活: 最终 loss = {loss_relu[-1]:.4f}, accuracy = {acc_relu:.3f}")
    print("\n→ 线性激活的网络最多 50% 准确率(随机猜), ReLU 能 100% 学会 XOR")

    # =========================================================
    # 3. 可视化决策边界对比
    # =========================================================
    if args.plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        plot_boundary(axes[0], model_linear, X, y,
                      f"线性激活 (恒等)  acc={acc_lin:.0%}\n→ 只能画直线, 学不会 XOR")
        plot_boundary(axes[1], model_relu, X, y,
                      f"ReLU 激活  acc={acc_relu:.0%}\n→ 能弯曲决策边界, 学会了 XOR")

        plt.tight_layout()
        path = common.save_fig("03_why_nonlinearity")
        print(f"\n图已保存到 {path}")
    else:
        print("\n(未画图。加 --plot 生成 plots/03_why_nonlinearity.png)")


if __name__ == "__main__":
    main()
