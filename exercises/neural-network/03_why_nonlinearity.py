"""
03 - 为什么需要非线性激活? (XOR 实验)

================ 给零基础读者的 5 分钟讲解 ================

【这个脚本要证明什么?】
  上一个脚本 02 说"激活函数让网络能学非线性"。
  这个脚本用一个经典反例直接验证: XOR 问题。

【什么是 XOR?】
  XOR (异或) = "两个输入不同时输出 1, 相同时输出 0"
      (0, 0) → 0     (1, 1) → 0
      (0, 1) → 1     (1, 0) → 1
  把这 4 个点画在 2D 平面上:
       ●(0,1)=1   ●(1,1)=0       ← 红和蓝交错
       ●(0,0)=0   ●(1,0)=1
  你会发现: **无论画一条什么直线, 都不能让红蓝两类各在一边。**
  → XOR 是"线性不可分"的最经典例子。

【实验设计】
  造两个网络, 除了激活函数不同外, 其他完全一样:
    - model_linear: 用 nn.Identity() (恒等映射 y=x, 等于"没激活")
    - model_relu  : 用 nn.ReLU()
  都训练 3000 轮, 看谁能学会 XOR。

【期待结果】
  线性激活 → 50% 准确率 (相当于随机猜, 因为多层线性 = 一层线性 = 一条直线)
  ReLU 激活 → 100% 准确率 (能折出弯曲的决策边界)

【新出现的 PyTorch 概念】
  nn.Module        网络的"基类", 自定义网络要继承它
  nn.Linear(in,out) 一层 W·x+b 运算, in/out 是输入/输出维度
  forward(x)       定义"输入 x 怎么一步步算到输出"
  nn.BCEWithLogitsLoss  二分类损失 (= sigmoid + 交叉熵, 但数值更稳)
  torch.optim.Adam      自适应学习率的梯度下降 (现代默认优化器)
  opt.zero_grad()       清空上一轮的梯度 (PyTorch 默认会累加梯度)
  loss.backward()       自动反向传播, 算出每个参数的梯度
  opt.step()            按梯度更新参数 (= W -= lr * W.grad 类似)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import common


class TinyMLP(nn.Module):
    """
    一个 3 层的小型 MLP (多层感知机):
        输入 2 维 → 隐藏 8 维 → 隐藏 8 维 → 输出 1 维
    activation 参数是可换的, 用来对比"有非线性 vs 没非线性"。
    """
    def __init__(self, activation):
        super().__init__()
        # nn.Linear(in_features, out_features) = 一个 W·x + b 层
        # W 的 shape 是 (out_features, in_features), b 是 (out_features,)
        # PyTorch 自动用合理的随机数初始化 W, b
        self.fc1 = nn.Linear(2, 8)   # 第 1 层: 2 → 8
        self.fc2 = nn.Linear(8, 8)   # 第 2 层: 8 → 8
        self.fc3 = nn.Linear(8, 1)   # 第 3 层: 8 → 1 (输出一个 logit)
        self.act = activation

    def forward(self, x):
        # forward = "前向传播", 定义数据从输入到输出怎么流
        # 注意: 最后一层不加激活, 因为后面的损失函数会自己处理
        x = self.act(self.fc1(x))    # 第 1 层 + 激活
        x = self.act(self.fc2(x))    # 第 2 层 + 激活
        return self.fc3(x)           # 第 3 层 (输出 logit, 没激活)


def train(model, X, y, epochs=3000, lr=0.05):
    """
    通用训练函数: 给定模型、数据、轮数, 跑梯度下降。
    """
    # Adam 是一种"自适应学习率"的优化器, 比朴素 SGD 收敛更快、更稳
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # BCEWithLogitsLoss = "二分类用的交叉熵 + sigmoid"
    # 它直接接受 logit (没经过 sigmoid 的原始输出), 数值更稳定
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(epochs):
        # squeeze(-1): 把 shape (N, 1) 压成 (N,), 和 y 的形状对齐
        logits = model(X).squeeze(-1)
        loss = loss_fn(logits, y)

        # ====== PyTorch 训练的"三件套" ======
        opt.zero_grad()    # 1. 清空梯度 (上一轮算的梯度还留着, 不清就累加)
        loss.backward()    # 2. 反向传播 (自动算每个参数的 .grad)
        opt.step()         # 3. 更新参数 (按 .grad 修改 W, b)

        losses.append(loss.item())
    return losses


def accuracy(model, X, y):
    """计算模型在 (X, y) 上的准确率。"""
    # torch.no_grad() = "下面这些操作不要追踪梯度" (推理时节省内存)
    with torch.no_grad():
        # 模型输出是 logit (任意实数), 用 sigmoid 压到 (0,1) 当概率
        # > 0.5 当作类 1, 否则类 0
        preds = (torch.sigmoid(model(X)).squeeze(-1) > 0.5).float()
        return (preds == y).float().mean().item()


def plot_boundary(ax, model, X, y, title):
    """
    画"决策边界": 在平面上每个位置都让模型预测一下,
    把"概率高的区域"涂红, "概率低的区域"涂蓝, 中间画一条 0.5 的等高线。
    """
    # 在 -0.5 到 1.5 之间撒一个 200×200 的网格
    xs = np.linspace(-0.5, 1.5, 200)
    ys = np.linspace(-0.5, 1.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    # 把 4 万个点拼成 (40000, 2) 输入网络
    grid = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)
    with torch.no_grad():
        # 每个点的预测概率 (0~1)
        ZZ = torch.sigmoid(model(grid)).numpy().reshape(XX.shape)

    # contourf = 填色等高线 (按概率涂红/蓝)
    ax.contourf(XX, YY, ZZ, levels=20, cmap="RdBu", alpha=0.6)
    # contour 在 0.5 等高线画一条黑色实线 = 决策边界
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
    # 固定随机种子, 保证两次运行的初始化、训练过程完全一致 (可复现)
    torch.manual_seed(0)
    np.random.seed(0)

    # =========================================================
    # 1. XOR 数据 (4 个点 + 4 个标签)
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
    # 2. 造两个网络: 唯一差别是激活函数
    #    用同一个 seed 初始化, 保证初始 W、b 完全相同
    # =========================================================
    torch.manual_seed(0)
    model_linear = TinyMLP(activation=nn.Identity())   # 恒等映射 = 无非线性
    torch.manual_seed(0)
    model_relu = TinyMLP(activation=nn.ReLU())         # 有非线性

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
    # 3. 可视化两个网络的"决策边界"
    #    左图: 一条直线 (线性激活只能这样)
    #    右图: 弯曲的两条线把 XOR 正确分开
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
