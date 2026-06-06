"""
06 - 完整 MLP 训练 (nn.Module + Adam + 真实数据集)

================ 给零基础读者的 5 分钟讲解 ================

【这是整个系列的"集大成"脚本】
  前面 5 步:
    01 单个神经元 → 02 激活函数 → 03 非线性的必要性
    04 手写反向传播 → 05 PyTorch autograd 验证
  全部加起来就是这里写的"现代神经网络训练范式"。
  本脚本展示工程中真实代码长什么样: 简洁、模块化、可读。

【三个新东西】
  1. 数据集换成 make_moons (sklearn 提供的两个月牙形)
     比 XOR 复杂得多, 接近真实问题: 400 个点, 有噪声, 非线性可分。

  2. 用 nn.Sequential 串联多层
     之前 03 用 self.fc1, self.fc2, self.fc3 分别定义。
     Sequential 是更紧凑的写法 — 像"流水线"一样把层串起来, 数据按顺序流过。

  3. 训练过程中保存 3 张快照
     这样可以画出"决策边界如何随训练逐渐弯曲" — 直观看到模型在学习什么。

【训练范式的"四步循环"】 (现代 PyTorch 训练代码都长这样)
    opt.zero_grad()       # 1. 清梯度
    logits = model(X)     # 2. 前向 → 算 loss
    loss = loss_fn(...)
    loss.backward()       # 3. 反向 → 自动算梯度
    opt.step()            # 4. 更新参数

【Adam 优化器是什么?】
  最朴素的优化器是 SGD: W -= lr * grad
  Adam 在 SGD 基础上增加了:
    - 动量 (momentum): 像滚雪球一样, 让更新方向变化更平滑
    - 自适应学习率: 每个参数自动用合适的步长
  → 收敛更快、更稳, 现代深度学习的默认选择。

【为什么 BCEWithLogitsLoss 不需要单独 sigmoid?】
  它内部做了 sigmoid + 交叉熵的"融合公式", 数值上更稳定。
  所以 model 输出的是 logit (任意实数, 没经过 sigmoid),
  到推理/画图时才用 torch.sigmoid() 转成概率 (0~1)。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

import common


class MLP(nn.Module):
    """
    一个 4 层的 MLP: 输入 2 维 → 隐藏 16 → 隐藏 16 → 输出 1 维
    激活用 GELU (现代 LLM 同款, 比 ReLU 平滑)。
    """
    def __init__(self):
        super().__init__()
        # nn.Sequential = 把多层"串"在一起的容器
        # 数据流过的顺序就是这里列出的顺序
        # 注意: 最后一层 Linear 没加激活, 输出 logit (任意实数)
        self.net = nn.Sequential(
            nn.Linear(2, 16),   # 2 → 16
            nn.GELU(),          # 激活
            nn.Linear(16, 16),  # 16 → 16
            nn.GELU(),          # 激活
            nn.Linear(16, 1),   # 16 → 1 (输出 logit, 不加激活)
        )

    def forward(self, x):
        # squeeze(-1): 把 (N, 1) 形状压成 (N,), 方便和标签 y (shape (N,)) 算 loss
        return self.net(x).squeeze(-1)


def plot_boundary(ax, model, X_np, y_np, title):
    """
    画"决策边界": 在数据范围内每个点都让模型预测一下,
    把高概率区涂红、低概率区涂蓝, 0.5 等高线画黑实线 (= 边界)。
    """
    # 在数据范围 + 边距内撒一个 200×200 的网格
    xs = np.linspace(X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5, 200)
    ys = np.linspace(X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)
    # 推理时关掉梯度追踪 (省内存、更快)
    with torch.no_grad():
        # 模型输出是 logit, 用 sigmoid 转成概率
        ZZ = torch.sigmoid(model(grid)).numpy().reshape(XX.shape)
    ax.contourf(XX, YY, ZZ, levels=20, cmap="RdBu", alpha=0.6)   # 填色等高线
    ax.contour(XX, YY, ZZ, levels=[0.5], colors="k", linewidths=2)  # 决策边界
    ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c="blue", s=20, edgecolors="k", alpha=0.7, label="类 0")
    ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c="red", s=20, edgecolors="k", alpha=0.7, label="类 1")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    args = common.parse_args()
    # 固定随机种子, 保证每次运行结果一样
    torch.manual_seed(42)
    np.random.seed(42)

    # =========================================================
    # 1. 数据: 两个交错的月牙形 (sklearn 提供, 经典非线性可分数据集)
    # =========================================================
    # n_samples=400 → 总共 400 个点 (200 个类 0 + 200 个类 1)
    # noise=0.2     → 加点高斯噪声让数据更接近真实
    X_np, y_np = make_moons(n_samples=400, noise=0.2, random_state=42)
    # 转成 PyTorch tensor (PyTorch 训练只接受 tensor, 不接受 numpy)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    print(f"数据集 make_moons: X.shape = {tuple(X.shape)}, y.shape = {tuple(y.shape)}")
    print(f"正负样本比例: {(y == 1).sum().item()} : {(y == 0).sum().item()}")

    # =========================================================
    # 2. 模型 + 损失 + 优化器 (现代训练的"三件套")
    # =========================================================
    model = MLP()
    loss_fn = nn.BCEWithLogitsLoss()             # 二分类损失 (sigmoid + 交叉熵, 数值稳)
    # Adam: 自适应学习率的梯度下降, 比朴素 SGD 收敛快
    # model.parameters() 自动收集 nn.Module 里所有需要训练的参数 (W, b)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)

    # 看看模型有多少参数 (这里大约 337 个 — MLP 很小)
    # 对比: GPT-3 有 1750 亿个参数
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {n_params}")

    # =========================================================
    # 3. 训练循环 + 在 3 个时间点保存"模型快照"
    #    snapshot 之后用来画"决策边界如何随训练演化"
    # =========================================================
    snapshots = {}    # {epoch: state_dict 的副本}
    epochs = 400

    losses = []   # 每轮的 loss
    accs = []     # 每轮的 accuracy

    print(f"\n{'epoch':>5} | {'loss':>10} | {'acc':>6}")
    print("-" * 30)
    for epoch in range(epochs + 1):
        # 在 epoch 0 (训练前) / epoch 100 (训练中) / epoch 400 (训练后) 各存一份
        # state_dict() = 模型当前所有参数的字典 {层名: 张量}
        # .detach().clone() 是为了拿独立的副本, 不被后续训练覆盖
        if epoch in (0, epochs // 4, epochs):
            snapshots[epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # ===== 训练四步循环 =====
        logits = model(X)              # 前向: 算 logit
        loss = loss_fn(logits, y)      # 算 loss
        opt.zero_grad()                # 清梯度
        loss.backward()                # 反向: 自动算梯度
        opt.step()                     # 用 Adam 公式更新所有参数

        # 顺便算 accuracy (推理, 不算梯度)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if epoch % 50 == 0 or epoch == epochs:
            print(f"{epoch:>5} | {loss.item():>10.4f} | {acc:>6.3f}")

    print(f"\n最终 accuracy = {accs[-1]:.3f}")

    # =========================================================
    # 4. 可视化: 2×3 网格
    #    上排: epoch 0 / 100 / 400 的决策边界 — 看模型怎么逐步学会月牙形
    #    下排: loss 曲线 + accuracy 曲线 + 实验配置文字总结
    # =========================================================
    if args.plot:
        fig = plt.figure(figsize=(15, 9))

        # ===== 上排 3 图: 3 个阶段的决策边界 =====
        snap_keys = sorted(snapshots.keys())   # [0, 100, 400]
        for i, ep in enumerate(snap_keys):
            ax = fig.add_subplot(2, 3, i + 1)
            # load_state_dict 把模型参数还原到那个 epoch 的快照
            model.load_state_dict(snapshots[ep])
            plot_boundary(ax, model, X_np, y_np,
                          f"epoch={ep}  acc={accs[ep]:.3f}")

        # 还原到最终模型 (画完别留在旧快照)
        model.load_state_dict(snapshots[epochs])

        # ===== 下排左: loss 曲线 (对数刻度, 看下降趋势更清楚) =====
        ax_loss = fig.add_subplot(2, 3, 4)
        ax_loss.plot(losses, color="tab:blue")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("BCE loss")
        ax_loss.set_title("Loss 曲线")
        ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.3)

        # ===== 下排中: accuracy 曲线 =====
        ax_acc = fig.add_subplot(2, 3, 5)
        ax_acc.plot(accs, color="tab:green")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_title("Accuracy 曲线")
        ax_acc.set_ylim(0.4, 1.05)
        ax_acc.grid(True, alpha=0.3)

        # ===== 下排右: 实验配置文字总结 =====
        ax_txt = fig.add_subplot(2, 3, 6)
        ax_txt.axis("off")
        summary = (
            f"数据集: make_moons\n"
            f"样本数: {len(X)}\n"
            f"网络: 2 → 16 → 16 → 1 (GELU)\n"
            f"参数: {n_params}\n"
            f"优化器: Adam (lr=0.02)\n"
            f"损失: BCEWithLogitsLoss\n"
            f"epochs: {epochs}\n\n"
            f"最终 loss: {losses[-1]:.4f}\n"
            f"最终 acc:  {accs[-1]:.3f}\n\n"
            f"上排 3 图: 决策边界\n"
            f"在训练中如何从直线 →\n"
            f"弯成月牙形。"
        )
        ax_txt.text(0.05, 0.95, summary, transform=ax_txt.transAxes,
                    fontsize=11, verticalalignment="top")

        plt.suptitle("完整 MLP 训练 (nn.Module + Adam + make_moons)", fontsize=14, y=1.00)
        plt.tight_layout()
        path = common.save_fig("06_mlp_complete", bbox_inches="tight")
        print(f"\n图已保存到 {path}")
    else:
        print("\n(未画图。加 --plot 生成 plots/06_mlp_complete.png)")


if __name__ == "__main__":
    main()
