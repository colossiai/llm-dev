"""
06 - 完整 MLP 训练 (nn.Module + Adam + 真实数据集)

直觉:
  - 前 5 步都是手动写循环、手动算梯度
  - 实际工程中, 全部交给 PyTorch:
      nn.Module      封装网络结构
      nn.Linear      封装 W·x+b 这一层
      optim.Adam     封装"自适应学习率的梯度下降"
      loss.backward  封装反向传播
      opt.step       封装参数更新
  - 同样的事情, 代码缩短到几十行

数据集: sklearn.make_moons (两个月牙形, 非线性可分, 接近真实问题)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

import common


class MLP(nn.Module):
    """input(2) → 16 → 16 → 1, GELU 激活 (现代 LLM 同款)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def plot_boundary(ax, model, X_np, y_np, title):
    xs = np.linspace(X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5, 200)
    ys = np.linspace(X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5, 200)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)
    with torch.no_grad():
        ZZ = torch.sigmoid(model(grid)).numpy().reshape(XX.shape)
    ax.contourf(XX, YY, ZZ, levels=20, cmap="RdBu", alpha=0.6)
    ax.contour(XX, YY, ZZ, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c="blue", s=20, edgecolors="k", alpha=0.7, label="类 0")
    ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c="red", s=20, edgecolors="k", alpha=0.7, label="类 1")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    args = common.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    # =========================================================
    # 1. 数据 (两个月牙形)
    # =========================================================
    X_np, y_np = make_moons(n_samples=400, noise=0.2, random_state=42)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    print(f"数据集 make_moons: X.shape = {tuple(X.shape)}, y.shape = {tuple(y.shape)}")
    print(f"正负样本比例: {(y == 1).sum().item()} : {(y == 0).sum().item()}")

    # =========================================================
    # 2. 模型 + 损失 + 优化器
    # =========================================================
    model = MLP()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总量: {n_params}")

    # =========================================================
    # 3. 记录 3 个阶段的模型快照, 训练完用来画决策边界演化
    # =========================================================
    snapshots = {}
    epochs = 400

    losses = []
    accs = []

    print(f"\n{'epoch':>5} | {'loss':>10} | {'acc':>6}")
    print("-" * 30)
    for epoch in range(epochs + 1):
        # 保存快照 (训练前 / 中 / 后)
        if epoch in (0, epochs // 4, epochs):
            snapshots[epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        logits = model(X)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == y).float().mean().item()

        losses.append(loss.item())
        accs.append(acc)

        if epoch % 50 == 0 or epoch == epochs:
            print(f"{epoch:>5} | {loss.item():>10.4f} | {acc:>6.3f}")

    print(f"\n最终 accuracy = {accs[-1]:.3f}")

    # =========================================================
    # 4. 可视化: loss + acc + 3 阶段决策边界 (2×3 网格)
    # =========================================================
    if args.plot:
        fig = plt.figure(figsize=(15, 9))

        # 上排: 决策边界演化
        snap_keys = sorted(snapshots.keys())
        for i, ep in enumerate(snap_keys):
            ax = fig.add_subplot(2, 3, i + 1)
            model.load_state_dict(snapshots[ep])
            plot_boundary(ax, model, X_np, y_np,
                          f"epoch={ep}  acc={accs[ep]:.3f}")

        # 还原到最终模型
        model.load_state_dict(snapshots[epochs])

        # 下排左: loss 曲线
        ax_loss = fig.add_subplot(2, 3, 4)
        ax_loss.plot(losses, color="tab:blue")
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("BCE loss")
        ax_loss.set_title("Loss 曲线")
        ax_loss.set_yscale("log")
        ax_loss.grid(True, alpha=0.3)

        # 下排中: accuracy 曲线
        ax_acc = fig.add_subplot(2, 3, 5)
        ax_acc.plot(accs, color="tab:green")
        ax_acc.set_xlabel("epoch")
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_title("Accuracy 曲线")
        ax_acc.set_ylim(0.4, 1.05)
        ax_acc.grid(True, alpha=0.3)

        # 下排右: 文字总结
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
