"""
15. LayerNorm —— 让深层网络稳定训练的归一化

为什么需要？
- 深层网络里，越靠后的层输入分布可能漂移得很离谱
  (有的维度数值巨大，有的接近 0)
- 这会让梯度更新不稳定 → 训练崩
- LayerNorm 把每个样本的每一层激活"重新整形":
  减去均值、除以标准差，再用可学习的 γ, β 缩放和平移

数学公式：
    LN(x) = γ * (x - μ) / sqrt(σ^2 + ε) + β
    μ = x.mean(dim=last)
    σ^2 = x.var(dim=last, unbiased=False)
    γ, β: 可学习参数, shape = (d,)

关键：沿哪个维度归一化？
- LayerNorm: 沿"特征维 (最后一维)" → 每个样本/每个 token 独立归一
- BatchNorm: 沿"batch 维"       → 不同样本相互影响 (LLM 不用)
- RMSNorm:   不减均值，只除 RMS  (LLaMA 等用)

LLM 里出现 N+1 次（N 个 block，每个 2 次；外加最后一次 ln_f）
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn as nn

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 基础：手算 LayerNorm
# ------------------------------------------------------------
print("=" * 60)
print("1. 手算 LayerNorm")
print("=" * 60)

x = torch.tensor([1.0, 2.0, 3.0, 4.0])  # 一个样本，4 个特征
mean = x.mean()
var = x.var(unbiased=False)
eps = 1e-5
y = (x - mean) / (var + eps).sqrt()
print(f"x         = {x.tolist()}")
print(f"mean      = {mean.item():.4f}")
print(f"var       = {var.item():.4f}")
print(f"normalized= {y.tolist()}")
print(f"y.mean()  = {y.mean().item():.6f}   (≈ 0)")
print(f"y.var()   = {y.var(unbiased=False).item():.6f}   (≈ 1)")


# ------------------------------------------------------------
# 2. 用 nn.LayerNorm 验证
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 与 nn.LayerNorm 对比")
print("=" * 60)

ln = nn.LayerNorm(4, elementwise_affine=False)  # 先不要 γ/β 看纯归一化
y_torch = ln(x)
print(f"nn.LayerNorm 输出 = {y_torch.tolist()}")
print(f"手算结果一致: {torch.allclose(y, y_torch, atol=1e-5)}")


# ------------------------------------------------------------
# 3. γ, β —— 可学习的缩放和平移
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 可学习的 γ (weight) 和 β (bias)")
print("=" * 60)

ln = nn.LayerNorm(4)  # 默认 elementwise_affine=True
print(f"ln.weight (γ) 初始 = {ln.weight.tolist()}   (全 1)")
print(f"ln.bias   (β) 初始 = {ln.bias.tolist()}   (全 0)")
print("→ 初始时 LayerNorm 输出 ≈ 标准化后的 x，训练中 γ, β 可以再调整")


# ------------------------------------------------------------
# 4. LLM 里的形状：归一化沿最后一维
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 在 (B, T, C) 张量上的 LayerNorm")
print("=" * 60)

B, T, C = 2, 5, 8
x = torch.randn(B, T, C) * 5 + 10  # 故意做得分布很糟
ln = nn.LayerNorm(C)
y = ln(x)
print(f"输入 shape = {tuple(x.shape)}")
print(f"输入 mean = {x.mean().item():.4f}, std = {x.std().item():.4f}")
print(f"输出 shape = {tuple(y.shape)}   (形状不变)")

# 每个 (b, t) 位置独立归一化 → 每个 (b, t) 的 C 维向量 mean≈0, std≈1
print(f"\n每个 token 自己的 mean (前几个):")
print(y.mean(dim=-1)[:1])
print(f"每个 token 自己的 std  (前几个):")
print(y.std(dim=-1, unbiased=False)[:1])
print("→ 每个样本 / 每个 token 的特征维都被归一到 mean=0, std=1")


# ------------------------------------------------------------
# 5. 手写 LayerNorm
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 手写 MyLayerNorm")
print("=" * 60)

class MyLayerNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.weight * x_hat + self.bias

# 验证
torch.manual_seed(0)
x = torch.randn(2, 5, 8)
my = MyLayerNorm(8)
ref = nn.LayerNorm(8)
my.weight.data = ref.weight.data.clone()
my.bias.data = ref.bias.data.clone()
print("手写 vs nn.LayerNorm:", torch.allclose(my(x), ref(x), atol=1e-5))


# ------------------------------------------------------------
# 6. RMSNorm —— 现代 LLM (LLaMA) 用的简化版
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. RMSNorm（LLaMA 等用）")
print("=" * 60)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms

rms = RMSNorm(8)
y = rms(x)
print(f"RMSNorm 输入 shape = {tuple(x.shape)}, 输出 shape = {tuple(y.shape)}")
print("→ 不减均值，只除以 RMS (均方根)")
print("  少一个参数 (β)、少两次操作，效果接近 LayerNorm，更快")


# ------------------------------------------------------------
# 7. 可视化：归一化前后
# ------------------------------------------------------------
if plot:
    torch.manual_seed(0)
    x = torch.randn(8) * 4 + 10   # 一组 mean≈10, std≈4 的乱序数据
    ln = nn.LayerNorm(8, elementwise_affine=False)
    y = ln(x).detach()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(8), x.numpy(), color="steelblue")
    axes[0].axhline(x.mean().item(), color="red", linestyle="--",
                    label=f"mean={x.mean().item():.2f}")
    axes[0].set_title(f"输入: mean={x.mean().item():.2f}, std={x.std(unbiased=False).item():.2f}")
    axes[0].legend()

    axes[1].bar(range(8), y.numpy(), color="seagreen")
    axes[1].axhline(0, color="red", linestyle="--", label="mean=0")
    axes[1].set_title(f"LayerNorm 后: mean≈0, std≈1")
    axes[1].legend()

    plt.suptitle("LayerNorm: (x - mean) / std")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 验证 LayerNorm 后每行 mean≈0, std≈1
x = torch.randn(4, 16) * 10 + 5
y = nn.LayerNorm(16, elementwise_affine=False)(x)
assert torch.allclose(y.mean(dim=-1), torch.zeros(4), atol=1e-5)
assert torch.allclose(y.std(dim=-1, unbiased=False), torch.ones(4), atol=1e-5)
print("练习 1 ✅  每行 mean≈0, std≈1")

# 练习 2: 给定 (B, T, C), LayerNorm 应该传哪个 normalized_shape？
# 答: 最后一维 C (沿着每个 token 的特征维归一化)
ln = nn.LayerNorm(64)
x = torch.randn(2, 10, 64)
assert ln(x).shape == x.shape
print("练习 2 ✅  nn.LayerNorm(C) 接受 (B, T, C) 输出 (B, T, C)")

# 练习 3: 若把 γ 设为 2、β 设为 5，输出会怎样？
ln = nn.LayerNorm(4)
ln.weight.data.fill_(2.0)
ln.bias.data.fill_(5.0)
x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
y = ln(x)
# 应该等价于 2 * normalized + 5
expected = 2 * ((x - x.mean()) / x.var(unbiased=False).add(1e-5).sqrt()) + 5
assert torch.allclose(y, expected, atol=1e-4)
print(f"练习 3 ✅  γ=2, β=5 → 输出 = 2*x_hat + 5")
