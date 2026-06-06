"""
04. 批量矩阵乘法 (Batch matmul)

数学回顾：
- 矩阵乘法 (M×K) @ (K×N) = (M×N)
- 内维 K 必须匹配 (中间消去)；外维 M, N 是结果的形状

PyTorch 中的几种乘法：
    a @ b              # 等价于 torch.matmul(a, b)，最通用
    torch.matmul(a,b)  # 自动处理批量维度
    torch.mm(a, b)     # 只能 2D × 2D（不推荐）
    torch.bmm(a, b)    # 3D × 3D（明确批量）
    torch.dot(a, b)    # 一维向量点积

批量规则（matmul）:
    (B, M, K) @ (B, K, N) → (B, M, N)
        共享前缀维度 B，最后两维做矩阵乘
    (..., M, K) @ (..., K, N) → (..., M, N)
        前缀维度可广播

LLM 中处处都是批量 matmul：
    注意力 QK^T:  (B, H, T, d) @ (B, H, d, T) → (B, H, T, T)
    其中 B=batch, H=heads, T=tokens, d=head_dim
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 基本 matmul
# ------------------------------------------------------------
print("=" * 60)
print("1. 基本矩阵乘法")
print("=" * 60)

A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]])      # 3×2
B = torch.tensor([[1.0, 0.0, 2.0],
                  [0.0, 1.0, 3.0]])  # 2×3

C = A @ B
print("A (3×2) @ B (2×3) = C (3×3)")
print("C =")
print(C)

# 手算验证 C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*1 + 2*0 = 1
print(f"\n手算 C[0,0] = 1×1 + 2×0 = {C[0,0].item()}")


# ------------------------------------------------------------
# 2. 形状不匹配会报错
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 形状不匹配的报错")
print("=" * 60)
try:
    bad = A @ A  # 3×2 @ 3×2 → 内维不匹配 (2 != 3)
except RuntimeError as e:
    print("A @ A 报错 →", str(e).split("\n")[0])


# ------------------------------------------------------------
# 3. 批量 matmul
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 批量 matmul")
print("=" * 60)

# 一批 5 个 (3×4) 矩阵 乘 一批 5 个 (4×2) 矩阵
batch_A = torch.randn(5, 3, 4)
batch_B = torch.randn(5, 4, 2)
batch_C = batch_A @ batch_B
print(f"{tuple(batch_A.shape)} @ {tuple(batch_B.shape)} → {tuple(batch_C.shape)}")
print("→ 每个样本独立做一次 (3×4) @ (4×2) = (3×2)，再叠在一起")


# ------------------------------------------------------------
# 4. 线性层（神经网络的基本单元）
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 线性层：y = x @ W + b")
print("=" * 60)
# 一批 8 个样本，每个 16 维 → 想变成 4 维
x = torch.randn(8, 16)
W = torch.randn(16, 4)
b = torch.randn(4)

y = x @ W + b  # 注意 b 自动广播到 (8, 4)
print(f"x {tuple(x.shape)} @ W {tuple(W.shape)} + b {tuple(b.shape)} = y {tuple(y.shape)}")
print("一次性把一批样本通过一个线性变换 (你学过的'线性变换')")


# ------------------------------------------------------------
# 5. LLM 风格的 attention 形状演示
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Transformer 注意力的形状")
print("=" * 60)

bs, nh, nt, hd = 2, 4, 10, 8  # batch, heads, tokens, head_dim
Q = torch.randn(bs, nh, nt, hd)
K = torch.randn(bs, nh, nt, hd)
V = torch.randn(bs, nh, nt, hd)

scores = Q @ K.transpose(-2, -1)  # (B,H,T,d) @ (B,H,d,T) = (B,H,T,T)
print(f"Q {tuple(Q.shape)}")
print(f"K.transpose(-2,-1) {tuple(K.transpose(-2,-1).shape)}")
print(f"Q @ K^T → scores {tuple(scores.shape)}  (每对 token 之间的相似度矩阵)")

attn = torch.softmax(scores / (hd ** 0.5), dim=-1)
out = attn @ V
print(f"attn @ V → out {tuple(out.shape)}")


# ------------------------------------------------------------
# 6. 可视化
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    mats = [(A, "A (3×2)"), (B, "B (2×3)"), (C, "C = A @ B  (3×3)")]
    for ax, (m, title) in zip(axes, mats):
        im = ax.imshow(m.numpy(), cmap="viridis", aspect="auto")
        ax.set_title(title)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i,j]:.1f}", ha="center", va="center", color="white")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 求结果 shape
# (32, 100) @ (100, 64) = ?
# TODO: 写出答案 (元组)
ans1 = (32, 64)
res = torch.randn(32, 100) @ torch.randn(100, 64)
assert res.shape == ans1
print("练习 1 ✅  (32, 100) @ (100, 64) =", ans1)

# 练习 2: (B=4, T=10, d=64) @ (d=64, d2=32) = ?
# 提示: PyTorch 允许最后一维匹配，前面广播
ans2 = (4, 10, 32)
res = torch.randn(4, 10, 64) @ torch.randn(64, 32)
assert res.shape == ans2
print("练习 2 ✅  (4, 10, 64) @ (64, 32) =", ans2)

# 练习 3: 用 matmul 计算两个向量的点积
u = torch.tensor([1.0, 2.0, 3.0])
v = torch.tensor([4.0, 5.0, 6.0])
# 方法 a: torch.dot
dot_a = torch.dot(u, v)
# 方法 b: matmul (1D @ 1D 自动当成点积)
dot_b = u @ v
# 方法 c: 元素乘后求和
dot_c = (u * v).sum()
assert torch.allclose(dot_a, dot_b) and torch.allclose(dot_b, dot_c)
print(f"练习 3 ✅  u·v = {dot_a.item()}  (三种方法结果一致)")
