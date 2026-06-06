"""
01. 张量形状（shape）

数学对应：
- 标量(0D)   → 一个数
- 向量(1D)   → 一列数  [x1, x2, ..., xn]
- 矩阵(2D)   → 行×列的表
- 3D 张量    → "一摞矩阵"（常用于一批样本：batch × rows × cols）
- 4D 张量    → 图像批：batch × channel × height × width

核心 API：
    x.shape    # 形状 (元组)
    x.ndim     # 维度数（"几维"）
    x.numel()  # 元素总数
    x.dtype    # 数据类型
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import sys

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"

# ------------------------------------------------------------
# 1. 创建不同维度的张量，观察 shape
# ------------------------------------------------------------
print("=" * 60)
print("1. 不同维度的张量")
print("=" * 60)

scalar = torch.tensor(7.0)
vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
mat = torch.tensor([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])
batch = torch.arange(24).reshape(2, 3, 4).float()  # 2 个 3×4 的矩阵

for name, t in [("scalar", scalar), ("vec", vec), ("mat", mat), ("batch", batch)]:
    print(f"{name:8s}  shape={tuple(t.shape)}  ndim={t.ndim}  numel={t.numel()}  dtype={t.dtype}")


# ------------------------------------------------------------
# 2. 形状的"读法"：从外到内
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 形状的读法 (从外到内)")
print("=" * 60)
print(f"batch.shape = {tuple(batch.shape)}")
print("  → 最外层 2 个 '块'")
print("  → 每块是 3 行")
print("  → 每行是 4 列")
print("batch[0] 是第一个 3×4 矩阵：")
print(batch[0])
print("batch[0][1] 是第一个矩阵的第 1 行（4 个数）：")
print(batch[0][1])


# ------------------------------------------------------------
# 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习 (检查 shape 理解)")
print("=" * 60)

# 练习 1：把一个 100 维的向量写成 PyTorch
# TODO: 创建一个形状为 (100,) 的全零向量
x1 = torch.zeros(100)  # 参考答案
assert x1.shape == (100,), f"期待 (100,)，实际 {tuple(x1.shape)}"
print("练习 1 通过 ✅  shape =", tuple(x1.shape))

# 练习 2：图像批 = (batch=8, channel=3, height=32, width=32)
# TODO: 创建上述形状的随机张量
x2 = torch.randn(8, 3, 32, 32)  # 参考答案
assert x2.shape == (8, 3, 32, 32)
assert x2.numel() == 8 * 3 * 32 * 32 == 24576
print("练习 2 通过 ✅  numel =", x2.numel())

# 练习 3：(4,) 和 (4, 1) 有什么区别？
v_flat = torch.tensor([1.0, 2.0, 3.0, 4.0])      # 一维向量
v_col = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 二维"列向量"
print(f"\n(4,)   ndim={v_flat.ndim}  → 数学上的 '向量'")
print(f"(4,1)  ndim={v_col.ndim}  → 数学上的 '4×1 矩阵 / 列向量'")
print("    虽然元素一样，但维度不同；做 matmul / broadcasting 时行为完全不同！")



# ------------------------------------------------------------
# 可视化
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 1D 向量 → 用 bar 图
    axes[0].bar(range(len(vec)), vec.numpy(), color="steelblue")
    axes[0].set_title(f"1D vec  shape={tuple(vec.shape)}")
    axes[0].set_xlabel("index")

    # 2D 矩阵 → heatmap
    im1 = axes[1].imshow(mat.numpy(), cmap="viridis")
    axes[1].set_title(f"2D mat  shape={tuple(mat.shape)}")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            axes[1].text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", color="white")
    plt.colorbar(im1, ax=axes[1])

    # 3D → 把两个矩阵并排画
    im2 = axes[2].imshow(torch.cat([batch[0], batch[1]], dim=1).numpy(), cmap="viridis")
    axes[2].set_title(f"3D batch  shape={tuple(batch.shape)}\n(把 2 个 3×4 矩阵并排画)")
    axes[2].axvline(x=3.5, color="red", linestyle="--", linewidth=2)
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

