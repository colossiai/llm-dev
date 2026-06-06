"""
03. transpose / permute —— 维度交换

数学对应：
- 矩阵转置 A^T：行变列、列变行
- 转置后的形状: (m, n) → (n, m)
- 重要恒等式: (A @ B)^T = B^T @ A^T

API：
    a.T              # 2D 张量专用的转置
    a.transpose(i, j)  # 交换 第 i 维 和 第 j 维
    a.permute(d0, d1, ...) # 完全重排所有维度（更强大）

reshape vs transpose 区别（**重要**）：
- reshape: 数据"重新排列"——保持读取顺序，只换分块方式
- transpose: 数据"重新映射"——同一个元素的"坐标"变了
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 2D 矩阵转置
# ------------------------------------------------------------
print("=" * 60)
print("1. 二维矩阵转置")
print("=" * 60)

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
print("原矩阵 a (2×3):")
print(a)
print("\na.T (3×2):")
print(a.T)
print("\na.transpose(0, 1) (等价):")
print(a.transpose(0, 1))


# ------------------------------------------------------------
# 2. 对比：reshape vs transpose
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. reshape(3,2) ≠ transpose(0,1)")
print("=" * 60)

print("a.reshape(3, 2)：按顺序填进新形状")
print(a.reshape(3, 2))
print("→ 元素读取顺序：1,2,3,4,5,6 → 重新分成 3 行 2 列")

print("\na.transpose(0, 1)：行列互换")
print(a.transpose(0, 1))
print("→ a[i,j] 现在变成 result[j,i]")


# ------------------------------------------------------------
# 3. 高维 permute
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. permute (高维)")
print("=" * 60)

# 假设图像批: (batch=2, channel=3, H=4, W=5)
img = torch.randn(2, 3, 4, 5)
print("原 shape (B, C, H, W):", tuple(img.shape))

# 常见操作: 转成 (B, H, W, C) 给 numpy/matplotlib 显示
img_hwc = img.permute(0, 2, 3, 1)
print("permute(0,2,3,1) → (B, H, W, C):", tuple(img_hwc.shape))


# ------------------------------------------------------------
# 4. 验证转置的数学性质 (A @ B)^T = B^T @ A^T
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 验证: (A @ B)^T = B^T @ A^T")
print("=" * 60)

A = torch.randn(3, 4)
B = torch.randn(4, 5)
lhs = (A @ B).T
rhs = B.T @ A.T
print("(A @ B)^T 与 B^T @ A^T 是否一致：", torch.allclose(lhs, rhs))


# ------------------------------------------------------------
# 5. 可视化转置
# ------------------------------------------------------------
if plot:
    m = torch.arange(12).reshape(3, 4).float()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].imshow(m.numpy(), cmap="viridis", aspect="auto")
    axes[0].set_title(f"原矩阵  shape={tuple(m.shape)}")
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, f"{int(m[i,j])}", ha="center", va="center", color="white")

    axes[1].imshow(m.T.numpy(), cmap="viridis", aspect="auto")
    axes[1].set_title(f"转置后  shape={tuple(m.T.shape)}")
    for i in range(4):
        for j in range(3):
            axes[1].text(j, i, f"{int(m.T[i,j])}", ha="center", va="center", color="white")

    plt.suptitle("转置：a[i,j] → a[j,i]  (颜色相同 = 同一个元素)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 6. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 把 (B=4, C=3, H=64, W=64) 转成 matplotlib 需要的 (B, H, W, C)
imgs = torch.randn(4, 3, 64, 64)
# TODO
imgs_hwc = imgs.permute(0, 2, 3, 1)
assert imgs_hwc.shape == (4, 64, 64, 3)
print("练习 1 通过 ✅  shape =", tuple(imgs_hwc.shape))

# 练习 2: 给定 (seq_len=10, batch=4, dim=8)，
# 转成 (batch=4, seq_len=10, dim=8) —— PyTorch 里许多 RNN 输出长这样
x = torch.randn(10, 4, 8)
# TODO
x_bsd = x.transpose(0, 1)  # 只交换前两维
assert x_bsd.shape == (4, 10, 8)
print("练习 2 通过 ✅  shape =", tuple(x_bsd.shape))

# 练习 3: 给定 4×3 矩阵 M，用两种方式得到点积 M[0] · M[1]
M = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [1.0, 0.0, 1.0]])
dot1 = torch.dot(M[0], M[1])
dot2 = (M[0] * M[1]).sum()
assert torch.allclose(dot1, dot2)
print(f"练习 3 通过 ✅  M[0]·M[1] = {dot1.item()}")
