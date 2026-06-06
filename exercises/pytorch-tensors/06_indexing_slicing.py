"""
06. 索引 / 切片 (indexing / slicing)

类比 Python list / numpy 数组，但更强大。

核心写法：
    x[i]           # 第 i 个 (沿第 0 维)
    x[i, j]        # 二维选元素
    x[start:end]   # 切片 (不含 end)
    x[:, j]        # 全部行的第 j 列
    x[::2]         # 每隔一个取
    x[mask]        # 布尔索引: mask 是同形的 bool 张量
    x[[1,3,5]]     # fancy 索引: 用列表选若干行
    x[..., 0]      # ... 表示"前面所有维度都全取"

数学对应：
    选行/列 = 抽取子向量
    布尔索引 = 条件筛选
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import numpy as np
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 基础索引
# ------------------------------------------------------------
print("=" * 60)
print("1. 基础索引")
print("=" * 60)

x = torch.arange(20).reshape(4, 5)
print("x =")
print(x)

print(f"\nx[0]      = {x[0]}      (第 0 行)")
print(f"x[-1]     = {x[-1]}      (最后一行)")
print(f"x[1, 2]   = {x[1, 2]}    (第 1 行第 2 列)")
print(f"x[:, 0]   = {x[:, 0]}    (第 0 列)")
print(f"x[:, -1]  = {x[:, -1]}    (最后一列)")


# ------------------------------------------------------------
# 2. 切片
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 切片")
print("=" * 60)

print("x[1:3] (第 1, 2 行) =")
print(x[1:3])

print("\nx[:, 1:4] (第 1~3 列) =")
print(x[:, 1:4])

print("\nx[::2] (每隔一行) =")
print(x[::2])

print("\nx[:, ::2] (每隔一列) =")
print(x[:, ::2])


# ------------------------------------------------------------
# 3. 布尔索引（"过滤"）
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 布尔索引")
print("=" * 60)

mask = x > 10
print("mask (x > 10):")
print(mask)

print("\nx[mask] =", x[mask])
print("→ 把满足条件的元素拉成一维")

# 修改满足条件的位置
x_copy = x.clone()
x_copy[x_copy > 10] = -1
print("\n把 >10 的位置都置为 -1:")
print(x_copy)


# ------------------------------------------------------------
# 4. Fancy 索引
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. Fancy 索引（用列表选行/列）")
print("=" * 60)

print("x[[0, 2, 3]] (选第 0, 2, 3 行) =")
print(x[[0, 2, 3]])

print("\nx[[0, 1], [2, 3]] (选 (0,2) 和 (1,3) 两个点) =")
print(x[[0, 1], [2, 3]])


# ------------------------------------------------------------
# 5. ... (省略号) 在高维张量里
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 省略号 ...")
print("=" * 60)

img = torch.randn(2, 3, 4, 5)  # (B, C, H, W)
print(f"img shape = {tuple(img.shape)}")
print(f"img[..., 0].shape = {tuple(img[..., 0].shape)}  (取 W 维的第 0 列)")
print(f"img[0, ...].shape = {tuple(img[0, ...].shape)}  (取 batch 第 0 个样本)")
print(f"img[:, 0].shape   = {tuple(img[:, 0].shape)}   (取所有 batch 的第 0 通道)")


# ------------------------------------------------------------
# 6. 修改与索引（注意：切片是 view，会共享内存）
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. 注意：切片共享内存！")
print("=" * 60)

y = torch.arange(10)
sub = y[2:5]
sub[0] = 99
print(f"修改 sub 后，原张量 y 也变了: {y}")
print("→ 切片返回的是 view (不复制)。如要独立副本，用 .clone()")


# ------------------------------------------------------------
# 7. 可视化
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    x_vis = x.numpy().astype(float)

    def show(ax, data, highlight_mask, title):
        rgba = np.zeros((*data.shape, 4))
        rgba[..., 0] = 0.3  # R
        rgba[..., 1] = 0.6  # G
        rgba[..., 2] = 0.9  # B
        rgba[..., 3] = 0.3  # 默认低透明
        rgba[highlight_mask, 3] = 1.0  # 选中位置高亮
        ax.imshow(rgba, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{int(data[i,j])}", ha="center", va="center",
                        color="black" if not highlight_mask[i,j] else "white",
                        fontsize=10)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    # x[1, 2]
    mask1 = np.zeros_like(x_vis, dtype=bool)
    mask1[1, 2] = True
    show(axes[0], x_vis, mask1, "x[1, 2]")

    # x[:, 0]
    mask2 = np.zeros_like(x_vis, dtype=bool)
    mask2[:, 0] = True
    show(axes[1], x_vis, mask2, "x[:, 0]")

    # x[1:3]
    mask3 = np.zeros_like(x_vis, dtype=bool)
    mask3[1:3, :] = True
    show(axes[2], x_vis, mask3, "x[1:3]")

    # x[x > 10]
    mask4 = x_vis > 10
    show(axes[3], x_vis, mask4, "x[x > 10]")

    plt.suptitle("索引示意（高亮 = 选中位置）")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 给定一批 5 张图 (5, 3, 64, 64)，取出第 0 张图的红色通道
imgs = torch.randn(5, 3, 64, 64)
# TODO
red0 = imgs[0, 0]   # 等价于 imgs[0, 0, :, :]
assert red0.shape == (64, 64)
print("练习 1 ✅  shape =", tuple(red0.shape))

# 练习 2: 给定 logits (B=4, V=1000)，取每个样本的 top-1 词（最大值的索引）
logits = torch.randn(4, 1000)
top1 = logits.argmax(dim=-1)  # (4,)
assert top1.shape == (4,)
print(f"练习 2 ✅  top1 = {top1.tolist()}")

# 练习 3: 把张量里所有负数置零（ReLU 的手写实现）
t = torch.tensor([-1.0, 2.0, -3.0, 4.0])
# TODO
t_relu = t.clone()
t_relu[t_relu < 0] = 0
assert torch.equal(t_relu, torch.tensor([0., 2., 0., 4.]))
print("练习 3 ✅  ReLU(t) =", t_relu)
