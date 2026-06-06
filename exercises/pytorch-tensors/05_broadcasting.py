"""
05. 广播 (Broadcasting) —— 自动补齐维度

什么是广播？
    当两个张量形状不同时，PyTorch 会"假装"复制小的那个，让它跟大的对齐，
    然后再做逐元素运算。整个过程不真的占用内存（只是改读法）。

广播规则（从最后一维往前对齐）：
    1. 维度长度相等 ✅
    2. 其中一个长度为 1 → 沿这个维度复制 ✅
    3. 缺维度 → 看作长度 1 ✅
    其他情况报错 ❌

最常见的例子：
    matrix + bias_vector
    (32, 4) + (4,) → bias 被复制 32 份，每行都加上 bias
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import numpy as np
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 标量 + 张量
# ------------------------------------------------------------
print("=" * 60)
print("1. 标量 + 张量")
print("=" * 60)

x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print("x =")
print(x)
print("x + 10 =")
print(x + 10)
print("→ 10 被广播到每个位置")


# ------------------------------------------------------------
# 2. 行向量 + 矩阵
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 矩阵 + 行向量")
print("=" * 60)

m = torch.zeros(3, 4)
row = torch.tensor([10., 20., 30., 40.])
print(f"m {tuple(m.shape)} + row {tuple(row.shape)}:")
result = m + row
print(result)
print("→ row 被复制 3 行")


# ------------------------------------------------------------
# 3. 列向量 + 矩阵
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 矩阵 + 列向量")
print("=" * 60)

col = torch.tensor([[100.], [200.], [300.]])  # (3, 1)
print('col =')
print(col)
print(f"m {tuple(m.shape)} + col {tuple(col.shape)}:")
result = m + col
print(result)
print("→ col 被复制 4 列")


# ------------------------------------------------------------
# 4. 行 + 列 → 外积式扩张
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 行向量 + 列向量 → 二维矩阵 (外加法)")
print("=" * 60)

a = torch.tensor([1., 2., 3., 4.])         # (4,)
b = torch.tensor([[10.], [20.], [30.]])    # (3, 1)
print(f"a {tuple(a.shape)} + b {tuple(b.shape)} → 输出 {tuple((a + b).shape)}")
print('a = ')
print(a)
print('b = ')
print(b)
print(a + b)
print("→ 把 a 复制 3 行、把 b 复制 4 列，得到 (3, 4)")


# ------------------------------------------------------------
# 5. 不能广播的例子
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 不能广播 → 报错")
print("=" * 60)
try:
    torch.zeros(3, 4) + torch.zeros(3, 5)  # 最后一维 4 vs 5 不匹配也不为 1
except RuntimeError as e:
    print("(3,4) + (3,5) 报错 →", str(e).split("\n")[0])


# ------------------------------------------------------------
# 6. 实战：减去均值、除以标准差（标准化）
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. 实战：对一批样本做标准化")
print("=" * 60)

# 假设 5 个样本，每个 3 个特征
data = torch.tensor([[1.0, 100.0, 0.01],
                     [2.0, 200.0, 0.02],
                     [3.0, 300.0, 0.03],
                     [4.0, 400.0, 0.04],
                     [5.0, 500.0, 0.05]])
print("原始 data (5×3):")
print(data)

mean = data.mean(dim=0)  # 沿 batch 维平均，得 (3,)
std = data.std(dim=0)    # (3,)
print(f"\nmean shape={tuple(mean.shape)} = {mean}")
print(f"std  shape={tuple(std.shape)} = {std}")

normalized = (data - mean) / std  # 广播：(5,3) - (3,) → (5,3)
print("\n标准化后:")
print(normalized)
print(f"每列均值 ≈ 0: {normalized.mean(dim=0)}")
print(f"每列标准差 ≈ 1: {normalized.std(dim=0)}")


# ------------------------------------------------------------
# 7. 可视化广播
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 左：(3, 1) col 向量
    col_data = np.array([[1.], [2.], [3.]])
    axes[0].imshow(col_data, cmap="Blues", aspect="auto", vmin=0, vmax=5)
    axes[0].set_title("col: shape (3, 1)")
    for i in range(3):
        axes[0].text(0, i, f"{int(col_data[i,0])}", ha="center", va="center", color="black", fontsize=14)

    # 中：(1, 4) row 向量
    row_data = np.array([[10., 20., 30., 40.]])
    axes[1].imshow(row_data, cmap="Reds", aspect="auto", vmin=0, vmax=50)
    axes[1].set_title("row: shape (1, 4)")
    for j in range(4):
        axes[1].text(j, 0, f"{int(row_data[0,j])}", ha="center", va="center", color="black", fontsize=14)

    # 右：col + row → (3, 4)
    sum_data = col_data + row_data
    axes[2].imshow(sum_data, cmap="viridis", aspect="auto")
    axes[2].set_title("col + row → shape (3, 4)")
    for i in range(3):
        for j in range(4):
            axes[2].text(j, i, f"{int(sum_data[i,j])}", ha="center", va="center", color="white", fontsize=12)

    plt.suptitle("广播：(3,1) + (1,4) → (3,4)")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 给定 5 张 RGB 图 (5, 3, 64, 64)，给每个通道加一个偏置
# bias shape 应该是？
imgs = torch.randn(5, 3, 64, 64)
# TODO: 让 bias 能广播到每张图、每个通道
bias = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1)
result = imgs + bias
assert result.shape == (5, 3, 64, 64)
print("练习 1 ✅  bias shape =", tuple(bias.shape))

# 练习 2: 用广播算两点之间的距离矩阵
# 给定 P (5, 2) 5 个二维点
P = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 2.]])
# 用广播：P[:, None, :] - P[None, :, :]  → (5, 5, 2)
diff = P[:, None, :] - P[None, :, :]
dist = (diff ** 2).sum(dim=-1).sqrt()
print("练习 2 ✅  距离矩阵 (5×5):")
print(dist)
print("  (对角线为 0，对称矩阵)")

# 练习 3: 形状能否广播？
# (3, 1, 4) 和 (2, 4) → 能 / 不能？
# 提示：从右往左对齐：4↔4 ✅，1↔2 ✅(广播)，3 ↔ 空 ✅(补 1)
# 结果 shape = (3, 2, 4)
a = torch.zeros(3, 1, 4)
b = torch.zeros(2, 4)
assert (a + b).shape == (3, 2, 4)
print("练习 3 ✅  (3,1,4) + (2,4) → (3,2,4)")
