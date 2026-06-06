"""
02. reshape / view —— 把同一组数据"换一种排列方式"

数学直觉：
- 数据本身没变，只是**怎么看**它变了
- 12 个数可以看成 (12,) / (3,4) / (4,3) / (2,2,3) / (1,12) ……
- 元素总数 (numel) 必须相同

reshape vs view：
- view  : 不复制内存，只换"读法"。要求张量在内存里是连续的
- reshape: 优先不复制；不行就自动复制一份再 view

`-1` 的妙用：让 PyTorch 自动算缺的那一维
  x.reshape(2, -1)  # 第二维自动算
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 同样 12 个数，多种 shape
# ------------------------------------------------------------
print("=" * 60)
print("1. 同一组数据，多种形状")
print("=" * 60)

x = torch.arange(12)  # [0,1,2,...,11]
print("原始:", x, "shape =", tuple(x.shape))

for new_shape in [(3, 4), (4, 3), (2, 6), (6, 2), (2, 2, 3), (12, 1), (1, 12)]:
    y = x.reshape(new_shape)
    print(f"\nreshape{new_shape}  →  shape={tuple(y.shape)}")
    print(y)


# ------------------------------------------------------------
# 2. -1 的用法：自动推断
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 使用 -1 让 PyTorch 自动算")
print("=" * 60)
print("x.reshape(3, -1)  →", tuple(x.reshape(3, -1).shape), "  (-1 被推断为 4)")
print("x.reshape(-1, 6) →", tuple(x.reshape(-1, 6).shape), "  (-1 被推断为 2)")
print("x.reshape(-1)    →", tuple(x.reshape(-1).shape), "  (展平成一维)")


# ------------------------------------------------------------
# 3. view 和 reshape 的区别
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. view vs reshape")
print("=" * 60)

a = torch.arange(12).reshape(3, 4)
print("原矩阵 a:")
print(a)
print("a.shape =", a.shape)
print("a.is_contiguous() =", a.is_contiguous())

# view 在连续张量上 OK
b = a.view(4, 3)
print("\na.view(4, 3) 成功:")
print(b)

# 非连续张量 (转置后) 不能直接 view
a_t = a.transpose(0, 1)  # 4×3，但内存非连续
print("\na.transpose(0,1).is_contiguous() =", a_t.is_contiguous())
try:
    a_t.view(12)
except RuntimeError as e:
    print("a_t.view(12) 报错 →", str(e).split("\n")[0])

# reshape 自动处理
c = a_t.reshape(12)
print("a_t.reshape(12) 自动成功 →", c)
print("  (内部其实复制了一份内存)")


# ------------------------------------------------------------
# 4. 可视化：同一组数据的不同排列
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    data = torch.arange(12).float()
    shapes = [(12,), (3, 4), (4, 3), (2, 2, 3)]

    for ax, sh in zip(axes, shapes):
        t = data.reshape(sh)
        if t.ndim == 1:
            ax.bar(range(12), t.numpy())
            ax.set_xticks(range(12))
        elif t.ndim == 2:
            ax.imshow(t.numpy(), cmap="viridis", aspect="auto")
            for i in range(sh[0]):
                for j in range(sh[1]):
                    ax.text(j, i, f"{int(t[i,j])}", ha="center", va="center", color="white")
        else:
            # 3D: 把两个 2×3 横向拼起来画
            merged = torch.cat([t[0], t[1]], dim=1)
            ax.imshow(merged.numpy(), cmap="viridis", aspect="auto")
            for i in range(2):
                for j in range(6):
                    ax.text(j, i, f"{int(merged[i,j])}", ha="center", va="center", color="white")
            ax.axvline(x=2.5, color="red", linestyle="--")
        ax.set_title(f"shape={sh}")

    plt.suptitle("同一组 12 个数字，4 种排列方式（数据没变）")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 5. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1：把 (32, 3, 28, 28) 的图像批展平成 (32, ?)
img_batch = torch.randn(32, 3, 28, 28)
# TODO: 展平每张图，保留 batch
flat = img_batch.reshape(32, -1)  # 参考答案
assert flat.shape == (32, 3 * 28 * 28) == (32, 2352)
print("练习 1 通过 ✅  flat.shape =", tuple(flat.shape))

# 练习 2：把 (24,) 一维向量变成 3 个 2×4 的矩阵
v = torch.arange(24)
# TODO
m = v.reshape(3, 2, 4)
assert m.shape == (3, 2, 4)
print("练习 2 通过 ✅  m.shape =", tuple(m.shape))

# 练习 3：什么时候 view 一定能用？
# 答：原张量内存连续时。一般刚 reshape / arange / randn 出来的张量都是连续的。
print("\n口诀：拿不准就用 reshape，它兼容所有情况。")
