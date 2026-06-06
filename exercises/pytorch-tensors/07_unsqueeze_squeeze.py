"""
07. unsqueeze / squeeze —— 加 / 去掉"长度为 1 的维度"

为什么需要？
- 做广播或 matmul 前，常常需要让形状对齐
- 一维向量 (n,) 和 (1, n)、(n, 1) 在数学上"差不多"，但在 PyTorch 里行为完全不同

API：
    x.unsqueeze(dim)  # 在第 dim 个位置插入一个 size=1 的维
    x.squeeze(dim)    # 去掉第 dim 个 size=1 的维 (如果不是 1 则不变)
    x.squeeze()       # 不指定 dim 时，去掉所有 size=1 的维

等价写法 (numpy 风格)：
    x[None]      # 在最前面 unsqueeze
    x[:, None]   # 在第 1 维 unsqueeze
    x[None, :]   # 同 x.unsqueeze(0)
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import numpy as np
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. unsqueeze 基础
# ------------------------------------------------------------
print("=" * 60)
print("1. unsqueeze: 加一个长度为 1 的维度")
print("=" * 60)

v = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"原 v shape = {tuple(v.shape)}  (一维向量)")
print(f"v.unsqueeze(0).shape = {tuple(v.unsqueeze(0).shape)}  (行向量 1×4)")
print(f"v.unsqueeze(1).shape = {tuple(v.unsqueeze(1).shape)}  (列向量 4×1)")
print(f"v[None].shape        = {tuple(v[None].shape)}        (同 unsqueeze(0))")
print(f"v[:, None].shape     = {tuple(v[:, None].shape)}     (同 unsqueeze(1))")


# ------------------------------------------------------------
# 2. squeeze 基础
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. squeeze: 去掉长度为 1 的维度")
print("=" * 60)

x = torch.zeros(1, 3, 1, 5)
print(f"x.shape = {tuple(x.shape)}")
print(f"x.squeeze().shape  = {tuple(x.squeeze().shape)}   (去掉所有 1)")
print(f"x.squeeze(0).shape = {tuple(x.squeeze(0).shape)}  (只去掉第 0 维的 1)")
print(f"x.squeeze(1).shape = {tuple(x.squeeze(1).shape)}  (第 1 维不是 1，无变化)")


# ------------------------------------------------------------
# 3. 实战：广播之前先 unsqueeze
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 实战：外积 (outer product)")
print("=" * 60)

a = torch.tensor([1., 2., 3.])  # (3,)
b = torch.tensor([10., 20., 30., 40.])  # (4,)

# 想得到 outer[i, j] = a[i] * b[j]，形状 (3, 4)
# 直接 a * b 会报错（shape 不兼容）
try:
    a * b
except RuntimeError as e:
    print("a * b 报错 →", str(e).split("\n")[0])

# 正确写法：把 a 变成列向量，b 变成行向量，让广播帮你
outer = a.unsqueeze(1) * b.unsqueeze(0)  # (3,1) * (1,4) → (3,4)
print(f"\na.unsqueeze(1) shape = {tuple(a.unsqueeze(1).shape)}")
print(f"b.unsqueeze(0) shape = {tuple(b.unsqueeze(0).shape)}")
print("a ⊗ b =")
print(outer)


# ------------------------------------------------------------
# 4. 实战：单个样本送入模型
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 实战：单样本预测 (模型要 batch 维)")
print("=" * 60)

# 模型期望输入 (B, features)，但我手上只有一个样本 (features,)
single = torch.randn(16)
print(f"单样本 shape = {tuple(single.shape)}")

batched = single.unsqueeze(0)  # 变成 (1, 16)
print(f"加上 batch 维 = {tuple(batched.shape)}")
print("→ 这样就能塞进期望 (B, 16) 输入的模型了")


# ------------------------------------------------------------
# 5. 实战：图像 (H, W) → (1, 1, H, W) 喂卷积
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 实战：单张灰度图喂给卷积")
print("=" * 60)

gray = torch.randn(28, 28)
print(f"灰度图 = {tuple(gray.shape)}")
ready = gray.unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
print(f"喂卷积前 = {tuple(ready.shape)}  (B=1, C=1, H=28, W=28)")


# ------------------------------------------------------------
# 6. 可视化
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    v_data = np.array([1, 2, 3, 4])

    # (4,) 一维 → bar
    axes[0].bar(range(4), v_data, color="steelblue")
    axes[0].set_title("v: shape (4,)\n一维向量")
    axes[0].set_xticks(range(4))

    # (1, 4) 行向量
    axes[1].imshow(v_data.reshape(1, 4), cmap="Blues", aspect="auto", vmin=0, vmax=5)
    axes[1].set_title("v.unsqueeze(0): (1, 4)\n行向量")
    for j in range(4):
        axes[1].text(j, 0, str(v_data[j]), ha="center", va="center", fontsize=14)
    axes[1].set_yticks([0])

    # (4, 1) 列向量
    axes[2].imshow(v_data.reshape(4, 1), cmap="Blues", aspect="auto", vmin=0, vmax=5)
    axes[2].set_title("v.unsqueeze(1): (4, 1)\n列向量")
    for i in range(4):
        axes[2].text(0, i, str(v_data[i]), ha="center", va="center", fontsize=14)
    axes[2].set_xticks([0])

    plt.suptitle("同样 4 个数，三种'形状身份'")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 给两个一维向量计算它们的"差异矩阵" D[i,j] = a[i] - b[j]
a = torch.tensor([1., 2., 3., 4.])
b = torch.tensor([10., 20., 30.])
# TODO
D = a.unsqueeze(1) - b.unsqueeze(0)
assert D.shape == (4, 3)
print("练习 1 ✅ shape =", tuple(D.shape))
print(D)

# 练习 2: 给一个 batch 的向量 (B=5, d=8)，扩成 (B=5, 1, d=8)
# 为了后面跟 (B=5, T=10, d=8) 做广播
x = torch.randn(5, 8)
# TODO
x_expanded = x.unsqueeze(1)
assert x_expanded.shape == (5, 1, 8)
print("练习 2 ✅ shape =", tuple(x_expanded.shape))

# 练习 3: 模型输出 (1, 1, 10)，想去掉两个 size=1 的维度，得到 (10,)
out = torch.randn(1, 1, 10)
# TODO
clean = out.squeeze()
assert clean.shape == (10,)
print("练习 3 ✅ shape =", tuple(clean.shape))
