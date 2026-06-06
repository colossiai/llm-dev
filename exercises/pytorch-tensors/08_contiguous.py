"""
08. contiguous —— 内存连续性

为什么这个概念存在？
- 张量在内存里其实是**一维连续数组**
- 形状 / 步长 (stride) 只是"读法说明"
- 有些操作（transpose、permute）只改"读法"，不真的搬数据
  → 导致内存里的顺序跟 shape 看起来的顺序不一致 → 不连续
- view 要求张量连续；不连续就报错
- 解决：调 .contiguous() 强制把数据真的搬一份

关键 API：
    x.is_contiguous()  # 是否连续
    x.stride()         # 每个维度的步长 (按元素数)
    x.contiguous()     # 强制变连续 (会复制)

口诀：
    transpose / permute → 大概率非连续
    view 报错 → 加 .contiguous() 再 view，或直接用 reshape
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import numpy as np
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. stride 是什么？
# ------------------------------------------------------------
print("=" * 60)
print("1. stride: 每跳到下一行/列要走几个元素")
print("=" * 60)

x = torch.arange(12).reshape(3, 4)
print("x =")
print(x)
print(f"x.shape  = {tuple(x.shape)}")
print(f"x.stride = {x.stride()}")
print("→ 跳一行走 4 个元素（一整行长度），跳一列走 1 个元素")
print(f"x.is_contiguous() = {x.is_contiguous()}")


# ------------------------------------------------------------
# 2. transpose 之后变得不连续
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. transpose 之后")
print("=" * 60)

xt = x.transpose(0, 1)  # 4×3
print("xt =")
print(xt)
print(f"xt.shape  = {tuple(xt.shape)}")
print(f"xt.stride = {xt.stride()}")
print(f"xt.is_contiguous() = {xt.is_contiguous()}")
print("→ stride 变成 (1, 4)：要跳到下一行只走 1 个元素，跳到下一列要走 4 个")
print("  这意味着按 xt 的形状顺序读，元素在内存里不再是连续的")


# ------------------------------------------------------------
# 3. view 在不连续张量上会报错
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. view 报错的真正原因")
print("=" * 60)

try:
    xt.view(12)
except RuntimeError as e:
    msg = str(e).split("\n")[0]
    print(f"xt.view(12) → {msg}")
    print("  报错本质：内存不连续 → 没法直接换'读法'")


# ------------------------------------------------------------
# 4. 三种解决方法
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 三种解决方法")
print("=" * 60)

# 方法 1: 显式 .contiguous() 再 view
xt_c = xt.contiguous()
print(f"方法 1: xt.contiguous().is_contiguous() = {xt_c.is_contiguous()}")
print(f"        xt.contiguous().view(12) = {xt_c.view(12)}")

# 方法 2: 用 reshape（自动处理）
print(f"\n方法 2: xt.reshape(12) = {xt.reshape(12)}  (内部自动复制)")

# 方法 3: 用 flatten
print(f"方法 3: xt.flatten() = {xt.flatten()}")


# ------------------------------------------------------------
# 5. 内存视角图示
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 内存里的真实顺序")
print("=" * 60)

print("原 x 的内存顺序 (连续): 0,1,2,3,4,5,6,7,8,9,10,11")
print("xt 形状是 (4,3)，但内存里还是同一份: 0,1,2,3,4,5,6,7,8,9,10,11")
print("按 xt 的 (i,j) 读，等于读 x[j,i] → 跳着读")
print()
print("xt 看到的内容:")
print(xt)
print("xt 按行展平 = 想得到 [0,4,8,1,5,9,2,6,10,3,7,11]")
print("但内存里其实是 [0,1,2,3,4,5,6,7,8,9,10,11]")
print("→ 想得到前者就必须真复制一份 → 这就是 .contiguous() 做的事")


# ------------------------------------------------------------
# 6. 可视化
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 左: 原矩阵
    axes[0].imshow(x.numpy(), cmap="viridis", aspect="auto")
    axes[0].set_title(f"x (3×4)\nstride={x.stride()}\ncontiguous={x.is_contiguous()}")
    for i in range(3):
        for j in range(4):
            axes[0].text(j, i, f"{int(x[i,j])}", ha="center", va="center", color="white")

    # 中: 转置后
    axes[1].imshow(xt.numpy(), cmap="viridis", aspect="auto")
    axes[1].set_title(f"xt = x.T (4×3)\nstride={xt.stride()}\ncontiguous={xt.is_contiguous()}")
    for i in range(4):
        for j in range(3):
            axes[1].text(j, i, f"{int(xt[i,j])}", ha="center", va="center", color="white")

    # 右: 内存里的真实顺序 (一维)
    mem = x.flatten().numpy()
    axes[2].imshow(mem.reshape(1, -1), cmap="viridis", aspect="auto")
    axes[2].set_title("内存中的实际顺序（一维）\nx 和 xt 共享同一份")
    for j, val in enumerate(mem):
        axes[2].text(j, 0, f"{int(val)}", ha="center", va="center", color="white", fontsize=9)
    axes[2].set_yticks([])

    plt.suptitle("transpose 只改'读法'，不动内存")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 7. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 判断下面这些操作后是否连续
y = torch.arange(24).reshape(2, 3, 4)
print(f"y                       contiguous? {y.is_contiguous()}")  # True
print(f"y.transpose(0, 1)       contiguous? {y.transpose(0,1).is_contiguous()}")  # False
print(f"y.permute(2, 0, 1)      contiguous? {y.permute(2,0,1).is_contiguous()}")  # False
print(f"y.reshape(4, 6)         contiguous? {y.reshape(4,6).is_contiguous()}")    # True
print(f"y.flatten()             contiguous? {y.flatten().is_contiguous()}")       # True

# 练习 2: 修复一个报错代码
z = torch.arange(12).reshape(3, 4).transpose(0, 1)
try:
    bad = z.view(-1)
except RuntimeError:
    # TODO: 修复 (两种写法都对)
    good_a = z.contiguous().view(-1)
    good_b = z.reshape(-1)
    assert torch.equal(good_a, good_b)
    print("\n练习 2 ✅ 两种修复方法结果一致:")
    print("  方法 a: z.contiguous().view(-1) →", good_a)
    print("  方法 b: z.reshape(-1)           →", good_b)


# ------------------------------------------------------------
# 总结
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("一图流总结")
print("=" * 60)
print("""
    需要换形状
        │
        ├── 张量是 transpose/permute 来的？
        │       ├── 是 → 用 .reshape() 或先 .contiguous().view()
        │       └── 否 → .view() 或 .reshape() 都行
        │
        └── 拿不准 → 一律用 .reshape()，最安全
""")
