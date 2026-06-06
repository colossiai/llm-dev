"""
12. 掩码 (mask) —— 因果注意力的关键

问题：
    在 LLM 训练时，模型一次"看到"整个序列做并行计算，
    但生成时只能看到"过去的 token"。
    如果训练时让 token t 看到 t+1, t+2, ... 就"作弊"了
    → 需要一个**因果掩码** (causal mask) 阻止它

技术做法：
    在 softmax 之前，把"未来位置"的注意力分数置为 -∞
    softmax(-∞) = 0 → 这些位置就被"屏蔽"

核心 API：
    torch.tril(x)         # 取下三角 (lower triangular)，上三角清零
    torch.triu(x)         # 上三角
    x.masked_fill(mask, value)   # mask 为 True 的位置填 value
    register_buffer(...)  # 非参数但跟模型一起保存/移动设备的张量

形状惯例：
    mask: (T, T) 或广播到 (1, 1, T, T)
    mask[i, j] = 1 表示"token i 可以看到 j"
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn.functional as F

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. torch.tril：下三角
# ------------------------------------------------------------
print("=" * 60)
print("1. torch.tril 取下三角")
print("=" * 60)

T = 5
all_ones = torch.ones(T, T)
mask = torch.tril(all_ones)
print(f"torch.tril(ones({T},{T})) =")
print(mask)
print("→ 1 的位置 = 允许看到；0 的位置 = 未来 (要屏蔽)")
print("  含义：第 i 行只有 j ≤ i 的位置是 1 (只能看过去 + 自己)")


# ------------------------------------------------------------
# 2. masked_fill：按 mask 填值
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. masked_fill：把 mask=True 的位置填值")
print("=" * 60)

scores = torch.randn(T, T) * 2
print("注意力分数 scores =")
print(scores)

# 我们想把"未来位置 (mask==0)"填成 -inf
print("\nscores.masked_fill(mask == 0, -inf) =")
masked = scores.masked_fill(mask == 0, float("-inf"))
print(masked)
print("→ 上三角都变成 -inf")


# ------------------------------------------------------------
# 3. softmax 后：-inf 变成 0 概率
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. softmax 之后的因果注意力矩阵")
print("=" * 60)

attn = F.softmax(masked, dim=-1)
print(attn)
print("\n每行的和:", attn.sum(dim=-1))
print("→ 上三角全是 0：token i 完全不会注意到 j > i 的位置")
print("→ 每行加起来仍为 1：注意力还是合法的概率分布")


# ------------------------------------------------------------
# 4. 在多头 attention 里：广播到 (B, H, T, T)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 多头 attention 中的掩码")
print("=" * 60)

B, H, T, d = 2, 4, 6, 8
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)

scores = Q @ K.transpose(-2, -1) / (d ** 0.5)   # (B, H, T, T)
print(f"scores shape = {tuple(scores.shape)}")

# mask 形状 (T, T) → 用 None 加两个前导维变 (1, 1, T, T)，广播到 (B, H, T, T)
mask = torch.tril(torch.ones(T, T))
mask = mask[None, None, :, :]
print(f"mask shape  = {tuple(mask.shape)}   (会广播到 {tuple(scores.shape)})")

scores = scores.masked_fill(mask == 0, float("-inf"))
attn = F.softmax(scores, dim=-1)
print(f"attn[0, 0] (batch 0, head 0):")
print(attn[0, 0])


# ------------------------------------------------------------
# 5. register_buffer：把 mask 注册成"非参数张量"
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. register_buffer 的用法")
print("=" * 60)

import torch.nn as nn

class CausalAttn(nn.Module):
    def __init__(self, T):
        super().__init__()
        # mask 不应该被训练 → 不能用 nn.Parameter
        # 但要跟模型一起 .to(device) / .save() → 用 register_buffer
        mask = torch.tril(torch.ones(T, T))
        self.register_buffer("mask", mask)

m = CausalAttn(T=4)
print(f"m.mask (像普通属性一样访问):\n{m.mask}")
print(f"m.mask 在 state_dict 里? {'mask' in m.state_dict()}")
print(f"m.mask 在 parameters() 里? {any(p is m.mask for p in m.parameters())}")
print("→ buffer 会被保存/搬到 GPU，但不会被 optimizer 当作可训练参数")


# ------------------------------------------------------------
# 6. 处理变长输入：T < block_size
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. 实战：T < 预分配的 block_size 时怎么裁剪 mask")
print("=" * 60)

block_size = 8           # 模型预分配的最大长度
full_mask = torch.tril(torch.ones(block_size, block_size))
full_mask = full_mask[None, None, :, :]  # (1, 1, block_size, block_size)
print(f"full_mask shape = {tuple(full_mask.shape)}")

T_actual = 5  # 当前 batch 实际序列长度
sub_mask = full_mask[:, :, :T_actual, :T_actual]
print(f"sub_mask shape  = {tuple(sub_mask.shape)}   ← 用切片截取")
print("→ 这是 minimal_edu/model.py 里 self.mask[:, :, :T, :T] 的来源")


# ------------------------------------------------------------
# 7. 可视化
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    T_v = 6
    mask_v = torch.tril(torch.ones(T_v, T_v))

    axes[0].imshow(mask_v.numpy(), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_title("torch.tril(ones)\n白=能看, 黑=被屏蔽")
    axes[0].set_xlabel("key 位置 j")
    axes[0].set_ylabel("query 位置 i")
    for i in range(T_v):
        for j in range(T_v):
            v = int(mask_v[i, j].item())
            axes[0].text(j, i, str(v), ha="center", va="center",
                        color="white" if v == 0 else "black", fontsize=9)

    scores_v = torch.randn(T_v, T_v)
    masked_v = scores_v.masked_fill(mask_v == 0, float("-inf"))
    attn_v = F.softmax(masked_v, dim=-1)
    axes[1].imshow(attn_v.numpy(), cmap="viridis")
    axes[1].set_title("softmax 后的注意力\n(每行和为 1, 上三角全 0)")
    axes[1].set_xlabel("key 位置 j")
    for i in range(T_v):
        for j in range(T_v):
            v = attn_v[i, j].item()
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.4 else "black", fontsize=8)

    # 右图：演示 "每个 token 看到的范围"
    axes[2].imshow(mask_v.numpy(), cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("含义：token i 能看到 j≤i")
    axes[2].set_xlabel("被看到的位置 j")
    axes[2].set_ylabel("当前 token i")
    for i in range(T_v):
        axes[2].text(i, i, "★", ha="center", va="center", color="red", fontsize=14)

    plt.suptitle("因果掩码 (Causal Mask)：阻止 token 看到未来")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 写出 T=4 的因果掩码
mask = torch.tril(torch.ones(4, 4))
expected = torch.tensor([
    [1., 0., 0., 0.],
    [1., 1., 0., 0.],
    [1., 1., 1., 0.],
    [1., 1., 1., 1.],
])
assert torch.equal(mask, expected)
print("练习 1 ✅  T=4 的因果 mask 正确")

# 练习 2: 给 scores (B=1, H=1, T=4, T=4) 加因果掩码后做 softmax，
#         验证每行第 i 个位置之后的概率都为 0
torch.manual_seed(0)
scores = torch.randn(1, 1, 4, 4)
mask = torch.tril(torch.ones(4, 4))[None, None]
attn = F.softmax(scores.masked_fill(mask == 0, float("-inf")), dim=-1)
for i in range(4):
    for j in range(i + 1, 4):
        assert attn[0, 0, i, j].item() == 0.0
print("练习 2 ✅  上三角注意力全为 0")

# 练习 3: 不用 masked_fill，直接给 scores 加上一个 (1 - mask) * -1e9 实现等价效果
# (有些早期实现这么做；现在更推荐 masked_fill + -inf 或 bool mask)
scores = torch.randn(4, 4)
mask = torch.tril(torch.ones(4, 4))
a = F.softmax(scores.masked_fill(mask == 0, float("-inf")), dim=-1)
b = F.softmax(scores + (1 - mask) * -1e9, dim=-1)
assert torch.allclose(a, b, atol=1e-6)
print("练习 3 ✅  两种掩码写法数值结果一致")
