"""
11. Embedding —— "查表" = "one-hot @ 权重矩阵"

数学等价：
    one-hot(id) @ W  ==  W[id]
    (V,)      (V,d)    (d,)

Embedding 本质上就是一个**形状为 (V, d) 的查找表**：
- V: 词表大小 (vocab_size)
- d: 嵌入维度 (n_embd)
- 输入 token id (整数) → 输出 d 维向量

为什么不直接用 one-hot @ Linear？
- one-hot 是 V 维稀疏向量 (V 通常上万)
- 直接 matmul 极其浪费 (大部分元素是 0)
- 实现上就是按 id 取行 → O(1)

API：
    emb = nn.Embedding(num_embeddings=V, embedding_dim=d)
    emb.weight              # (V, d) 可训练参数
    emb(ids)                # ids 是 LongTensor，输出 (..., d)

LLM 里出现两次：
1. Token embedding: 把 token id → 向量
2. Position embedding (学习式): 把位置 0..T-1 → 向量
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn as nn
import torch.nn.functional as F

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 创建一个 embedding 层
# ------------------------------------------------------------
print("=" * 60)
print("1. nn.Embedding 基础")
print("=" * 60)

V, d = 10, 4  # 词表 10 个 token，每个 token 4 维
torch.manual_seed(0)
emb = nn.Embedding(V, d)
print(f"emb.weight.shape = {tuple(emb.weight.shape)}  (V, d)")
print("emb.weight =")
print(emb.weight)

ids = torch.tensor([3, 1, 5, 3])
out = emb(ids)
print(f"\n输入 ids = {ids.tolist()}")
print(f"输出 shape = {tuple(out.shape)}")
print(out)
print("→ 注意第 0 行和第 3 行一样 (都是 id=3 取出的向量)")


# ------------------------------------------------------------
# 2. 数学等价：one-hot @ W
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 等价于 one_hot @ weight")
print("=" * 60)

one_hot = F.one_hot(ids, num_classes=V).float()  # (4, V)
print(f"one_hot.shape = {tuple(one_hot.shape)}")
out_via_matmul = one_hot @ emb.weight
print("(one_hot @ emb.weight) 与 emb(ids) 是否一致：",
      torch.allclose(out_via_matmul, out))
print("→ Embedding = 高效的'按行索引'，避免 one-hot 矩阵乘法")


# ------------------------------------------------------------
# 3. 批量输入 (B, T)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 批量输入 (B, T) → (B, T, d)")
print("=" * 60)

B, T = 2, 5
ids = torch.randint(0, V, (B, T))
out = emb(ids)
print(f"ids.shape  = {tuple(ids.shape)}")
print(f"out.shape  = {tuple(out.shape)}   ← 在最后多了 d 维")
print("ids =")
print(ids)


# ------------------------------------------------------------
# 4. Embedding 是可训练的
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. emb.weight 可以反向传播")
print("=" * 60)

emb = nn.Embedding(V, d)
ids = torch.tensor([2, 5])
out = emb(ids).sum()
out.backward()
print(f"emb.weight.grad.shape = {tuple(emb.weight.grad.shape)}")
print(f"grad 非零行：{(emb.weight.grad.abs().sum(dim=-1) > 0).nonzero().flatten().tolist()}")
print("→ 只有被用到的 id (2 和 5) 对应行有梯度，其他行梯度为 0")
print("  这是 embedding 训练的一个性质：稀疏更新")


# ------------------------------------------------------------
# 5. LLM 中：token emb + position emb
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. Transformer 输入层：tok_emb + pos_emb")
print("=" * 60)

V, T_max, d = 50, 16, 8  # 词表 50, 上下文长 16, 嵌入 8
tok_emb = nn.Embedding(V, d)
pos_emb = nn.Embedding(T_max, d)

ids = torch.randint(0, V, (2, 10))  # (B=2, T=10)
B, T = ids.shape
positions = torch.arange(T)  # [0, 1, 2, ..., T-1]
x = tok_emb(ids) + pos_emb(positions)  # 广播：(B,T,d) + (T,d) → (B,T,d)
print(f"ids.shape       = {tuple(ids.shape)}")
print(f"tok_emb(ids)    = {tuple(tok_emb(ids).shape)}")
print(f"pos_emb(0..T-1) = {tuple(pos_emb(positions).shape)}")
print(f"x = tok + pos   = {tuple(x.shape)}   ← Transformer 第一层的输入")


# ------------------------------------------------------------
# 6. 权重共享 (weight tying)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. 权重共享：输入 embedding ↔ 输出 head")
print("=" * 60)

# 输出层把 (B, T, d) → (B, T, V) 是一个 d→V 的 Linear
# GPT-2 等模型让 head.weight = tok_emb.weight (共用一份矩阵)
# 节省 V*d 个参数，效果通常不变甚至更好
tok_emb = nn.Embedding(V, d)
head = nn.Linear(d, V, bias=False)
head.weight = tok_emb.weight   # 共享！
print("共享后 head.weight 与 tok_emb.weight 是同一个张量:")
print(f"  id(head.weight) == id(tok_emb.weight)? {head.weight is tok_emb.weight}")
print(f"  参数总数减少: {V * d}")


# ------------------------------------------------------------
# 7. 可视化：embedding 矩阵
# ------------------------------------------------------------
if plot:
    torch.manual_seed(42)
    V_v, d_v = 12, 8
    emb_v = nn.Embedding(V_v, d_v)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(emb_v.weight.detach().numpy(), cmap="RdBu", aspect="auto")
    axes[0].set_title(f"emb.weight  shape=({V_v}, {d_v})")
    axes[0].set_xlabel("embedding dim")
    axes[0].set_ylabel("token id")
    for i in range(V_v):
        axes[0].axhline(i + 0.5, color="white", linewidth=0.3)

    # 高亮被查到的行
    ids_v = torch.tensor([2, 5, 8, 2])
    selected = emb_v(ids_v).detach().numpy()
    axes[1].imshow(selected, cmap="RdBu", aspect="auto")
    axes[1].set_title(f"emb(ids={ids_v.tolist()})  → shape=(4, {d_v})")
    axes[1].set_xlabel("embedding dim")
    axes[1].set_ylabel("position in batch")

    plt.suptitle("Embedding = 按 id 从权重表里取行")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 创建一个 V=1000, d=64 的 embedding
emb = nn.Embedding(1000, 64)
assert emb.weight.shape == (1000, 64)
print("练习 1 ✅  emb.weight.shape =", tuple(emb.weight.shape))

# 练习 2: 给 (B=4, T=12) 的 token id batch，输出应该是什么形状？
ids = torch.randint(0, 1000, (4, 12))
out = emb(ids)
assert out.shape == (4, 12, 64)
print("练习 2 ✅  out.shape =", tuple(out.shape))

# 练习 3: 手写 embedding —— 直接用索引
class MyEmbedding(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(V, d) * 0.02)

    def forward(self, ids):
        return self.weight[ids]   # fancy indexing!

my_emb = MyEmbedding(50, 8)
my_emb.weight.data = nn.Embedding(50, 8).weight.data  # 同初始化
ref = nn.Embedding(50, 8)
ref.weight.data = my_emb.weight.data
ids = torch.tensor([3, 7, 1])
assert torch.allclose(my_emb(ids), ref(ids))
print("练习 3 ✅  手写 embedding (= weight[ids]) 与 nn.Embedding 一致")
