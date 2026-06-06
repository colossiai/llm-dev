"""
17. RoPE —— 旋转位置编码 (Rotary Position Embedding)

为什么需要位置编码？
- 自注意力本身是"对称"的：把 token 顺序打乱，结果不变
- 但语言显然有顺序 → 必须把"位置"信息注入

学习式位置嵌入 (minimal_edu 用的)：
    pos_emb = nn.Embedding(block_size, d)
    x = tok_emb + pos_emb(positions)
    缺点：模型固定上下文长度，外推困难

正弦位置编码 (原版 Transformer)：
    用固定的 sin/cos 函数生成位置向量
    缺点：是"加法"形式，注入位置后跟内容混合

RoPE (现代 LLaMA, Qwen 等用的)：
    不加 position 向量，而是**旋转 Q 和 K 向量本身**
    神奇性质: <RoPE(q, m), RoPE(k, n)> 只依赖 (m - n)  ← 相对位置！
    优点：天然编码"相对距离"、好外推、不增参数

数学定义 (二维基础):
    对于 token 在位置 m 的二维子向量 (x0, x1)：
        rot(x, m) = ( x0*cos(mθ) - x1*sin(mθ),
                       x0*sin(mθ) + x1*cos(mθ) )
    即在 (x0, x1) 平面上转 mθ 角度。
    高维：把 d 维向量分成 d/2 对，每对独立用一个 θ_i 旋转

    θ_i = base^(-2i/d),  i = 0..d/2 - 1
    base 常用 10000
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 准备 cos / sin 表 —— 跟具体输入无关，可预计算
# ------------------------------------------------------------
print("=" * 60)
print("1. 预计算 cos / sin 频率表")
print("=" * 60)

def build_cos_sin(seq_len: int, head_dim: int, base: float = 10000.0):
    """返回 (cos, sin)，各为 (seq_len, head_dim/2)"""
    # 一半的维度 d/2 对应不同的角频率 θ_i
    half = head_dim // 2
    i = torch.arange(half, dtype=torch.float32)
    theta = base ** (-i / half)               # (half,)   d/2 个不同频率
    pos = torch.arange(seq_len, dtype=torch.float32)  # (T,)
    angles = pos.unsqueeze(1) * theta.unsqueeze(0)    # (T, half)
    return angles.cos(), angles.sin()


T, d = 8, 8  # seq=8, head_dim=8 (d/2=4 对)
cos, sin = build_cos_sin(T, d)
print(f"cos.shape = {tuple(cos.shape)}   (T, d/2)")
print(f"sin.shape = {tuple(sin.shape)}")
print(f"位置 0 的 cos = {cos[0].tolist()}    (都是 1, 因为角度 0)")
print(f"位置 0 的 sin = {sin[0].tolist()}    (都是 0)")
print(f"位置 1 的 cos = {[f'{v:.3f}' for v in cos[1].tolist()]}")


# ------------------------------------------------------------
# 2. 应用 RoPE：把每对 (x_{2i}, x_{2i+1}) 旋转
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. apply_rope")
print("=" * 60)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (..., T, d)
    cos: (T, d/2)
    sin: (T, d/2)
    返回旋转后的 x，同形状
    """
    # 把 x 在最后一维分成两半: x0 = 偶数下标, x1 = 奇数下标
    x0 = x[..., 0::2]   # (..., T, d/2)
    x1 = x[..., 1::2]
    # 旋转
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    # 重组回 (..., T, d): 把 y0, y1 交替穿插回去
    y = torch.stack([y0, y1], dim=-1).flatten(-2)
    return y


torch.manual_seed(0)
x = torch.randn(1, T, d)
y = apply_rope(x, cos, sin)
print(f"输入  x[0,0,:] = {[f'{v:.3f}' for v in x[0,0,:].tolist()]}")
print(f"位置 0 旋转后  = {[f'{v:.3f}' for v in y[0,0,:].tolist()]}  ← 跟输入一致 (角度 0)")
print(f"\n位置 1 输入    = {[f'{v:.3f}' for v in x[0,1,:].tolist()]}")
print(f"位置 1 旋转后  = {[f'{v:.3f}' for v in y[0,1,:].tolist()]}  ← 数值被旋转了")


# ------------------------------------------------------------
# 3. 关键性质：<RoPE(q, m), RoPE(k, n)> 只依赖 (m - n)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 性质：旋转后内积只看相对距离")
print("=" * 60)

torch.manual_seed(0)
T_v, d_v = 16, 32
cos_v, sin_v = build_cos_sin(T_v, d_v)
q = torch.randn(d_v)
k = torch.randn(d_v)

# 把 q 看作位置 m，k 看作位置 n，分别旋转再求内积
def rope_one(v, pos):
    v = v.unsqueeze(0)        # (1, d)
    v = v.unsqueeze(0)        # (1, 1, d) — 视作 (B=1, T=1, d)
    cs = cos_v[pos:pos+1]
    sn = sin_v[pos:pos+1]
    return apply_rope(v, cs, sn).squeeze()


for (m, n) in [(0, 0), (0, 1), (3, 4), (10, 11), (5, 8), (10, 13)]:
    qm = rope_one(q, m)
    kn = rope_one(k, n)
    dot = (qm * kn).sum().item()
    print(f"  m={m:>2}, n={n:>2}, m-n={m-n:>3}  →  <RoPE(q,m), RoPE(k,n)> = {dot:.4f}")

print("→ 不同 (m,n) 但 m-n 相同的对，内积是同一个值")
print("  这就是 RoPE 编码相对位置的本质")


# ------------------------------------------------------------
# 4. 在注意力里怎么用
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. RoPE 在 attention 里的位置")
print("=" * 60)

print("""
普通注意力:
    Q = x @ W_Q
    K = x @ W_K
    scores = Q @ K.transpose(-2, -1) / sqrt(d)

加 RoPE 注意力:
    Q = x @ W_Q
    K = x @ W_K
    Q = apply_rope(Q, cos, sin)   # 只对 Q, K 应用
    K = apply_rope(K, cos, sin)   # V 不动
    scores = Q @ K.transpose(-2, -1) / sqrt(d)

→ RoPE 是在 attention 计算前的 "Q/K 修饰"，
  不像 nn.Embedding 那样在输入层加一次。
""")


# ------------------------------------------------------------
# 5. 模拟一个带 RoPE 的小型 attention
# ------------------------------------------------------------
print("=" * 60)
print("5. 小型 RoPE attention 演示")
print("=" * 60)

B, H, T_a, d_h = 1, 2, 6, 8
cos_a, sin_a = build_cos_sin(T_a, d_h)

Q = torch.randn(B, H, T_a, d_h)
K = torch.randn(B, H, T_a, d_h)

# 直接调用 apply_rope (cos/sin 会广播到 H)
Q_rot = apply_rope(Q, cos_a, sin_a)
K_rot = apply_rope(K, cos_a, sin_a)

scores = Q_rot @ K_rot.transpose(-2, -1) / (d_h ** 0.5)
print(f"Q.shape       = {tuple(Q.shape)}")
print(f"Q_rot.shape   = {tuple(Q_rot.shape)}   (形状不变)")
print(f"scores.shape  = {tuple(scores.shape)}  (普通注意力分数, 但含相对位置)")


# ------------------------------------------------------------
# 6. 可视化：cos/sin 表 + 一个二维向量被旋转
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 左: cos 表
    cos_full, sin_full = build_cos_sin(seq_len=32, head_dim=16)
    axes[0].imshow(cos_full.numpy(), cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    axes[0].set_title("cos 表 (T=32, d/2=8)")
    axes[0].set_xlabel("frequency idx i (d/2 个)")
    axes[0].set_ylabel("position m")

    # 中: 不同位置的 cos[m, 0]
    axes[1].plot(cos_full[:, 0].numpy(), label="cos (freq 0, 最高频)")
    axes[1].plot(cos_full[:, -1].numpy(), label="cos (最后, 最低频)")
    axes[1].set_xlabel("position m")
    axes[1].set_ylabel("cos(mθ_i)")
    axes[1].set_title("不同频率轴上的位置编码")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 右: 旋转一个二维向量 (x0, x1) 演示
    angles = torch.linspace(0, 2 * torch.pi, 9)
    x0_base, x1_base = 1.0, 0.0
    axes[2].set_xlim(-1.5, 1.5); axes[2].set_ylim(-1.5, 1.5)
    axes[2].axhline(0, color="gray", linewidth=0.5)
    axes[2].axvline(0, color="gray", linewidth=0.5)
    for a in angles:
        x0n = x0_base * a.cos() - x1_base * a.sin()
        x1n = x0_base * a.sin() + x1_base * a.cos()
        axes[2].arrow(0, 0, float(x0n), float(x1n), head_width=0.04,
                     length_includes_head=True, color="steelblue", alpha=0.6)
    axes[2].set_title("RoPE 的本质：在 2D 平面上旋转向量")
    axes[2].set_aspect("equal")

    plt.tight_layout()
    # plt.savefig('./plots/17_rope.png')
    plt.show()


# ------------------------------------------------------------
# 7. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 位置 0 应用 RoPE 是恒等变换
x = torch.randn(1, 1, 8)
cos1, sin1 = build_cos_sin(1, 8)
y = apply_rope(x, cos1, sin1)
assert torch.allclose(x, y, atol=1e-6)
print("练习 1 ✅  位置 0 处 RoPE = 恒等变换")

# 练习 2: 验证 RoPE 不改变向量的模长 (旋转保模)
x = torch.randn(1, 5, 16)
cos2, sin2 = build_cos_sin(5, 16)
y = apply_rope(x, cos2, sin2)
assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)
print("练习 2 ✅  RoPE 保持向量模长不变 (它是旋转)")

# 练习 3: 把 RoPE 套进一个 Q @ K^T，看不同位置对的相似度
torch.manual_seed(0)
T_e, d_e = 10, 32
cos_e, sin_e = build_cos_sin(T_e, d_e)
v = torch.randn(d_e)   # 故意让所有位置使用同一个内容向量
Q = v.unsqueeze(0).expand(T_e, d_e).clone().unsqueeze(0)  # (1, T, d)
K = Q.clone()
Q_r = apply_rope(Q, cos_e, sin_e)
K_r = apply_rope(K, cos_e, sin_e)
scores = (Q_r @ K_r.transpose(-2, -1))[0]
# 因为内容相同，所以 scores[i, j] 只依赖于 i - j
# 检查同一对角线上的值相等
for offset in range(1, 4):
    diag = scores.diagonal(offset=offset)
    assert torch.allclose(diag, diag[0].expand_as(diag), atol=1e-4)
print("练习 3 ✅  同内容下, RoPE 注意力分数沿对角线常数 → 只看相对位置")
