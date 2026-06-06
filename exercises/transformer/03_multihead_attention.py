"""
03 - 多头注意力 (Multi-Head Attention)

================ 给零基础读者的 5 分钟讲解 ================

【为什么要"多头"?】
  单头注意力一次只能学一种"关注模式"(比如"动词找主语")。
  多头 = 让 N 个头独立、并行地学不同的关注模式:
    - 一个头关注"语法依赖"
    - 一个头关注"指代关系"
    - 一个头关注"语义相似"
    - ...

  最后把所有头的输出拼起来 + 一层 Linear, 综合所有视角。

【实现技巧:不是真的跑 N 次,而是"形状变换"】
  把 d_model 维度切成 n_heads 份, 每份 head_dim = d_model // n_heads。
  例如 d_model=64, n_heads=8 → head_dim=8。
  每个头独立做 Attention, 用张量形状重排实现并行计算:

      (B, T, d_model)
          │
          ▼ reshape + transpose
      (B, n_heads, T, head_dim)        ← 每个头各自的 Q/K/V
          │
          ▼ 同时对所有头做 Attention
      (B, n_heads, T, head_dim)
          │
          ▼ transpose + reshape
      (B, T, d_model)                   ← 把头拼回来
          │
          ▼ W_o (Linear)
      (B, T, d_model)                   ← 最终输出

【形状变换的关键操作:reshape + transpose】
  下面代码里最绕的一步:
      x = x.view(B, T, n_heads, head_dim).transpose(1, 2)
  把"特征维"切成"头维 × 头内维度", 然后把"头维"换到第 2 位,
  这样 (B, n_heads, T, head_dim) 后续可以一次矩阵乘批量处理 n_heads 个头。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import common


class MultiHeadCausalSelfAttention(nn.Module):
    """多头因果自注意力 — Transformer 的标准组件。"""

    def __init__(self, d_model, n_heads, max_seq_len=64):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads   # 每个头分到的特征维度

        # 一次性把 Q/K/V 投影合并 (3 倍输出, 之后再切开 — 更高效)
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # 多头拼接后的输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 因果掩码 (注册成 buffer)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        x: (B, T, d_model)
        返回: out (B, T, d_model), attn (B, n_heads, T, T) — 注意力权重
        """
        B, T, d = x.shape
        nh, hd = self.n_heads, self.head_dim

        # ===== 步骤 1: 一次投影出 Q/K/V (合在一起做矩阵乘更快) =====
        qkv = self.W_qkv(x)                          # (B, T, 3*d)
        Q, K, V = qkv.chunk(3, dim=-1)               # 各自 (B, T, d)

        # ===== 步骤 2: 拆成多头并把 head 维放到前面 =====
        # (B, T, d) → (B, T, nh, hd) → (B, nh, T, hd)
        Q = Q.view(B, T, nh, hd).transpose(1, 2)
        K = K.view(B, T, nh, hd).transpose(1, 2)
        V = V.view(B, T, nh, hd).transpose(1, 2)
        # 现在每个头独立的 Q/K/V 形状是 (B, nh, T, hd)

        # ===== 步骤 3: 并行计算所有头的 attention =====
        # (B, nh, T, hd) @ (B, nh, hd, T) = (B, nh, T, T)
        # PyTorch 自动对前面的 batch 维 (B, nh) 做批量矩阵乘
        scores = Q @ K.transpose(-2, -1) / (hd ** 0.5)

        # 因果掩码 (对所有头、所有 batch 一样)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)              # (B, nh, T, T)

        # 加权求和
        out = attn @ V                                # (B, nh, T, hd)

        # ===== 步骤 4: 把多头拼回来 =====
        # (B, nh, T, hd) → (B, T, nh, hd) → (B, T, d)
        # .contiguous() 是因为 transpose 后内存不连续, view 要求连续
        out = out.transpose(1, 2).contiguous().view(B, T, d)

        # ===== 步骤 5: 输出投影 =====
        out = self.W_o(out)
        return out, attn


def main():
    args = common.parse_args()
    torch.manual_seed(0)

    # =========================================================
    # 1. 造一个 1 batch × 6 token × 32 维的输入
    # =========================================================
    B, T, d_model = 1, 6, 32
    n_heads = 4                # 32 / 4 = 8, 每个头 8 维
    tokens = ["The", "cat", "sat", "on", "the", "mat"]

    x = torch.randn(B, T, d_model)
    print(f"输入 x.shape = {tuple(x.shape)}")
    print(f"d_model = {d_model}, n_heads = {n_heads}, head_dim = {d_model // n_heads}")

    # =========================================================
    # 2. 跑 Multi-Head Attention
    # =========================================================
    mha = MultiHeadCausalSelfAttention(d_model, n_heads, max_seq_len=T)
    out, attn = mha(x)

    print(f"\n输出 out.shape = {tuple(out.shape)}  (B, T, d_model — 形状不变!)")
    print(f"注意力 attn.shape = {tuple(attn.shape)}  (B, n_heads, T, T)")
    print(f"  → 每个头都有自己的一份 attention 矩阵")

    # =========================================================
    # 3. 检查参数量
    # =========================================================
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n总参数量 = {total_params}")
    # 公式: W_qkv (d * 3d) + W_o (d * d) = 4 * d²
    print(f"理论值: 4 × {d_model}² = {4 * d_model ** 2}")

    # =========================================================
    # 4. 打印每个头各看一下注意力分布 (verify 不同头确实学到不同模式)
    # =========================================================
    print(f"\n=== 每个头的注意力矩阵 (B=0) ===")
    for h in range(n_heads):
        print(f"\nHead {h} (随机初始化, 还没训练):")
        print("        " + "    ".join(f"{t:>5}" for t in tokens))
        for i, t in enumerate(tokens):
            row = "  ".join(f"{w:.2f}" for w in attn[0, h, i])
            print(f"  {t:<5} | {row}")
    print("\n→ 4 个头的注意力矩阵都不同 — 即使初始化是随机的, 因为权重不同")

    # =========================================================
    # 5. 可视化: 4 个头的 attention 并排
    # =========================================================
    if args.plot:
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4.5))

        for h in range(n_heads):
            ax = axes[h]
            ax.imshow(attn[0, h].detach().numpy(), cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"Head {h}")
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_xlabel("key")
            if h == 0:
                ax.set_ylabel("query")
            for i in range(T):
                for j in range(T):
                    v = attn[0, h, i, j].detach().item()
                    if v > 0.001:
                        color = "white" if v > 0.5 else "black"
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color=color, fontsize=7)

        plt.suptitle(f"Multi-Head Attention — {n_heads} 个头同时关注不同模式\n"
                     f"(d_model={d_model}, head_dim={d_model // n_heads})",
                     fontsize=13)
        plt.tight_layout()
        path = common.save_fig("03_multihead_attention", bbox_inches="tight")
        print(f"\n图已保存到 {path}")
    else:
        print("\n(未画图。加 --plot 生成 plots/03_multihead_attention.png)")


if __name__ == "__main__":
    main()
