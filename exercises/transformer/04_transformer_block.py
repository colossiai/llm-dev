"""
04 - Transformer Block (一个完整的块)

================ 给零基础读者的 5 分钟讲解 ================

【一个 Transformer Block = 4 个东西串起来】
  1. Multi-Head Attention (上一个脚本)
  2. Feed-Forward Network (FFN, 一个小 MLP)
  3. Residual Connection (残差连接)
  4. LayerNorm (归一化)

【Pre-Norm vs Post-Norm】
  现代 LLM (GPT/LLaMA) 都用 "Pre-Norm":
      x = x + Attn(LayerNorm(x))   ← 先 norm 再 attn, 再加残差
      x = x + FFN(LayerNorm(x))
  原始论文 (2017) 用 "Post-Norm" (先 attn 再 norm), 现代发现 Pre-Norm 更稳。

【FFN 在干什么?】
  Attention 让 token 之间"交流", FFN 让每个 token 自己"再加工一遍"。
  FFN 是一个 2 层 MLP:
      hidden = activation(Linear1(x))   ← 维度扩展 (常 4 倍)
      out = Linear2(hidden)              ← 再降回 d_model
  现代 LLM 用 GELU 或 SiLU/SwiGLU 作为激活。

【残差连接 (x + ...) 在干什么?】
  在深层网络里, 让每一层"输出 = 输入 + 微调", 而不是完全替换。
  好处:
    1. 梯度有"快速通道"反向传播, 不会消失
    2. 即使某层学得不好, 网络仍能传递信息

【一个 Transformer Block 的数据流】
        x  (B, T, d)
        │
        │ ── 残差快捷线 ─────────────────┐
        ▼                                │
     LayerNorm                           │
        │                                │
        ▼                                │
     Multi-Head Attention                │
        │                                │
        ▼                                │
        + ← ──────────────────────────────┘
        │
        │ ── 残差快捷线 ─────────────────┐
        ▼                                │
     LayerNorm                           │
        │                                │
        ▼                                │
     FFN (Linear → GELU → Linear)        │
        │                                │
        ▼                                │
        + ← ──────────────────────────────┘
        │
        ▼
        x' (B, T, d)        ← 输出形状不变, 但内容变了

  → 整个 GPT/LLaMA 就是把这种 block 堆 N 层 (GPT-3 是 96 层)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import common


class MultiHeadCausalSelfAttention(nn.Module):
    """同 03 脚本的 MHA, 这里精简复用。"""

    def __init__(self, d_model, n_heads, max_seq_len=64):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        B, T, d = x.shape
        nh, hd = self.n_heads, self.head_dim
        Q, K, V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, nh, hd).transpose(1, 2)
        K = K.view(B, T, nh, hd).transpose(1, 2)
        V = V.view(B, T, nh, hd).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (hd ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    每个 token 独立通过同一个 MLP — 给 token 自己加工特征。
    标准配置: 中间层扩展 4 倍 (4 * d_model), GELU 激活。
    """

    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        # GELU 是 GPT/BERT 用的激活, 现代 LLaMA 用 SiLU 或 SwiGLU
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, d) → (B, T, 4d) → 激活 → (B, T, d)
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer Block (Pre-Norm 风格, GPT 系列同款)。
    输入和输出形状都是 (B, T, d_model), 可以无限堆叠。
    """

    def __init__(self, d_model, n_heads, max_seq_len=64, expansion=4):
        super().__init__()
        # 两个 LayerNorm: attention 前一个, FFN 前一个
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ffn = FeedForward(d_model, expansion)

    def forward(self, x):
        # Pre-Norm 子层 1: attention + 残差
        x = x + self.attn(self.ln1(x))
        # Pre-Norm 子层 2: FFN + 残差
        x = x + self.ffn(self.ln2(x))
        return x


def main():
    args = common.parse_args()
    torch.manual_seed(0)

    # =========================================================
    # 1. 造输入
    # =========================================================
    B, T, d_model = 2, 8, 32
    n_heads = 4

    x = torch.randn(B, T, d_model)
    print(f"输入 x.shape = {tuple(x.shape)}")
    print(f"配置: d_model={d_model}, n_heads={n_heads}, head_dim={d_model//n_heads}")

    # =========================================================
    # 2. 单个 block
    # =========================================================
    block = TransformerBlock(d_model, n_heads, max_seq_len=T)
    out = block(x)
    print(f"\n单个 Block 输出 out.shape = {tuple(out.shape)}  (形状不变!)")
    print(f"  → 关键性质: Transformer Block 不改变形状, 可以无限堆叠")

    # 看看参数构成
    print(f"\n单个 Block 参数细分:")
    for name, p in block.named_parameters():
        print(f"  {name:<30} shape={str(tuple(p.shape)):<18} numel={p.numel()}")
    total = sum(p.numel() for p in block.parameters())
    print(f"  总计 = {total} 个参数")

    # =========================================================
    # 3. 堆 6 个 block — 看输入怎么"层层加工"
    # =========================================================
    print(f"\n=== 堆叠 6 个 Block, 观察每层输出的统计量 ===")
    n_layers = 6
    blocks = nn.ModuleList([
        TransformerBlock(d_model, n_heads, max_seq_len=T)
        for _ in range(n_layers)
    ])

    h = x
    print(f"{'层':>4} | {'mean':>8} | {'std':>8} | {'range':>14}")
    print("-" * 50)
    print(f"{'输入':>4} | {h.mean().item():>+8.4f} | {h.std().item():>8.4f} | "
          f"[{h.min().item():>+6.2f}, {h.max().item():>+6.2f}]")
    for i, block in enumerate(blocks):
        h = block(h)
        print(f"{i+1:>4} | {h.mean().item():>+8.4f} | {h.std().item():>8.4f} | "
              f"[{h.min().item():>+6.2f}, {h.max().item():>+6.2f}]")

    print(f"\n→ 注意 std 保持在合理范围 (因为 LayerNorm 在工作)")
    print(f"→ 没有 LayerNorm 的话, 深层网络数值会失控")

    # 总参数量
    total_blocks_params = sum(p.numel() for p in blocks.parameters())
    print(f"\n6 层 Block 总参数 = {total_blocks_params} ({total_blocks_params/1e3:.1f}K)")
    print(f"对比 GPT-3: 96 层 × 12288 维 → 175B 参数")

    # =========================================================
    # 4. 可视化: 数据流过 6 层后的"激活分布"
    # =========================================================
    if args.draw:
        # 重新跑一遍, 收集每层输出
        outputs = [x]
        h = x
        for block in blocks:
            h = block(h)
            outputs.append(h)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # 左: 每层激活的分布 (直方图)
        for i, h in enumerate(outputs):
            label = "输入" if i == 0 else f"Layer {i}"
            axes[0].hist(h.detach().numpy().flatten(), bins=40, alpha=0.4,
                         label=label, density=True)
        axes[0].set_title("每层激活值的分布\n(LayerNorm 让分布保持稳定)")
        axes[0].set_xlabel("激活值")
        axes[0].set_ylabel("频率")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # 右: 每层的 std 变化
        stds = [h.std().item() for h in outputs]
        means = [h.mean().item() for h in outputs]
        axes[1].plot(range(len(stds)), stds, "o-", color="tab:blue", label="std")
        axes[1].plot(range(len(means)), means, "s-", color="tab:orange", label="mean")
        axes[1].axhline(0, color="gray", linewidth=0.5)
        axes[1].set_xlabel("层 (0=输入, 1-6=Block 输出)")
        axes[1].set_ylabel("数值")
        axes[1].set_title("激活的 mean 和 std 随深度的变化\n(没炸 = LayerNorm 起作用了)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f"6 层 Transformer Block 堆叠 (d={d_model}, heads={n_heads})", fontsize=13)
        plt.tight_layout()
        common.finalize(args, "04_transformer_block", bbox_inches="tight")
    else:
        print("\n(未画图。加 --plot 显示图, --save 保存到 plots/)")


if __name__ == "__main__":
    main()
