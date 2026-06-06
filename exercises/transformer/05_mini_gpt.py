"""
05 - Mini GPT (完整架构, 还没训练)

================ 给零基础读者的 5 分钟讲解 ================

【一个 GPT 模型 = 4 个部分】

  1. Token Embedding (lookup table)
     把 token id 转成 d_model 维向量。
     形状: (B, T) → (B, T, d_model)

  2. Positional Embedding (位置编码)
     给每个位置一个 d_model 维向量, 让模型知道"谁在前谁在后"。
     原始 GPT 用学习的位置 embedding, 现代 LLaMA 用 RoPE。
     形状: (T,) → (T, d_model), broadcasting 加到 token embedding 上。

  3. N 个 Transformer Block 堆叠 (上一个脚本)
     形状: (B, T, d_model) → ... → (B, T, d_model), 不变。

  4. LM Head (语言模型头)
     最后一层 LayerNorm + Linear, 把 d_model 投影到词表大小 vocab_size。
     形状: (B, T, d_model) → (B, T, vocab_size)
     每个位置输出一个"下一个 token 的打分"向量, 再 softmax 当概率。

【完整数据流】

   input ids (B, T)
        │
        ▼
   ┌─────────────────┐
   │ Token Embedding │ ─→ (B, T, d_model)
   └─────────────────┘
        │
        + ─ Positional Embedding (T, d_model) [broadcast]
        │
        ▼
   ┌─────────────────────────┐
   │ Transformer Block × N   │  ← 主体, 大部分参数都在这
   │ (Attention + FFN + ...) │
   └─────────────────────────┘
        │
        ▼
   ┌─────────────────┐
   │  LayerNorm      │
   │  + Linear (LM Head) │ ─→ (B, T, vocab_size)
   └─────────────────┘
        │
        ▼
   logits → softmax → 每个位置的下一个 token 概率分布

【这个脚本】
  搭起完整架构 (没训练), 用随机权重跑一遍, 验证形状对、参数量合理。
  下一个脚本 06 才会真正训练。

【参考: GPT 各代规模对比】
  GPT-2 small : d=768,   n_layers=12, n_heads=12 → 124M 参数
  GPT-2 large : d=1280,  n_layers=36, n_heads=20 → 774M 参数
  GPT-3       : d=12288, n_layers=96, n_heads=96 → 175B 参数
  本脚本 mini : d=64,    n_layers=4,  n_heads=4  → ~50K 参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import common


# ============================================================
# 复用 04 的 Transformer Block (精简定义在这里, 让本脚本独立可读)
# ============================================================
class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
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
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# Mini GPT 主模型
# ============================================================
class MiniGPT(nn.Module):
    """
    一个完整的小型 GPT 模型。
    所有现代 LLM (GPT, LLaMA, Mistral, Qwen) 都长这个样子, 只是规模不同。
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len

        # 1. Token embedding: 把 token id 转成向量
        # vocab_size 个不同 token, 每个映射到 d_model 维
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # 2. Positional embedding: 每个位置一个可学习向量
        # 这是 GPT-2 用的方式; LLaMA 用 RoPE 不需要这一层
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # 3. N 个 Transformer Block 堆叠
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len)
            for _ in range(n_layers)
        ])

        # 4. 最后的 LayerNorm + LM Head
        self.ln_f = nn.LayerNorm(d_model)
        # LM Head: d_model → vocab_size, 输出每个 token 的 logit
        # 注意: 没有 bias (现代 LLM 标配)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        """
        idx: (B, T)  每个元素是 token id (0 到 vocab_size-1)
        返回: logits (B, T, vocab_size) — 每个位置预测下一个 token 的打分
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"序列长度 {T} 超过 max_seq_len {self.max_seq_len}"

        # 1. 查 token embedding
        tok = self.tok_emb(idx)                          # (B, T, d_model)

        # 2. 查位置 embedding (位置 0, 1, 2, ..., T-1)
        pos_ids = torch.arange(T, device=idx.device)     # (T,)
        pos = self.pos_emb(pos_ids)                       # (T, d_model)

        # 3. 加起来 (pos broadcasting 到 B 维)
        x = tok + pos                                     # (B, T, d_model)

        # 4. 过 N 个 Transformer Block
        for block in self.blocks:
            x = block(x)

        # 5. 最后 LayerNorm + LM Head
        x = self.ln_f(x)
        logits = self.lm_head(x)                          # (B, T, vocab_size)
        return logits


def main():
    args = common.parse_args()
    torch.manual_seed(0)

    # =========================================================
    # 1. 配置一个 mini GPT
    # =========================================================
    vocab_size = 100   # 假设词表只有 100 个 token (实际 GPT-2 是 50257)
    d_model = 64       # 每个 token 的 embedding 维度
    n_layers = 4       # 堆 4 个 Transformer Block
    n_heads = 4        # 每个 block 4 个头
    max_seq_len = 32   # 最长序列长度

    print(f"配置:")
    print(f"  vocab_size = {vocab_size}")
    print(f"  d_model    = {d_model}")
    print(f"  n_layers   = {n_layers}")
    print(f"  n_heads    = {n_heads}")
    print(f"  max_seq_len= {max_seq_len}")

    model = MiniGPT(vocab_size, d_model, n_layers, n_heads, max_seq_len)

    # =========================================================
    # 2. 看参数细分
    # =========================================================
    print(f"\n=== 参数细分 ===")
    total = 0
    by_group = {"embedding": 0, "blocks": 0, "head": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if "emb" in name:
            by_group["embedding"] += n
        elif "blocks" in name:
            by_group["blocks"] += n
        else:
            by_group["head"] += n

    print(f"  Token + Pos Embedding : {by_group['embedding']:>8} ({by_group['embedding']/total*100:.1f}%)")
    print(f"  N Transformer Blocks  : {by_group['blocks']:>8} ({by_group['blocks']/total*100:.1f}%)")
    print(f"  LayerNorm + LM Head   : {by_group['head']:>8} ({by_group['head']/total*100:.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  总计 = {total} ({total/1e3:.1f}K)")
    print(f"\n  对比: GPT-3 = 175B = {175e9 / total:.0f} 倍本模型")

    # =========================================================
    # 3. 前向跑一遍, 验证形状
    # =========================================================
    print(f"\n=== 前向测试 ===")
    B, T = 2, 16
    idx = torch.randint(0, vocab_size, (B, T))   # 随便造一些 token id
    print(f"输入  idx.shape   = {tuple(idx.shape)}  (token ids)")

    logits = model(idx)
    print(f"输出  logits.shape = {tuple(logits.shape)}  (每个位置的下个 token 打分)")

    # 验证: logits[b, t, :] 通过 softmax 后是概率分布
    probs = F.softmax(logits[0, -1], dim=-1)
    print(f"\n第 0 个序列的最后一个位置预测下个 token 的概率分布:")
    print(f"  前 10 个 token 的概率: {probs[:10].tolist()}")
    print(f"  概率之和 (应该 = 1)  : {probs.sum().item():.6f}")

    # =========================================================
    # 4. 生成一段文本 (随机权重, 看一下接口长什么样)
    # =========================================================
    print(f"\n=== 用随机权重生成 (没训练, 输出是乱的) ===")
    model.eval()
    with torch.no_grad():
        # 从一个 token 开始, 自回归生成 20 个 token
        start = torch.tensor([[42]])    # 起点 token id
        generated = start
        for _ in range(20):
            logits = model(generated)
            next_logits = logits[:, -1, :]                       # 最后一个位置
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # 贪心采样
            generated = torch.cat([generated, next_token], dim=1)

        print(f"起始 token: 42")
        print(f"生成序列: {generated[0].tolist()}")
        print(f"→ 用随机权重生成的, 没意义。06 脚本会训练后再生成。")

    # 没有 plot 内容, 这个脚本主要是架构展示
    print("\n(本脚本不画图, 关注架构和形状)")


if __name__ == "__main__":
    main()
