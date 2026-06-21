"""
06 - 训练 Mini GPT + 生成续写

================ 给零基础读者的 5 分钟讲解 ================

【这是整个系列的"集大成"】
  前面 5 步:
    01 Attention → 02 因果 → 03 多头 → 04 Block → 05 Mini GPT 架构
  这里把它们组装起来, **在一段真实文本上训练**, 然后看它能不能"续写"。

【训练数据】
  一段经典的英文打字练习句:
      "the quick brown fox jumps over the lazy dog. ..."
  我们用**字符级**分词 — 词表就是所有出现过的字符 (~30 个), 简单到极致。
  这是个"过拟合"实验:让小 GPT 把这段话背下来。

【数据怎么喂?】
  给定一段 token 序列 [t0, t1, t2, t3, t4]:
    input:  [t0, t1, t2, t3]   (前 4 个)
    target: [t1, t2, t3, t4]   (后 4 个 — 错位 1 个)
  模型要在每个位置预测下一个 token, 这就是 LLM 训练范式。

【训练循环】
  每个 epoch:
    1. 从训练文本里随机切一段长度 T 的窗口
    2. 模型 forward → logits (B, T, vocab_size)
    3. cross-entropy loss = -log(真实下一个 token 的概率)
    4. loss.backward(), optimizer.step()

【训练完干什么?】
  用模型"续写":给一个起始字符串, 让它自回归生成后续字符。
  如果训练成功, 它应该能续出原文。

【实测预期】
  ~3 万 step 后, loss 应该 << 1, 生成的文本能基本复述训练句。
  这就证明:**Transformer + Cross-Entropy + Adam = 真的能学语言模式**。
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import common

CKPT_DIR = Path(__file__).parent / "checkpoints"


# ============================================================
# 复用 05 的模型 (精简实现, 让本脚本独立可读)
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


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        从给定 token 序列 idx 开始, 自回归生成 max_new_tokens 个新 token。
        idx: (B, T)
        返回: (B, T + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 序列太长就只保留最后 max_seq_len 个 (滑窗)
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)
            # 只看最后一个位置的预测
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # 按概率采样下一个 token
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


def main():
    def add_args(p):
        p.add_argument("--epochs", type=int, default=3000, help="训练 step 数")
        p.add_argument("--save_model", action="store_true",
                       help="把训好的模型保存到 checkpoints/ 子目录")

    args = common.parse_args(add_args)
    torch.manual_seed(42)

    # =========================================================
    # 1. 训练数据: 一段经典英文文本 (字符级)
    # =========================================================
    text = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "pack my box with five dozen liquor jugs. "
        "pack my box with five dozen liquor jugs. "
        "the five boxing wizards jump quickly. "
        "the five boxing wizards jump quickly. "
        "how vexingly quick daft zebras jump. "
    )
    print(f"训练文本长度: {len(text)} 字符")
    print(f"内容: {text[:80]}...")

    # 字符级分词: 词表 = 所有不重复字符
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    print(f"\n词表大小 vocab_size = {vocab_size}")
    print(f"字符: {chars}")

    encode = lambda s: [char_to_id[c] for c in s]
    decode = lambda ids: "".join(id_to_char[i] for i in ids)

    # 把整段文本转成 token id 张量
    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"data.shape = {tuple(data.shape)}")

    # =========================================================
    # 2. 配置 + 实例化 Mini GPT
    # =========================================================
    d_model = 64
    n_layers = 3
    n_heads = 4
    max_seq_len = 32
    batch_size = 16
    lr = 3e-3

    model = MiniGPT(vocab_size, d_model, n_layers, n_heads, max_seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {n_params} ({n_params/1e3:.1f}K)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # =========================================================
    # 3. 数据采样函数 — 每次随机切一段
    # =========================================================
    def get_batch():
        """从 data 里随机切 batch_size 段, 每段长 max_seq_len + 1。"""
        # 起点: 在 [0, len-max_seq_len-1] 之间随机
        ix = torch.randint(0, len(data) - max_seq_len - 1, (batch_size,))
        # x[i] = data[ix[i] : ix[i]+max_seq_len]
        # y[i] = data[ix[i]+1 : ix[i]+max_seq_len+1]  ← 错位 1 个
        x = torch.stack([data[i:i+max_seq_len] for i in ix])
        y = torch.stack([data[i+1:i+max_seq_len+1] for i in ix])
        return x, y

    # =========================================================
    # 4. 训练循环
    # =========================================================
    print(f"\n=== 开始训练 {args.epochs} 步 ===")
    losses = []
    print(f"{'step':>6} | {'loss':>8}")
    print("-" * 20)
    for step in range(args.epochs):
        x, y = get_batch()
        logits = model(x)                            # (B, T, V)
        # cross_entropy 需要 (B*T, V) 和 (B*T,) 形状
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % (args.epochs // 20) == 0 or step == args.epochs - 1:
            print(f"{step:>6} | {loss.item():>8.4f}")

    print(f"\n最终 loss = {losses[-1]:.4f}")
    print(f"参考: log({vocab_size}) = {torch.tensor(vocab_size).float().log().item():.4f} "
          f"(完全随机时的 loss)")

    # =========================================================
    # 5. 用训好的模型生成续写
    # =========================================================
    print(f"\n=== 用训好的模型生成 ===")
    prompts = ["the q", "pack ", "how v"]
    for prompt in prompts:
        ids = torch.tensor([encode(prompt)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=40, temperature=0.8)
        generated_text = decode(out[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"续写: '{generated_text}'")

    # =========================================================
    # 6. 保存模型到磁盘
    # =========================================================
    # 注意: 只存 model 本身不够 — 重新加载时还需要知道模型怎么搭 (config)
    # 以及字符↔id 的映射 (vocab), 否则没法 encode/decode。
    # 所以把三样东西打包成一个自包含的 checkpoint。
    if args.save_model:
        CKPT_DIR.mkdir(exist_ok=True)
        ckpt_path = CKPT_DIR / "06_minigpt.pt"
        torch.save(
            {
                "model_state": model.state_dict(),          # 学到的权重 (核心)
                "config": {                                  # 怎么重建模型骨架
                    "vocab_size": vocab_size,
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "max_seq_len": max_seq_len,
                },
                "vocab": {                                   # 字符 ↔ id 映射
                    "char_to_id": char_to_id,
                    "id_to_char": id_to_char,
                },
                "final_loss": losses[-1],
            },
            ckpt_path,
        )
        print(f"\n模型已保存到 {ckpt_path}")
        print("  重新加载示例:")
        print("    ckpt = torch.load(path)")
        print("    model = MiniGPT(**ckpt['config'])")
        print("    model.load_state_dict(ckpt['model_state'])")
    else:
        print("\n(未保存模型。加 --save_model 保存到 checkpoints/)")

    # =========================================================
    # 7. 画 loss 曲线
    # =========================================================
    if args.draw:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(losses, color="tab:blue", linewidth=0.8, alpha=0.5, label="原始 loss")
        # 滑动平均让曲线平滑
        window = max(1, len(losses) // 50)
        if window > 1:
            smoothed = torch.tensor(losses).unfold(0, window, 1).mean(dim=-1)
            ax.plot(range(window - 1, len(losses)), smoothed,
                    color="tab:red", linewidth=2, label=f"滑动平均 (win={window})")
        ax.axhline(torch.tensor(vocab_size).float().log().item(),
                   color="gray", linestyle="--",
                   label=f"随机基线 log({vocab_size})")
        ax.set_xlabel("step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.set_title(f"Mini GPT 训练曲线 ({n_params/1e3:.0f}K 参数, "
                     f"{n_layers} 层, d={d_model}, heads={n_heads})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        common.finalize(args, "06_train_and_generate")
    else:
        print("\n(未画图。加 --plot 显示图, --save 保存到 plots/)")


if __name__ == "__main__":
    main()
