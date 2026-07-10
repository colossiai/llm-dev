"""
shared.py - 所有 debug 案例共用的"正确基线"

================ 它是干嘛的 ================

这个文件 = 一份**没有 bug 的、能正常训练的 Mini GPT**(直接搬自
exercises/transformer/06_train_and_generate.py, 结构逐行一致)。

每个 debug 案例(01~05)都:
  1. 先用这里的正确代码跑一遍 → 得到"正确 loss 曲线"作参照系
  2. 再只改动**一处**(注入一个经典 bug)跑一遍 → 得到"buggy 曲线"
  3. 并排对比, 你先预测、再验证

把正确代码集中在这里, 是为了让每个 bug 文件**只剩那一处改动**,
bug 一眼可见, 不被上百行样板淹没。
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 全局超参 (与 06 一致)
# ============================================================
D_MODEL = 64
N_LAYERS = 3
N_HEADS = 4
MAX_SEQ_LEN = 32
BATCH_SIZE = 16
LR = 3e-3
SEED = 42
PROMPTS = ["the q", "pack ", "how v"]   # 生成用的起始片段


# ============================================================
# 模型 (与 06 逐行一致; 唯一新增: 可替换的 attn_cls, 供 01 号 bug 用)
# ============================================================
class MultiHeadCausalSelfAttention(nn.Module):
    """正确的多头因果自注意力。01 号 bug 会子类化它, 去掉 mask。"""

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
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # ← 因果 mask
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
    def __init__(self, d_model, n_heads, max_seq_len,
                 attn_cls=MultiHeadCausalSelfAttention):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = attn_cls(d_model, n_heads, max_seq_len)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_seq_len,
                 attn_cls=MultiHeadCausalSelfAttention):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len, attn_cls=attn_cls)
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
    def generate(self, idx, max_new_tokens, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ============================================================
# 数据 (与 06 同一段绕口令, 字符级)
# ============================================================
def make_data():
    """返回 (data, vocab_size, encode, decode)。"""
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
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    encode = lambda s: [char_to_id[c] for c in s]
    decode = lambda ids: "".join(id_to_char[i] for i in ids)
    data = torch.tensor(encode(text), dtype=torch.long)
    return data, vocab_size, encode, decode


# ============================================================
# 工具: 固定随机种子 / 建模型 / 采一批数据 / 训练
# ============================================================
def set_seed(seed=SEED):
    """每次训练前调用, 保证 correct 与 buggy 跑在同一批随机数据上, 曲线可比。"""
    torch.manual_seed(seed)


def build_model(vocab_size, attn_cls=MultiHeadCausalSelfAttention):
    return MiniGPT(vocab_size, D_MODEL, N_LAYERS, N_HEADS, MAX_SEQ_LEN, attn_cls=attn_cls)


def get_batch(data, batch_size, max_seq_len, shift_targets=True):
    """
    从 data 随机切 batch_size 段。
    shift_targets=True  (正确): y 相对 x 错位 1 个 → 学"预测下一个 token"
    shift_targets=False (02号bug): y = x 本身       → 学"抄写当前 token"
    """
    ix = torch.randint(0, len(data) - max_seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + max_seq_len] for i in ix])
    if shift_targets:
        y = torch.stack([data[i + 1:i + max_seq_len + 1] for i in ix])
    else:
        y = x.clone()
    return x, y


def train(model, data, *, epochs, lr=LR, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQ_LEN,
          zero_grad=True, shift_targets=True, do_step=True):
    """
    正确训练循环。几个"正确行为"做成开关, 让 bug 文件只翻转一个:
      zero_grad=False    → 03号bug: 忘记 optimizer.zero_grad(), 梯度跨步累积
      shift_targets=False→ 02号bug: x/y 不错位
      do_step=False      → 05号bug: 忘记 optimizer.step(), 参数永不更新
      lr=1.0             → 04号bug: 学习率过大
    返回 losses 列表。
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        x, y = get_batch(data, batch_size, max_seq_len, shift_targets=shift_targets)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if zero_grad:
            optimizer.zero_grad()
        loss.backward()
        if do_step:
            optimizer.step()
        losses.append(loss.item())
    return losses


# ============================================================
# 汇报: 并排打印 loss 进度 + 两个模型的生成, 可选画图
# ============================================================
def _fmt(v):
    return "  nan  " if v != v else f"{v:7.4f}"   # v!=v 判 NaN


def report(correct_losses, buggy_losses, correct_model, buggy_model,
           encode, decode, *, title="", plot=False, n_rows=11, max_new_tokens=40):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")

    # --- loss 并排表 ---
    epochs = len(correct_losses)
    idxs = sorted(set(int(i * (epochs - 1) / (n_rows - 1)) for i in range(n_rows)))
    print(f"\n{'step':>7} | {'✅ 正确 loss':>12} | {'🐞 buggy loss':>13}")
    print("-" * 40)
    for i in idxs:
        print(f"{i:>7} | {_fmt(correct_losses[i]):>12} | {_fmt(buggy_losses[i]):>13}")
    print("-" * 40)
    print(f"{'final':>7} | {_fmt(correct_losses[-1]):>12} | {_fmt(buggy_losses[-1]):>13}")

    # --- 生成对比 ---
    print(f"\n--- 生成对比 (temperature=0.8) ---")
    for prompt in PROMPTS:
        ids = torch.tensor([encode(prompt)], dtype=torch.long)
        ok = decode(correct_model.generate(ids.clone(), max_new_tokens)[0].tolist())
        bad = decode(buggy_model.generate(ids.clone(), max_new_tokens)[0].tolist())
        print(f"\n  prompt '{prompt}'")
        print(f"    ✅ 正确 : {ok!r}")
        print(f"    🐞 buggy: {bad!r}")

    # --- 可选画图 ---
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.rcParams["axes.unicode_minus"] = False
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(correct_losses, label="correct", color="tab:green", lw=1.2)
            ax.plot(buggy_losses, label="buggy", color="tab:red", lw=1.2)
            ax.set_xlabel("step")
            ax.set_ylabel("cross-entropy loss")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\n(画图失败, 降级为纯文本: {e})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2000, help="训练 step 数")
    p.add_argument("--plot", action="store_true", help="用 matplotlib 画 correct vs buggy 两条曲线")
    return p.parse_args()
