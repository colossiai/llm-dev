"""
02 - 因果自注意力 (Causal Self-Attention)

================ 给零基础读者的 5 分钟讲解 ================

【为什么叫"自"注意力 (Self-Attention)?】
  01 里 Q/K/V 都来自同一份输入 X — 序列在"自己关注自己"。
  → "自注意力" = Q, K, V 都从同一个输入 X 投影出来。
  (相对的是 "Cross Attention": Q 来自一个序列, K/V 来自另一个序列)

【为什么需要"因果掩码" (Causal Mask)?】
  LLM 在训练时要做"预测下一个 token"任务:
    输入: "The cat sat on the"
    目标: "cat sat on the mat"
  也就是说, **token i 预测时只能看到 token 0..i, 不能看到未来的 i+1..T-1**。
  否则训练时模型直接"作弊"看答案, 推理时就废了。

  因果掩码的做法: 在 softmax 之前, 把 attn[i, j] (其中 j > i) 设成 -∞,
  这样 softmax 后这些位置的权重就是 0, 模型"看不见"未来。

【掩码长什么样? (T=5 时)】
       k=0    k=1    k=2    k=3    k=4
  q=0 │  0   -inf  -inf  -inf  -inf │   只能看自己
  q=1 │  0    0    -inf  -inf  -inf │   能看到 0, 1
  q=2 │  0    0    0    -inf  -inf │   能看到 0, 1, 2
  q=3 │  0    0    0    0    -inf │
  q=4 │  0    0    0    0    0   │   全部能看

  → 用 PyTorch 的 torch.tril (lower triangular) 生成这个掩码非常方便。

【为什么用 -∞ 而不是 0?】
  因为 softmax 之前要 mask, exp(-∞) = 0, 这样 softmax 后该位置权重就是 0。
  如果用 0 mask, exp(0) = 1, 反而会把权重分一些过去。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import common


class CausalSelfAttention(nn.Module):
    """因果自注意力 (单头) — LLM 的核心组件。"""

    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        # Q, K, V 各一个 Linear 层 — 把同一个 X 投影到 3 个视角
        # 注意: 这里没有 bias, 现代 LLM (LLaMA) 不用 bias
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 输出投影 — 把 attention 输出再过一层 Linear
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # 预先生成因果掩码 (下三角矩阵), 注册成 buffer (不是参数, 但跟着 model 移动)
        # mask[i, j] = 1 表示允许, 0 表示禁止
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        """
        x: (B, T, d_model)  B 个序列, 每个长度 T, 每个 token d_model 维
        """
        B, T, d = x.shape

        # 算 Q, K, V — 全都是 (B, T, d)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 打分: Q · K^T, shape 变化 (B, T, d) @ (B, d, T) = (B, T, T)
        scores = Q @ K.transpose(-2, -1) / (d ** 0.5)

        # ★ 因果掩码: 把"未来位置"设成 -inf
        # mask[:T, :T] 取当前序列长度的子矩阵 (1=允许, 0=禁止)
        # masked_fill: 把 mask==0 的位置填 -inf
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # Softmax 后的注意力权重 (B, T, T)
        attn = F.softmax(scores, dim=-1)

        # 加权求和 V → (B, T, d)
        out = attn @ V

        # 输出投影
        out = self.W_o(out)
        return out, attn


def main():
    args = common.parse_args()
    torch.manual_seed(0)

    # =========================================================
    # 1. 造一个序列: 6 个 token, embedding 维度 16
    # =========================================================
    B, T, d_model = 1, 6, 16
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    x = torch.randn(B, T, d_model)
    print(f"输入 x.shape = {tuple(x.shape)}  ({B} batch, {T} 个 token, {d_model} 维)")

    # =========================================================
    # 2. 同时跑"有掩码"和"无掩码"两次, 对比注意力分布
    # =========================================================
    # 有掩码 — 因果自注意力
    attn_module = CausalSelfAttention(d_model, max_seq_len=T)
    out, attn_masked = attn_module(x)
    print(f"输出 out.shape = {tuple(out.shape)}")
    print(f"注意力 attn_masked.shape = {tuple(attn_masked.shape)}")

    # 没掩码版本 — 把 mask 全设成 1 (允许看任何位置), 复用同一组权重比较
    with torch.no_grad():
        Q = attn_module.W_q(x)
        K = attn_module.W_k(x)
        scores = Q @ K.transpose(-2, -1) / (d_model ** 0.5)
        attn_unmasked = F.softmax(scores, dim=-1)

    # =========================================================
    # 3. 打印两份注意力矩阵, 直接看差别
    # =========================================================
    print("\n=== 无掩码 (允许看全部) ===")
    print("        " + "    ".join(f"{t:>5}" for t in tokens))
    for i, t in enumerate(tokens):
        row = "  ".join(f"{w:.3f}" for w in attn_unmasked[0, i])
        print(f"  {t:<5} | {row}")

    print("\n=== 有因果掩码 (只能看左侧, 看不见未来) ===")
    print("        " + "    ".join(f"{t:>5}" for t in tokens))
    for i, t in enumerate(tokens):
        row = "  ".join(f"{w:.3f}" for w in attn_masked[0, i])
        print(f"  {t:<5} | {row}")
    print("→ 上三角全 0.000, 因为被掩码屏蔽")

    # 验证: 有掩码时, 上三角应该全 0
    upper = attn_masked[0].triu(diagonal=1).abs().sum().item()
    print(f"\n验证: 上三角权重之和 = {upper:.6f}  (应该 = 0)")

    # =========================================================
    # 4. 可视化: 三张热图并列
    # =========================================================
    if args.plot:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # 左: 因果掩码本身 (下三角 1, 上三角 0)
        mask = attn_module.mask[:T, :T]
        axes[0].imshow(mask.numpy(), cmap="Greys", vmin=0, vmax=1)
        axes[0].set_title("因果掩码 (1=允许看, 0=禁止)\ntorch.tril(torch.ones)")
        for i in range(T):
            for j in range(T):
                color = "white" if mask[i, j] == 1 else "black"
                axes[0].text(j, i, int(mask[i, j].item()), ha="center", va="center",
                             color=color, fontsize=11)

        # 中: 无掩码注意力
        axes[1].imshow(attn_unmasked[0].numpy(), cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("无掩码 attention\n(能看到所有 token, 训练会作弊)")
        for i in range(T):
            for j in range(T):
                v = attn_unmasked[0, i, j].item()
                color = "white" if v > 0.5 else "black"
                axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                             color=color, fontsize=9)

        # 右: 有掩码注意力
        axes[2].imshow(attn_masked[0].detach().numpy(), cmap="Blues", vmin=0, vmax=1)
        axes[2].set_title("因果掩码 attention\n(只能看自己 + 左侧, LLM 标配)")
        for i in range(T):
            for j in range(T):
                v = attn_masked[0, i, j].detach().item()
                color = "white" if v > 0.5 else "black"
                axes[2].text(j, i, f"{v:.2f}", ha="center", va="center",
                             color=color, fontsize=9)

        for ax in axes:
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(tokens)
            ax.set_yticklabels(tokens)
            ax.set_xlabel("key (被关注)")
            ax.set_ylabel("query (在关注)")

        plt.suptitle("因果自注意力 — 为什么需要掩码", fontsize=14)
        plt.tight_layout()
        path = common.save_fig("02_causal_self_attention", bbox_inches="tight")
        print(f"\n图已保存到 {path}")
    else:
        print("\n(未画图。加 --plot 生成 plots/02_causal_self_attention.png)")


if __name__ == "__main__":
    main()
