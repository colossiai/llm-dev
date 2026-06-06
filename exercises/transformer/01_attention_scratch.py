"""
01 - Attention 本体:Scaled Dot-Product Attention

================ 给零基础读者的 5 分钟讲解 ================

【Attention 是什么? 一句话直觉】
  每个 token 用自己的"提问" Q 去和其他所有 token 的"标签" K 对比, 算出
  "我应该关注谁、关注多少" 的权重, 然后把所有 token 的"内容" V 按这个权重
  加权求和, 当成自己的新表示。

  → 这是 LLM 的"心脏", 也是它能理解上下文的根本机制。

【公式】
                Q · K^T
  Attention(Q, K, V) = softmax( ────────── ) · V
                       √(d_k)

  分三步:
    1. 算"打分": scores = Q · K^T         (每个 token 对每个 token 的相关度分数)
    2. 缩放 + softmax: 把分数转成"注意力权重"(每行加起来 = 1)
    3. 用权重对 V 加权求和

【为什么除以 √d_k?】
  d_k 越大, Q·K 的方差越大, softmax 会变得"尖锐"(几乎只关注一个 token)。
  除以 √d_k 把方差控制住, softmax 输出更平滑, 训练更稳。

【为什么需要 Q、K、V 三个矩阵?】
  每个 token 的 embedding 通过 3 个不同的 Linear 层投影出 Q/K/V。
  类比:
    Q (Query) = "我想找什么"     (问题)
    K (Key)   = "我能提供什么"   (索引/标签)
    V (Value) = "我真实的信息"   (内容)
  Q 和 K 比对相关性, 决定从哪个 token 的 V 取信息。

【本脚本】
  从零(用 numpy)手写一遍 Attention, 不用 PyTorch 高级 API,
  让你看穿每一步 shape 怎么变。
  最后画注意力矩阵热图, 直观看到"每个 token 关注哪些 token"。
"""

import numpy as np
import matplotlib.pyplot as plt

import common


def softmax(x, axis=-1):
    # 减去最大值是数值稳定的标准技巧 (避免 exp(大数) 溢出)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Q: (T, d_k)  T 个 token 的 query
    K: (T, d_k)  T 个 token 的 key
    V: (T, d_v)  T 个 token 的 value

    返回: 输出 (T, d_v), 注意力权重 (T, T)
    """
    d_k = Q.shape[-1]

    # 1. 算打分: 每个 token 的 query 和每个 token 的 key 做点积
    #    scores[i, j] = Q[i] · K[j], 意思是 token i 对 token j 的相关度
    scores = Q @ K.T              # (T, T)

    # 2. 缩放 + softmax → 注意力权重 (每行加起来 = 1)
    scores = scores / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)  # (T, T)

    # 3. 用权重加权求和 V
    #    output[i] = sum_j attn[i, j] * V[j]
    out = attn @ V                # (T, d_v)
    return out, attn


def main():
    args = common.parse_args()
    np.random.seed(0)

    # =========================================================
    # 1. 造一个小例子: 5 个 token, 每个 token 的 embedding 是 8 维
    # =========================================================
    T = 5         # 序列长度 (5 个 token)
    d_model = 8   # 每个 token 的 embedding 维度
    d_k = 8       # Q/K 的维度 (这里和 d_model 一样)

    # 模拟 5 个 token 的 embedding (实际场景中来自 embedding 层)
    tokens = ["The", "cat", "sat", "on", "mat"]
    X = np.random.randn(T, d_model)
    print(f"输入 X.shape = {X.shape}  ({T} 个 token, 每个 {d_model} 维)")

    # =========================================================
    # 2. 三个投影矩阵 W_Q, W_K, W_V (实际中是可学习的 nn.Linear)
    #    它们的作用: 把同一个 X 投影到三种不同的"视角"
    # =========================================================
    W_Q = np.random.randn(d_model, d_k) * 0.5
    W_K = np.random.randn(d_model, d_k) * 0.5
    W_V = np.random.randn(d_model, d_k) * 0.5

    Q = X @ W_Q   # (T, d_k) — "我想找什么"
    K = X @ W_K   # (T, d_k) — "我能提供什么"
    V = X @ W_V   # (T, d_v) — "我真实的内容"
    print(f"Q.shape = {Q.shape}  ({T} 个 query)")
    print(f"K.shape = {K.shape}  ({T} 个 key)")
    print(f"V.shape = {V.shape}  ({T} 个 value)")

    # =========================================================
    # 3. 跑 scaled dot-product attention
    # =========================================================
    out, attn = scaled_dot_product_attention(Q, K, V)
    print(f"\n输出 out.shape = {out.shape}  (T, d_v)")
    print(f"注意力矩阵 attn.shape = {attn.shape}  (T, T)")

    # 每一行加起来应该 = 1 (这是 softmax 的性质)
    print(f"\n每行权重之和:")
    for i, row_sum in enumerate(attn.sum(axis=-1)):
        print(f"  token {i} ({tokens[i]:<4}): {row_sum:.4f}  (应该=1)")

    # =========================================================
    # 4. 看一下具体的注意力分布
    # =========================================================
    print(f"\n注意力权重矩阵 attn[i, j] = token i 关注 token j 的程度:")
    print("        " + "    ".join(f"{t:>5}" for t in tokens))
    for i, t in enumerate(tokens):
        row = "  ".join(f"{w:.3f}" for w in attn[i])
        print(f"  {t:<5} | {row}")

    # =========================================================
    # 5. 可视化注意力矩阵 (热图)
    # =========================================================
    if args.draw:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图: 原始打分 (未 softmax)
        scores = (Q @ K.T) / np.sqrt(d_k)
        im0 = axes[0].imshow(scores, cmap="RdBu_r", aspect="auto")
        axes[0].set_title("原始打分 (Q·K^T / √d_k)\n数值任意, 还不是概率")
        axes[0].set_xticks(range(T))
        axes[0].set_yticks(range(T))
        axes[0].set_xticklabels(tokens)
        axes[0].set_yticklabels(tokens)
        axes[0].set_xlabel("key (被关注的 token)")
        axes[0].set_ylabel("query (在关注的 token)")
        for i in range(T):
            for j in range(T):
                axes[0].text(j, i, f"{scores[i, j]:.2f}",
                             ha="center", va="center", fontsize=9, color="black")
        plt.colorbar(im0, ax=axes[0])

        # 右图: softmax 后的注意力权重 (每行和为 1)
        im1 = axes[1].imshow(attn, cmap="Blues", aspect="auto", vmin=0, vmax=1)
        axes[1].set_title("Softmax 后:注意力权重\n(每行加起来=1, 越深越关注)")
        axes[1].set_xticks(range(T))
        axes[1].set_yticks(range(T))
        axes[1].set_xticklabels(tokens)
        axes[1].set_yticklabels(tokens)
        axes[1].set_xlabel("key (被关注的 token)")
        axes[1].set_ylabel("query (在关注的 token)")
        for i in range(T):
            for j in range(T):
                color = "white" if attn[i, j] > 0.5 else "black"
                axes[1].text(j, i, f"{attn[i, j]:.2f}",
                             ha="center", va="center", fontsize=9, color=color)
        plt.colorbar(im1, ax=axes[1])

        plt.suptitle("Scaled Dot-Product Attention 注意力矩阵", fontsize=14)
        plt.tight_layout()
        common.finalize(args, "01_attention_scratch", bbox_inches="tight")
    else:
        print("\n(未画图。加 --plot 显示图, --save 保存到 plots/)")


if __name__ == "__main__":
    main()
