"""
09. softmax —— 把"打分"变成"概率"

数学定义：
    softmax(x)_i = exp(x_i) / sum_j exp(x_j)

直觉：
- 输入 logits（任意实数，叫"打分"）→ 输出概率（[0,1]，加起来 = 1）
- 分越大概率越高；指数让"分高一点点"的差距被放大
- softmax 不改变大小顺序（最大的 logit 仍是最大的概率）

数值稳定性（**很重要**）：
    exp(1000) = inf  会溢出！
    技巧：先减最大值再做 exp
        softmax(x) = softmax(x - max(x))   （恒等式）
    PyTorch 内部已经这么做了，自己写时要记得

LLM 里的用途：
1. 注意力：softmax(QK^T/√d) → 每个 token 对其他 token 的"注意力权重"
2. 输出层：softmax(logits) → 词表上的下一个 token 概率分布
3. 采样：temperature 改变分布的"尖锐 / 平滑"程度

温度（temperature）：
    softmax(logits / T)
    T < 1: 分布更尖锐（贪婪 / 自信）
    T = 1: 原样
    T > 1: 分布更平滑（多样 / 随机）
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn.functional as F

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 基本 softmax
# ------------------------------------------------------------
print("=" * 60)
print("1. softmax 基础")
print("=" * 60)

logits = torch.tensor([2.0, 1.0, 0.1])
logits_weight = logits/ logits.sum().item()
probs = F.softmax(logits, dim=-1)
print(f"logits = {logits}")
print(f"logits_weight = {logits_weight}   (直接除以 sum 不是概率分布！)")
print(f"probs  = {probs}")
print(f"probs.sum() = {probs.sum().item():.4f}   (恒为 1)")
print(f"argmax 不变: logit argmax={logits.argmax().item()}, prob argmax={probs.argmax().item()}")


print(f'manual calc: ')
manual_exp_list=[]
for x in logits:
    manual_exp_list.append(torch.exp(x))
manual_probs = torch.stack(manual_exp_list) / sum(manual_exp_list)
print(f"手动计算的 probs = {manual_probs}")
print(f"和 F.softmax 一致：", torch.allclose(manual_probs, probs))

# ------------------------------------------------------------
# 2. 手写 softmax —— 一行实现
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 手写 softmax (含数值稳定性技巧)")
print("=" * 60)

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # 减去最大值：避免 exp 溢出
    x = x - x.max(dim=dim, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=dim, keepdim=True)

print("和 F.softmax 一致：", torch.allclose(my_softmax(logits), probs))

# 演示：不减最大值会溢出
big = torch.tensor([1000.0, 999.0, 998.0])
naive = big.exp() / big.exp().sum()
print(f"\nnaive  exp(1000)/sum: {naive}   ← nan/inf!")
print(f"stable F.softmax    : {F.softmax(big, dim=-1)}   ← 正常")


# ------------------------------------------------------------
# 3. dim 参数 —— 沿哪个维度求和为 1
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. dim 参数")
print("=" * 60)

x = torch.tensor([[1.0, 2.0, 3.0],
                  [1.0, 1.0, 1.0]])
print("x =")
print(x)
print("\nF.softmax(x, dim=-1)  (沿行)：")
print(F.softmax(x, dim=-1))
print("每行加起来 = 1:", F.softmax(x, dim=-1).sum(dim=-1))

print("\nF.softmax(x, dim=0)   (沿列)：")
print(F.softmax(x, dim=0))
print("每列加起来 = 1:", F.softmax(x, dim=0).sum(dim=0))

print("\n口诀：dim 指哪一维，那一维就被'压成概率'，加起来为 1")


# ------------------------------------------------------------
# 4. 温度 (temperature) 的效果
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 温度 temperature")
print("=" * 60)

logits = torch.tensor([3.0, 2.0, 1.0, 0.0])
for T in [0.5, 1.0, 2.0, 5.0]:
    p = F.softmax(logits / T, dim=-1)
    print(f"T={T:>3}  probs = {[f'{v:.3f}' for v in p.tolist()]}  最大概率={p.max().item():.3f}")
print("→ T 越小越尖锐 (贪婪)；T 越大越平滑 (多样)")


# ------------------------------------------------------------
# 5. log_softmax —— 更稳定的"log 概率"
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. log_softmax (cross_entropy 的内部就用它)")
print("=" * 60)

logits = torch.tensor([2.0, 1.0, 0.1])
print("log(softmax(x)) =", F.softmax(logits, dim=-1).log())
print("log_softmax(x)  =", F.log_softmax(logits, dim=-1))
print("→ 数学上等价，但 log_softmax 数值更稳定")
print("  (softmax 后再 log 容易出现 log(0) = -inf)")


# ------------------------------------------------------------
# 6. 在 LLM 注意力里
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. 在 Transformer 注意力里")
print("=" * 60)

B, H, T, d = 1, 2, 4, 8
Q = torch.randn(B, H, T, d)
K = torch.randn(B, H, T, d)
scores = Q @ K.transpose(-2, -1) / (d ** 0.5)  # (B, H, T, T)
attn = F.softmax(scores, dim=-1)
print(f"scores shape = {tuple(scores.shape)}")
print(f"attn   shape = {tuple(attn.shape)}")
print(f"attn[0, 0] (head 0 的注意力矩阵):")
print(attn[0, 0])
print("每行加起来 = 1 (每个 token 对其他 token 的关注是概率分布):")
print(attn[0, 0].sum(dim=-1))


# ------------------------------------------------------------
# 7. 可视化温度的效果
# ------------------------------------------------------------
if plot:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0])
    for ax, T in zip(axes, [0.5, 1.0, 2.0, 5.0]):
        p = F.softmax(logits / T, dim=-1).numpy()
        ax.bar(range(len(p)), p, color="steelblue")
        ax.set_ylim(0, 1)
        ax.set_title(f"T = {T}")
        ax.set_xticks(range(len(p)))
        for i, v in enumerate(p):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.suptitle("同一组 logits，不同温度下的概率分布")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: softmax 之后每行和为 1
x = torch.randn(4, 10)
# TODO
p = F.softmax(x, dim=-1)
assert torch.allclose(p.sum(dim=-1), torch.ones(4))
print("练习 1 ✅  softmax(x, dim=-1).sum(-1) =", p.sum(dim=-1).tolist())

# 练习 2: 给定 logits，求"前 3 大概率"对应的 token id
logits = torch.tensor([0.1, 5.0, 2.0, 3.0, 0.5, 4.0])
# TODO: 用 topk 拿前 3
top_probs, top_ids = F.softmax(logits, dim=-1).topk(3)
assert top_ids.tolist() == [1, 5, 3]
print(f"练习 2 ✅  top-3 ids = {top_ids.tolist()}  probs = {[f'{v:.3f}' for v in top_probs.tolist()]}")

# 练习 3: softmax 对 logits 加常数不变 (平移不变性)
a = torch.tensor([1.0, 2.0, 3.0])
b = a + 100  # 加 100
assert torch.allclose(F.softmax(a, dim=-1), F.softmax(b, dim=-1))
print("练习 3 ✅  softmax(x) == softmax(x + c)  (平移不变)")
print("    数值稳定性技巧就是利用这个性质：减去最大值")
