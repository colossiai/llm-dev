"""
10. cross-entropy —— 语言模型的损失函数

核心定义：
    H(p, q) = - sum_i p_i log q_i
    p: 真实分布 (在 LLM 里通常是 one-hot：只有正确 token 是 1)
    q: 预测分布 (= softmax(logits))

当 p 是 one-hot 时（只有正确类别 c 为 1）：
    H = - log q_c = - log softmax(logits)_c

直觉：
- 模型给"正确答案"分配的概率 q_c 越高 → -log q_c 越小 → loss 越低
- q_c = 1.0 → loss = 0 (完美)
- q_c = 0.5 → loss ≈ 0.693
- q_c = 0.0 → loss = +∞ (灾难)

PyTorch 的 API（**关键陷阱**）：
    F.cross_entropy(logits, target)
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    输入是未经 softmax 的 logits！不是概率！
    内部 = log_softmax + nll_loss (这样更数值稳定)

LLM 训练目标 (next-token prediction):
    给定前 t 个 token，预测第 t+1 个
    logits shape: (B, T, V)   (B 个样本, T 个位置, V 个词)
    target shape: (B, T)      (每个位置的"正确下一个 token id")
    loss = mean over all (B*T) positions of -log p(target | context)
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn.functional as F

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. 基础：单个样本
# ------------------------------------------------------------
print("=" * 60)
print("1. 单样本：3 类分类")
print("=" * 60)

logits = torch.tensor([2.0, 1.0, 0.1])
target = torch.tensor(0)   # 正确类别 = 0

# 方法 1: F.cross_entropy (推荐)
loss = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
print(f"logits = {logits}")
print(f"target = {target.item()}  (正确类别 = 0)")
print(f"loss   = {loss.item():.4f}")

# 方法 2: 手算验证
probs = F.softmax(logits, dim=-1)
manual = -probs[target].log()
print(f"手算 -log(p[0]) = -log({probs[0].item():.4f}) = {manual.item():.4f}")
print("✅ 一致")


# ------------------------------------------------------------
# 2. loss 与"正确类别概率"的关系
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. loss 随正确概率的变化")
print("=" * 60)

for p_correct in [0.99, 0.9, 0.5, 0.1, 0.01]:
    loss = -torch.tensor(p_correct).log().item()
    print(f"p(正确) = {p_correct:>4}   →   loss = -log(p) = {loss:.4f}")
print("→ 自信预测对 = 低 loss；自信预测错 = 巨大 loss")


# ------------------------------------------------------------
# 3. 批量：B 个样本同时算
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 批量分类")
print("=" * 60)

B, C = 4, 5  # 4 个样本，5 个类别
logits = torch.randn(B, C)
targets = torch.tensor([0, 2, 4, 1])
loss = F.cross_entropy(logits, targets)
print(f"logits.shape  = {tuple(logits.shape)}")
print(f"targets.shape = {tuple(targets.shape)}")
print(f"loss (mean over batch) = {loss.item():.4f}")

# reduction='none' 看每个样本的 loss
per_sample = F.cross_entropy(logits, targets, reduction="none")
print(f"per-sample loss = {per_sample.tolist()}")
print(f"mean = {per_sample.mean().item():.4f}  ← 和上面一致")


# ------------------------------------------------------------
# 4. LLM 风格：(B, T, V) + (B, T)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. LLM 风格：next-token prediction")
print("=" * 60)

B, T, V = 2, 10, 1000  # batch, 序列长, 词表
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
print(f"logits.shape  = {tuple(logits.shape)}  (B, T, V)")
print(f"targets.shape = {tuple(targets.shape)}  (B, T)")

# 关键技巧：把 (B, T) 摊平成 (B*T)，把 (B, T, V) 摊平成 (B*T, V)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
print(f"loss = {loss.item():.4f}")
print("\n等价写法 (PyTorch 也支持直接传 (B, V, T))：")
loss2 = F.cross_entropy(logits.transpose(1, 2), targets)  # (B, V, T) + (B, T)
print(f"loss2 = {loss2.item():.4f}  ← 完全一致")

# 随机模型的 baseline：loss ≈ log(V)
print(f"\n随机初始化模型的 loss 应该 ≈ log(V) = log({V}) = {torch.tensor(float(V)).log().item():.4f}")
print("→ 训练时第一步 loss 接近这个值就说明实现没问题")


# ------------------------------------------------------------
# 5. ignore_index —— 忽略 padding
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. ignore_index：忽略 padding 位置")
print("=" * 60)

logits = torch.randn(3, 5)
targets = torch.tensor([0, -100, 4])  # 中间那个用 -100 表示"忽略"
loss = F.cross_entropy(logits, targets, ignore_index=-100)
print(f"target = {targets.tolist()}   (-100 = 不参与 loss)")
print(f"loss   = {loss.item():.4f}  (只用了 2 个样本求平均)")


# ------------------------------------------------------------
# 6. nll_loss + log_softmax = cross_entropy
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. cross_entropy 的内部分解")
print("=" * 60)

logits = torch.randn(4, 5)
targets = torch.tensor([0, 1, 2, 3])

ce = F.cross_entropy(logits, targets)
ll = F.nll_loss(F.log_softmax(logits, dim=-1), targets)
print(f"F.cross_entropy           = {ce.item():.6f}")
print(f"F.nll_loss(log_softmax)   = {ll.item():.6f}")
print("→ 两者完全等价。cross_entropy = log_softmax + nll_loss")
print("  (合并是为了数值稳定 + 一次性 fused 计算)")


# ------------------------------------------------------------
# 7. 可视化：loss 随概率的曲线
# ------------------------------------------------------------
if plot:
    fig, ax = plt.subplots(figsize=(7, 4))
    p = torch.linspace(0.01, 1.0, 100)
    loss_curve = -p.log()
    ax.plot(p.numpy(), loss_curve.numpy(), "steelblue", linewidth=2)
    ax.set_xlabel("p (正确类别的预测概率)")
    ax.set_ylabel("cross-entropy loss = -log(p)")
    ax.set_title("loss 随'正确概率'的关系：自信错 → 巨额惩罚")
    ax.grid(True, alpha=0.3)
    for p_v, lbl in [(0.5, "随机猜"), (0.1, "差"), (0.01, "极差")]:
        l = -torch.tensor(p_v).log().item()
        ax.scatter([p_v], [l], color="red", zorder=5)
        ax.annotate(f"p={p_v}\nloss={l:.2f}", (p_v, l), xytext=(10, 5), textcoords="offset points", fontsize=9)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 用 cross_entropy 算"完美预测"的 loss
logits = torch.tensor([[100.0, 0.0, 0.0]])
target = torch.tensor([0])
loss = F.cross_entropy(logits, target)
assert loss.item() < 1e-6
print(f"练习 1 ✅  完美预测 loss = {loss.item():.2e}  (接近 0)")

# 练习 2: 把 LLM 风格的损失展平为 2D
B, T, V = 3, 7, 20
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))
# TODO: 用 reshape 把 (B, T, V) → (B*T, V)，(B, T) → (B*T,)
loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
assert loss.dim() == 0  # 标量
print(f"练习 2 ✅  loss = {loss.item():.4f}")

# 练习 3: 假设词表 V=10000，随机模型 loss 应该是多少？
import math
expected = math.log(10000)
print(f"练习 3 ✅  log(10000) = {expected:.4f}")
print("    训练第一步如果 loss 远超这个值 → 模型或数据有 bug")
