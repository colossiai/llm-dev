"""
16. 采样 (sampling) —— 从概率分布生成下一个 token

LLM 推理生成的核心循环:
    while not done:
        logits = model(context)[:, -1, :]  # 最后一个位置的预测分布
        logits = logits / temperature
        probs  = softmax(logits)
        next   = sample(probs)
        context = cat([context, next])

三种采样策略：
1. greedy (argmax)     —— 永远选概率最大的；确定性、保守、易重复
2. multinomial          —— 按概率随机抽；有随机性、更多样
3. top-k + top-p (核)   —— 先剔除"尾巴"再抽；既多样又不会胡言乱语

控制参数：
    temperature T:
        T < 1 → 分布尖锐 → 接近 greedy
        T = 1 → 原样
        T > 1 → 分布平滑 → 更随机
    top-k = 40:
        只在概率最高的 40 个候选里采样，其余的概率清零
    top-p (nucleus) = 0.9:
        从最高概率开始累计，累计到 0.9 为止，剩下的清零
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn.functional as F

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. argmax —— 贪婪采样
# ------------------------------------------------------------
print("=" * 60)
print("1. greedy: argmax")
print("=" * 60)

logits = torch.tensor([0.1, 0.5, 2.0, 1.5, 0.0])
greedy = logits.argmax().item()
print(f"logits = {logits.tolist()}")
print(f"argmax = {greedy}   (永远选 logit 最大的)")


# ------------------------------------------------------------
# 2. multinomial —— 按概率随机
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. multinomial 按概率随机抽")
print("=" * 60)

torch.manual_seed(0)
probs = F.softmax(logits, dim=-1)
print(f"probs = {[f'{p:.3f}' for p in probs.tolist()]}")

# 抽 10000 次统计频率
samples = torch.multinomial(probs, num_samples=10000, replacement=True)
freq = torch.bincount(samples, minlength=5).float() / 10000
print(f"10000 次采样频率 = {[f'{f:.3f}' for f in freq.tolist()]}")
print("→ 与 probs 接近")


# ------------------------------------------------------------
# 3. topk —— 取前 k 大的值与索引
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. topk")
print("=" * 60)

logits = torch.tensor([0.1, 5.0, 2.0, 3.0, 0.5, 4.0])
vals, ids = logits.topk(3)
print(f"logits   = {logits.tolist()}")
print(f"top-3 vals = {vals.tolist()}")
print(f"top-3 ids  = {ids.tolist()}")


# ------------------------------------------------------------
# 4. top-k 采样
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. top-k 采样：保留前 k 个，其余 -inf")
print("=" * 60)

def top_k_sample(logits, k):
    # 找到第 k 大的值
    vals, _ = logits.topk(k)
    threshold = vals[..., -1:]   # (..., 1)
    # 把小于阈值的位置改为 -inf
    logits = logits.clone()
    logits[logits < threshold] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1)

logits = torch.tensor([0.1, 5.0, 2.0, 3.0, 0.5, 4.0])
torch.manual_seed(42)
sample = top_k_sample(logits, k=3).item()
print(f"top-3 采样结果: {sample}   (只可能从 {logits.topk(3).indices.tolist()} 里出)")


# ------------------------------------------------------------
# 5. top-p (nucleus) 采样
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. top-p (nucleus) 采样")
print("=" * 60)

def top_p_sample(logits, p):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=-1)
    # 找到累计概率超过 p 的位置以后全部置 0
    mask = cumsum > p
    # 保留刚好越过 p 的那个 token (右移一位)
    mask = torch.cat([torch.zeros_like(mask[..., :1]), mask[..., :-1]], dim=-1)
    sorted_probs[mask] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    # 在排序后的概率上采样
    sample = torch.multinomial(sorted_probs, 1)
    return sorted_idx.gather(-1, sample)

logits = torch.tensor([0.1, 5.0, 2.0, 3.0, 0.5, 4.0])
probs = F.softmax(logits, dim=-1).tolist()
print(f"原 probs (排序后):")
sorted_p, sorted_i = F.softmax(logits, dim=-1).sort(descending=True)
print(f"  ids   {sorted_i.tolist()}")
print(f"  probs {[f'{p:.3f}' for p in sorted_p.tolist()]}")
print(f"  累计  {[f'{c:.3f}' for c in sorted_p.cumsum(-1).tolist()]}")
print(f"top-p=0.8 保留：累计到 ≥0.8 为止的几个 id")


# ------------------------------------------------------------
# 6. 温度的效果（直观）
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. 温度对采样多样性的影响")
print("=" * 60)

torch.manual_seed(0)
logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0])
for T in [0.1, 0.5, 1.0, 2.0]:
    p = F.softmax(logits / T, dim=-1)
    samples = torch.multinomial(p, 5000, replacement=True)
    freq = torch.bincount(samples, minlength=5).float() / 5000
    print(f"T={T:>4}  freq = {[f'{f:.2f}' for f in freq.tolist()]}")
print("→ T 小：几乎只采样到 token 0；T 大：分布更均匀")


# ------------------------------------------------------------
# 7. 完整的生成循环 (模拟，用随机 logits 替代真模型)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("7. 完整生成循环 (mock model)")
print("=" * 60)

V = 8
context = torch.tensor([[3, 5]])   # 起始 token ids，shape (B=1, T=2)
torch.manual_seed(0)

def fake_model(ids):
    # 返回随机 logits，模拟一个 LLM 输出 (B, T, V)
    B, T = ids.shape
    return torch.randn(B, T, V)

for step in range(5):
    logits = fake_model(context)[:, -1, :]      # 取最后一个位置 (B, V)
    logits = logits / 1.0                        # temperature
    # top-k
    vals, _ = logits.topk(3)
    logits[logits < vals[:, [-1]]] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, 1)        # (B, 1)
    context = torch.cat([context, next_id], dim=1)
    print(f"step {step+1}: 新 token = {next_id.item()}, context = {context[0].tolist()}")


# ------------------------------------------------------------
# 8. 可视化
# ------------------------------------------------------------
if plot:
    logits = torch.tensor([3.0, 2.0, 1.0, 0.5, 0.0, -1.0, -2.0])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # (a) greedy
    p = F.softmax(logits, dim=-1).numpy()
    bars = axes[0, 0].bar(range(len(p)), p, color="lightgray")
    bars[int(logits.argmax())].set_color("red")
    axes[0, 0].set_title("greedy: 永远选概率最大的 (红色)")

    # (b) multinomial
    axes[0, 1].bar(range(len(p)), p, color="steelblue")
    axes[0, 1].set_title("multinomial: 按概率抽 (每次都可能不同)")

    # (c) top-k=3
    vals, ids = logits.topk(3)
    mask = torch.full_like(logits, float("-inf"))
    mask[ids] = logits[ids]
    p_topk = F.softmax(mask, dim=-1).numpy()
    colors = ["red" if i in ids.tolist() else "lightgray" for i in range(len(p_topk))]
    axes[1, 0].bar(range(len(p_topk)), p_topk, color=colors)
    axes[1, 0].set_title("top-k=3: 只在前 3 大里抽")

    # (d) temperature
    for T, color in zip([0.5, 1.0, 2.0], ["red", "steelblue", "green"]):
        p_t = F.softmax(logits / T, dim=-1).numpy()
        axes[1, 1].plot(range(len(p_t)), p_t, marker="o", label=f"T={T}", color=color)
    axes[1, 1].set_title("temperature: 控制分布尖锐程度")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 实现 greedy 解码 (一步)
logits = torch.tensor([[0.1, 3.0, 2.5, 0.0]])
# TODO
next_id = logits.argmax(dim=-1, keepdim=True)
assert next_id.item() == 1
print("练习 1 ✅  greedy 选到 id =", next_id.item())

# 练习 2: 用 multinomial 从分布 [0.7, 0.2, 0.1] 采样 1000 次，验证频率近似
torch.manual_seed(0)
probs = torch.tensor([0.7, 0.2, 0.1])
samples = torch.multinomial(probs, 1000, replacement=True)
freq = torch.bincount(samples, minlength=3).float() / 1000
print(f"练习 2 ✅  采样频率 = {freq.tolist()}  (期望 ≈ [0.7, 0.2, 0.1])")

# 练习 3: 实现一个简化的 top-k 函数 —— 把非 top-k 位置的 logit 设为 -inf
def topk_filter(logits, k):
    # TODO
    vals, _ = logits.topk(k)
    threshold = vals[..., -1:]
    out = logits.clone()
    out[out < threshold] = float("-inf")
    return out

logits = torch.tensor([1.0, 5.0, 2.0, 4.0, 0.5])
filtered = topk_filter(logits, k=2)
assert (filtered == float("-inf")).sum().item() == 3
assert torch.equal((filtered != float("-inf")).nonzero().flatten(),
                   torch.tensor([1, 3]))
print(f"练习 3 ✅  top-2 后只保留 id={[1, 3]}, 其余 -inf")
