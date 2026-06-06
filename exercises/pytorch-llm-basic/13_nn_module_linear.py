"""
13. nn.Module / nn.Linear / nn.Parameter —— 神经网络的"积木"

为什么需要 nn.Module？
- 之前我们直接写 `y = x @ W + b`，参数 W, b 是裸张量
- 大模型有几十层、上千个参数，手动管理太难：
  * 收集所有参数（给 optimizer）
  * 搬到 GPU
  * 保存 / 加载权重
- nn.Module 就是这些事情的"容器"

三个核心抽象：
    nn.Parameter:   被自动登记为"可训练参数"的张量
    nn.Module:      可以包含 Parameter + 子 Module 的容器，定义 forward
    nn.Linear:      最常用的层 = y = x @ W^T + b

LLM 里每一层都是 nn.Module:
    nn.Linear, nn.LayerNorm, nn.Embedding, nn.Dropout, nn.Sequential, ...
    你自己写的 CausalSelfAttention, TransformerBlock 也是 nn.Module
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn as nn

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. nn.Linear
# ------------------------------------------------------------
print("=" * 60)
print("1. nn.Linear: y = x @ W^T + b")
print("=" * 60)

linear = nn.Linear(in_features=16, out_features=4)
print(f"linear.weight.shape = {tuple(linear.weight.shape)}   (out, in) 注意!")
print(f"linear.bias.shape   = {tuple(linear.bias.shape)}")

x = torch.randn(8, 16)
y = linear(x)
print(f"\nx.shape  = {tuple(x.shape)}")
print(f"y.shape  = {tuple(y.shape)}")

# 手算验证
y_manual = x @ linear.weight.T + linear.bias
print("手算 x @ W^T + b == linear(x):", torch.allclose(y, y_manual))


# ------------------------------------------------------------
# 2. nn.Parameter —— 张量 vs 参数
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. nn.Parameter")
print("=" * 60)

class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(3, 4))   # 会被登记
        self.W2 = torch.randn(3, 4)                  # 不会被登记 (普通张量)

a = A()
params = list(a.parameters())
print(f"a.parameters() 数量 = {len(params)}   (只有 W1 是 Parameter)")
print(f"params[0].shape = {tuple(params[0].shape)}")
print("→ 只有 nn.Parameter 包裹的张量会被 optimizer 当作可训练")
print("  普通张量赋值给 self.xxx 不会被自动收集 (除非用 register_buffer)")


# ------------------------------------------------------------
# 3. 手写一个 Linear (理解内部)
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 手写 MyLinear")
print("=" * 60)

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

my = MyLinear(16, 4)
x = torch.randn(8, 16)
print(f"MyLinear(16, 4)(x) shape = {tuple(my(x).shape)}")
print(f"parameters 数量 = {len(list(my.parameters()))}  (weight + bias)")


# ------------------------------------------------------------
# 4. 组合：Module 里嵌套 Module
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 嵌套 Module")
print("=" * 60)

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

mlp = MLP(16, 32, 4)
print(mlp)
print(f"\n参数总数 = {sum(p.numel() for p in mlp.parameters())}")
print("  = 16*32 + 32 (fc1) + 32*4 + 4 (fc2) =", 16*32 + 32 + 32*4 + 4)


# ------------------------------------------------------------
# 5. nn.Sequential —— 串联多个 Module
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. nn.Sequential")
print("=" * 60)

net = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)
print(net)
x = torch.randn(8, 16)
print(f"\nnet(x) shape = {tuple(net(x).shape)}")


# ------------------------------------------------------------
# 6. .train() / .eval() —— 切换模式
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. train / eval 模式")
print("=" * 60)

net = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 1))
net.train()
print(f"train 模式 training={net.training}")
net.eval()
print(f"eval  模式 training={net.training}")
print("→ Dropout / BatchNorm 等层在两种模式下行为不同")
print("  训练时 net.train()；推理 / 验证时 net.eval()")


# ------------------------------------------------------------
# 7. 保存 / 加载
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("7. state_dict 保存与加载")
print("=" * 60)

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
sd = net.state_dict()
print("state_dict 包含的键:")
for k, v in sd.items():
    print(f"  {k:>18s}  shape={tuple(v.shape)}")

# 创建另一个一样结构的，加载权重
net2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
net2.load_state_dict(sd)
x = torch.randn(3, 4)
print(f"\nnet 和 net2 输出一致: {torch.allclose(net(x), net2(x))}")


# ------------------------------------------------------------
# 8. 可视化：Linear 的 W 是一个矩阵
# ------------------------------------------------------------
if plot:
    torch.manual_seed(0)
    linear = nn.Linear(8, 4)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].imshow(linear.weight.detach().numpy(), cmap="RdBu", aspect="auto")
    axes[0].set_title(f"linear.weight  shape={tuple(linear.weight.shape)} (out, in)")
    axes[0].set_xlabel("input dim (in_features)")
    axes[0].set_ylabel("output dim (out_features)")

    axes[1].bar(range(4), linear.bias.detach().numpy(), color="steelblue")
    axes[1].set_title(f"linear.bias  shape={tuple(linear.bias.shape)}")
    axes[1].set_xlabel("output dim")
    axes[1].axhline(0, color="black", linewidth=0.5)

    plt.suptitle("nn.Linear(in=8, out=4): y[b] = x[b] @ W^T + b")
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 9. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: Linear(64, 16) 有多少参数？
lin = nn.Linear(64, 16)
n = sum(p.numel() for p in lin.parameters())
assert n == 64 * 16 + 16
print(f"练习 1 ✅  Linear(64,16) 参数数 = {n} = 64*16 + 16")

# 练习 2: 写一个 2 层 MLP：(d_in=10) → 32 → relu → 1
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

m = M()
x = torch.randn(5, 10)
assert m(x).shape == (5, 1)
print(f"练习 2 ✅  MLP(5,10) → {tuple(m(x).shape)}")

# 练习 3: Linear 默认 weight 的初始化是？
print(f"\n练习 3:  nn.Linear 默认初始化 = kaiming_uniform_")
print(f"   linear.weight 范围 ≈ ±{(1/64)**0.5:.4f} 对 in=64 来说")
print(f"   (实际范围: [{lin.weight.min():.4f}, {lin.weight.max():.4f}])")
