"""
14. autograd —— PyTorch 的"自动求导"

核心思想：
- 你做的每一步运算，PyTorch 都偷偷搭一棵"计算图"
- 调 loss.backward()，它从 loss 反向走计算图，把梯度填到每个参数的 .grad 上
- optimizer 再用 .grad 更新参数

数学上对应：
    loss 关于参数 W 的偏导数 ∂loss/∂W
    用链式法则一层层算回去
    PyTorch 帮你自动完成

最少需要知道的 API：
    x.requires_grad_(True)   # 把 x 标记为"需要梯度"
    nn.Parameter(...)         # 自动 requires_grad=True
    loss.backward()           # 从 loss 反向传播梯度
    p.grad                    # 该参数累计的梯度
    optimizer.zero_grad()     # 清零（不清的话会累加，几乎总是 bug）
    optimizer.step()          # 用 .grad 更新参数
    torch.no_grad():          # 推理 / 评估时不建图，省内存
    x.detach()                # 把张量从计算图里"剪掉"
"""

import sys

import matplotlib.pyplot as plt

import common  # noqa: F401  (configures matplotlib for Chinese)
import torch
import torch.nn as nn

plot = len(sys.argv) > 1 and sys.argv[1] == "--plot"


# ------------------------------------------------------------
# 1. requires_grad 与 .grad
# ------------------------------------------------------------
print("=" * 60)
print("1. 一个最简单的反向传播")
print("=" * 60)

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2  # y = x^2 → dy/dx = 2x
print(f"x = {x.item()}")
print(f"y = x^2 = {y.item()}")
y.backward()
print(f"x.grad (= dy/dx = 2x) = {x.grad.item()}")
print("→ 在 x=3 处导数是 6 ✅")


# ------------------------------------------------------------
# 2. 多变量 + 链式法则
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 链式法则示例")
print("=" * 60)

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(4.0)
y_true = torch.tensor(10.0)

# 前向: y = w*x + b ;  loss = (y - y_true)^2
y = w * x + b
loss = (y - y_true) ** 2
loss.backward()

print(f"y = w*x + b = {y.item()}")
print(f"loss = (y - y_true)^2 = {loss.item()}")
print(f"\nw.grad = d(loss)/d(w) = 2*(y - y_true) * x = {w.grad.item()}")
print(f"b.grad = d(loss)/d(b) = 2*(y - y_true)     = {b.grad.item()}")
print("→ 手算 = 2*(9-10)*4 = -8;  2*(9-10) = -2  ✅")


# ------------------------------------------------------------
# 3. nn.Module 里的参数自动 requires_grad
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. nn.Module 的参数")
print("=" * 60)

linear = nn.Linear(3, 1)
print(f"linear.weight.requires_grad = {linear.weight.requires_grad}")
print(f"linear.bias.requires_grad   = {linear.bias.requires_grad}")
print("→ nn.Parameter 默认 requires_grad=True")


# ------------------------------------------------------------
# 4. 完整的训练步骤
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 完整的训练一步")
print("=" * 60)

torch.manual_seed(0)
# 目标：拟合 y = 3x + 2
true_w, true_b = 3.0, 2.0
model = nn.Linear(1, 1)
opt = torch.optim.SGD(model.parameters(), lr=0.05)

x = torch.randn(64, 1)
y_true = true_w * x + true_b

print(f"初始 w = {model.weight.item():.4f}, b = {model.bias.item():.4f}")
losses = []
for step in range(200):
    # 1. 前向
    pred = model(x)
    loss = ((pred - y_true) ** 2).mean()

    # 2. 清零旧梯度 (重要!)
    opt.zero_grad()

    # 3. 反向：自动算梯度填到 .grad
    loss.backward()

    # 4. 更新参数
    opt.step()
    losses.append(loss.item())

print(f"训练后 w = {model.weight.item():.4f}  (目标 3.0)")
print(f"       b = {model.bias.item():.4f}  (目标 2.0)")
print(f"最终 loss = {losses[-1]:.6f}")


# ------------------------------------------------------------
# 5. zero_grad 的重要性
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 为什么必须 zero_grad？")
print("=" * 60)

x = torch.tensor(1.0, requires_grad=True)
y = x ** 2
y.backward()
print(f"第 1 次 backward: x.grad = {x.grad.item()}")

# 不清零再 backward 一次
y2 = x ** 2  # 需要重新建图
y2.backward()
print(f"第 2 次 backward (没 zero_grad): x.grad = {x.grad.item()}  ← 累加！")

x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"zero_grad 后 backward: x.grad = {x.grad.item()}  ← 重置")
print("→ PyTorch 默认累加梯度。不清零会让训练完全错乱。")


# ------------------------------------------------------------
# 6. torch.no_grad / detach
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("6. no_grad / detach (推理时省内存)")
print("=" * 60)

x = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = x ** 2  # 这块代码不会建图
    print(f"  no_grad 块内: y.requires_grad = {y.requires_grad}")

z = x ** 2
z_d = z.detach()
print(f"detach 后: z_d.requires_grad = {z_d.requires_grad}")
print("→ 用法:")
print("  • 推理: with torch.no_grad():")
print("  • 把一部分变量当成常数: x.detach()")
print("  • model.generate() 的装饰器 @torch.no_grad()")


# ------------------------------------------------------------
# 7. 可视化训练曲线
# ------------------------------------------------------------
if plot:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, "steelblue", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("用 SGD 拟合 y = 3x + 2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 8. 练习
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("练习")
print("=" * 60)

# 练习 1: 求 f(x) = x^3 在 x=2 的导数 (应为 12)
x = torch.tensor(2.0, requires_grad=True)
f = x ** 3
f.backward()
assert abs(x.grad.item() - 12.0) < 1e-6
print("练习 1 ✅  d(x^3)/dx |_(x=2) =", x.grad.item())

# 练习 2: 求 ∇f 其中 f(x) = sum(x^2)，x 是向量 (4,)
x = torch.tensor([1., 2., 3., 4.], requires_grad=True)
f = (x ** 2).sum()
f.backward()
assert torch.allclose(x.grad, 2 * x.detach())
print(f"练习 2 ✅  ∇(sum(x^2)) = 2x = {x.grad.tolist()}")

# 练习 3: 实现一步梯度下降 (不用 optimizer)
torch.manual_seed(0)
w = torch.tensor(5.0, requires_grad=True)
target = 3.0
lr = 0.1
for _ in range(50):
    loss = (w - target) ** 2
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        w.grad.zero_()
assert abs(w.item() - target) < 1e-3
print(f"练习 3 ✅  手写 GD 后 w = {w.item():.4f}  (目标 3.0)")
