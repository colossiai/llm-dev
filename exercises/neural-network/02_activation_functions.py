"""
02 - 激活函数全家桶

直觉:
  - 没有激活函数, 多层网络等价于一层 (复合线性函数还是线性的)
  - 激活函数 = 在每一层后面"折一下", 让网络能拟合任意非线性函数
  - 不同激活函数的差异:
      ReLU      简单粗暴, 现代 LLM 的 FFN 早期用这个
      Sigmoid   早期用, 输出在 (0,1), 但两端"饱和"导致梯度消失
      Tanh      Sigmoid 的中心化版本, 输出 (-1, 1)
      GELU      GPT/BERT 用的, 比 ReLU 平滑
      LeakyReLU 修复 ReLU 在负半轴"死亡"的问题
      SiLU      LLaMA 用的(也叫 Swish), x * sigmoid(x)

为什么导数曲线也要看?
  反向传播 = 沿链式法则乘导数, 导数为 0 的区域梯度就消失了。
  看导数曲线就知道这个激活会在哪些地方"卡住"。
"""

import matplotlib.pyplot as plt
import numpy as np

import common


# =============================================================
# 激活函数及其导数 (纯 numpy 实现)
# =============================================================
def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1 - np.tanh(x) ** 2


def gelu(x):
    # GELU 的 tanh 近似 (BERT/GPT 实际使用的版本)
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    # 数值近似 (避免推导完整解析式)
    eps = 1e-4
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)


def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x, alpha=0.1):
    return np.where(x > 0, 1.0, alpha)


def silu(x):
    return x * sigmoid(x)


def silu_grad(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)


def main():
    args = common.parse_args()
    x = np.linspace(-5, 5, 400)

    funcs = [
        ("ReLU",       relu,       relu_grad,       "现代经典, FFN 常用"),
        ("Sigmoid",    sigmoid,    sigmoid_grad,    "两端饱和→梯度消失"),
        ("Tanh",       tanh,       tanh_grad,       "中心化版 Sigmoid"),
        ("GELU",       gelu,       gelu_grad,       "GPT/BERT 用"),
        ("LeakyReLU",  leaky_relu, leaky_relu_grad, "ReLU 的死亡修复版"),
        ("SiLU/Swish", silu,       silu_grad,       "LLaMA 用"),
    ]

    # =========================================================
    # 1. 打印关键点的值, 直观感受
    # =========================================================
    sample_x = np.array([-2, 0, 2])
    print(f"{'激活函数':<12} | x=-2          x=0           x=2")
    print("-" * 55)
    for name, f, _, _ in funcs:
        vals = f(sample_x)
        print(f"{name:<12} | {vals[0]:+.4f}      {vals[1]:+.4f}      {vals[2]:+.4f}")

    # =========================================================
    # 2. 画 6 个子图, 每个 subplot 同时画函数和导数
    # =========================================================
    if args.plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))

        for ax, (name, f, g, note) in zip(axes.flat, funcs):
            ax.plot(x, f(x), "b-", linewidth=2, label=f"{name}(x)")
            ax.plot(x, g(x), "r--", linewidth=1.5, label=f"{name}'(x) 导数")
            ax.axhline(0, color="k", linewidth=0.5)
            ax.axvline(0, color="k", linewidth=0.5)
            ax.set_title(f"{name}  —  {note}")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-1.5, 3)

        plt.suptitle("常见激活函数及其导数 (蓝=函数, 红虚线=导数)", fontsize=14, y=1.00)
        plt.tight_layout()
        path = common.save_fig("02_activation_functions", bbox_inches="tight")
        print(f"\n图已保存到 {path}")
    else:
        print("\n(未画图。加 --plot 生成 plots/02_activation_functions.png)")


if __name__ == "__main__":
    main()
