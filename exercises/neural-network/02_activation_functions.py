"""
02 - 激活函数全家桶

================ 给零基础读者的 5 分钟讲解 ================

【什么是激活函数 (activation function) ?】
  神经网络的每一层都是: y = W·x + b
  这是一个"线性"运算 (画在图上是直线/平面)。
  如果只堆这种层, 不管堆多少层, 整体还是一条直线
  (数学事实: 线性函数的复合还是线性函数)。

  → 为了让网络能学"弯的东西" (比如 XOR、月牙形分布、人脸图像),
    必须在每一层后面"折一下" —— 这个"折"就叫激活函数。

  最简单的"折":  ReLU(x) = max(0, x)
  小于 0 全部压成 0, 大于 0 不变 → 在 x=0 处折了一下。

【为什么这里画 6 个?】
  历史上人们试过很多种激活函数, 各有优缺点。
  这里画出现代最常见的 6 种, 一眼对比:
    ReLU       简单粗暴, 现代 LLM 的 FFN 早期用这个
    Sigmoid    早期用, 输出在 (0,1), 但两端"饱和"导致梯度消失
    Tanh       Sigmoid 的中心化版本, 输出 (-1, 1)
    GELU       GPT/BERT 用的, 比 ReLU 平滑
    LeakyReLU  修复 ReLU 在负半轴"死亡"的问题
    SiLU       LLaMA 用 (也叫 Swish), 等于 x * sigmoid(x)

【为什么导数曲线也要画?】
  神经网络训练靠"反向传播", 每一步都要乘上激活函数的导数。
  如果某个区域导数 ≈ 0, 那里就"传不动"梯度, 训练会卡住。
  看导数曲线 = 提前知道这个激活在哪些区域会"罢工"。
  例如 Sigmoid 在 |x| > 4 时导数几乎为 0, 这就是"梯度消失"问题。
"""

import matplotlib.pyplot as plt
import numpy as np

import common


# =============================================================
# 6 个激活函数 + 它们的导数 (纯 numpy 实现, 数学公式直接照搬)
# 注意: 每个 "_grad" 函数返回的是该激活函数关于 x 的导数,
#       反向传播时会用到 (后面的脚本会讲)。
# =============================================================
def relu(x):
    # 公式: max(0, x) — 负数变 0, 正数原样保留
    return np.maximum(0, x)


def relu_grad(x):
    # 导数: x>0 时是 1, x<0 时是 0 (在 x=0 处不可导, 实践中当 0 处理)
    return (x > 0).astype(np.float64)


def sigmoid(x):
    # 公式: 1 / (1 + e^(-x)) — 把任意数压到 (0, 1) 区间
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    # 导数: σ(x) · (1 - σ(x)) — 形状像一个钟形, 在 x=0 处最大 (0.25)
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    # 公式: (e^x - e^-x) / (e^x + e^-x) — 把任意数压到 (-1, 1) 区间
    return np.tanh(x)


def tanh_grad(x):
    # 导数: 1 - tanh²(x)
    return 1 - np.tanh(x) ** 2


def gelu(x):
    # GELU 的 tanh 近似 (BERT/GPT 实际使用的版本)
    # 直觉: 像 ReLU 但更"平滑", 在 0 附近不会有硬折角
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    # GELU 的解析导数公式比较长, 这里用数值微分代替 (差分逼近)
    # 原理: 导数定义 f'(x) ≈ (f(x+ε) - f(x-ε)) / (2ε)
    eps = 1e-4
    return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)


def leaky_relu(x, alpha=0.1):
    # ReLU 的改良: x<0 时不压成 0, 而是乘一个小数 (默认 0.1)
    # 这样负半轴还有一点信号能传过去
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x, alpha=0.1):
    # 导数: x>0 时是 1, x<0 时是 alpha (而不是 0)
    return np.where(x > 0, 1.0, alpha)


def silu(x):
    # 公式: x * sigmoid(x), 也叫 Swish
    # 形状像 ReLU 但在 0 附近平滑, 负半轴也有微小信号
    return x * sigmoid(x)


def silu_grad(x):
    # 用乘积法则推导: d/dx [x · σ(x)] = σ(x) + x · σ'(x) = σ(x) + x·σ(x)(1-σ(x))
    s = sigmoid(x)
    return s + x * s * (1 - s)


def main():
    args = common.parse_args()
    # x 取 -5 到 5 的 400 个点, 用来画函数曲线
    x = np.linspace(-5, 5, 400)

    # 把 6 个激活打包: (名字, 函数, 导数, 一句话注释)
    funcs = [
        ("ReLU",       relu,       relu_grad,       "现代经典, FFN 常用"),
        ("Sigmoid",    sigmoid,    sigmoid_grad,    "两端饱和→梯度消失"),
        ("Tanh",       tanh,       tanh_grad,       "中心化版 Sigmoid"),
        ("GELU",       gelu,       gelu_grad,       "GPT/BERT 用"),
        ("LeakyReLU",  leaky_relu, leaky_relu_grad, "ReLU 的死亡修复版"),
        ("SiLU/Swish", silu,       silu_grad,       "LLaMA 用"),
    ]

    # =========================================================
    # 1. 打印关键点的值, 直观感受 "x=-2/0/2 时各激活输出多少"
    # =========================================================
    sample_x = np.array([-2, 0, 2])
    print(f"{'激活函数':<12} | x=-2          x=0           x=2")
    print("-" * 55)
    for name, f, _, _ in funcs:
        vals = f(sample_x)
        print(f"{name:<12} | {vals[0]:+.4f}      {vals[1]:+.4f}      {vals[2]:+.4f}")

    # 观察这个表你能看到:
    #   ReLU 把负数压成 0
    #   Sigmoid 把所有值压在 (0, 1)
    #   Tanh 把所有值压在 (-1, 1)
    #   GELU/SiLU 在负数区间不是 0, 而是有微小负值 (LLM 喜欢这个)

    # =========================================================
    # 2. 画 6 个子图, 每个 subplot 同时画函数(蓝实线) + 导数(红虚线)
    #    导数 = 0 的区域 → 梯度消失高发区
    # =========================================================
    if args.draw:
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
        common.finalize(args, "02_activation_functions", bbox_inches="tight")
    else:
        print("\n(未画图。加 --plot 显示图, --save 保存到 plots/)")


if __name__ == "__main__":
    main()
