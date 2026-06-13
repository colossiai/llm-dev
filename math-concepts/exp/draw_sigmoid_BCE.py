"""
画三张图, 把 Sigmoid 和 BCE 看清楚, 顺便理解为啥它俩"天生一对":

  图 1  Sigmoid 自己:        σ(z) = 1 / (1 + e^{-z})        和它的导数 σ'(z) = σ(z)(1-σ(z))
  图 2  BCE 自己:            L(p | y) = -[ y·ln(p) + (1-y)·ln(1-p) ]   横轴是 p ∈ (0, 1)
  图 3  Sigmoid + BCE 组合:  把 z 经过 σ 得到 p, 再算 BCE → L(z | y)    这就是 BCEWithLogitsLoss

直觉:
  - Sigmoid 把"模型吐出的任意实数 z (logit)"挤进 (0, 1), 当概率用
  - BCE 衡量"预测概率 p 和真实 0/1 标签 y 的差距", 错得越自信惩罚越凶
  - 把两个串起来, 就是大多数分类任务 (含 RLHF reward model) 真正在最小化的 loss
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def draw_sigmoid(ax):
    """图 1: Sigmoid 曲线 + 它的导数。"""
    z = np.linspace(-8, 8, 400)
    s = sigmoid(z)
    ds = s * (1 - s)

    ax.plot(z, s, color="#1f77b4", lw=2.5, label=r"σ(z) = 1 / (1 + e$^{-z}$)")
    ax.plot(z, ds, color="#d62728", lw=2, linestyle="--",
            label=r"σ'(z) = σ(z)·(1 - σ(z))")

    # 两条渐近线 + 中点
    ax.axhline(1.0, color="gray", lw=0.8, linestyle=":")
    ax.axhline(0.0, color="gray", lw=0.8, linestyle=":")
    ax.axhline(0.5, color="gray", lw=0.6, linestyle=":")
    ax.axvline(0.0, color="gray", lw=0.6, linestyle=":")

    # 标记最大斜率 (z=0, 导数=0.25)
    ax.plot(0, 0.5, "o", color="#1f77b4", markersize=7, zorder=5)
    ax.plot(0, 0.25, "o", color="#d62728", markersize=7, zorder=5)
    ax.annotate("σ(0)=0.5", xy=(0, 0.5), xytext=(1.2, 0.55), fontsize=10, color="#1f77b4")
    ax.annotate("σ'(0)=0.25\n(最大斜率)", xy=(0, 0.25), xytext=(1.2, 0.30),
                fontsize=10, color="#d62728")

    ax.set_xlim(-8, 8); ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("z  (logit, 模型原始输出)")
    ax.set_ylabel("σ(z)  或  σ'(z)")
    ax.set_title("图 1: Sigmoid 把任意实数 z 压进 (0,1) 当概率\n"
                 "导数最大才 0.25 → 多层堆叠会梯度消失")
    ax.legend(loc="center right"); ax.grid(True, alpha=0.3)


def draw_bce(ax):
    """图 2: BCE 损失随预测概率 p 的变化, 分别看 y=1 和 y=0。"""
    eps = 1e-6                                  # 避免 log(0)
    p = np.linspace(eps, 1 - eps, 400)
    L_y1 = -np.log(p)                           # y=1: L = -ln(p)
    L_y0 = -np.log(1 - p)                       # y=0: L = -ln(1-p)

    ax.plot(p, L_y1, color="#1f77b4", lw=2.5, label="y = 1 真:  L = -ln(p)")
    ax.plot(p, L_y0, color="#d62728", lw=2.5, label="y = 0 真:  L = -ln(1 - p)")

    # 中间点 (p=0.5, L=ln2≈0.693): 两条曲线在这里相交
    ax.plot(0.5, np.log(2), "ko", markersize=6, zorder=5)
    ax.annotate("p=0.5\nL = ln2 ≈ 0.693", xy=(0.5, np.log(2)), xytext=(0.55, 1.5),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="black", lw=1))

    # 标注"自信错"惩罚陡增
    ax.annotate("自信猜错\n惩罚陡增 →∞",
                xy=(0.02, -np.log(0.02)), xytext=(0.18, 5.0),
                fontsize=10, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1))
    ax.annotate("自信猜错\n惩罚陡增 →∞",
                xy=(0.98, -np.log(1 - 0.98)), xytext=(0.55, 5.0),
                fontsize=10, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))

    ax.set_xlim(0, 1); ax.set_ylim(0, 6.5)
    ax.set_xlabel("p  (预测为 1 的概率)")
    ax.set_ylabel("L  (BCE 损失)")
    ax.set_title("图 2: BCE = -[ y·ln(p) + (1-y)·ln(1-p) ]\n"
                 "错得越自信, 惩罚越凶 (对数发散)")
    ax.legend(loc="upper center"); ax.grid(True, alpha=0.3)


def draw_sigmoid_plus_bce(ax):
    """图 3: 把 sigmoid 和 BCE 串起来 (= BCEWithLogitsLoss), 横轴是 logit z。

    数学化简:
        y=1: L(z) = -ln(σ(z))   = ln(1 + e^{-z})  = softplus(-z)
        y=0: L(z) = -ln(1-σ(z)) = ln(1 + e^{ z})  = softplus( z)
    """
    z = np.linspace(-6, 6, 400)
    L_y1 = np.log1p(np.exp(-z))                 # log(1 + e^-z), 数值稳定
    L_y0 = np.log1p(np.exp( z))                 # log(1 + e^+z)

    ax.plot(z, L_y1, color="#1f77b4", lw=2.5, label="y = 1 真:  L(z) = ln(1 + e$^{-z}$)")
    ax.plot(z, L_y0, color="#d62728", lw=2.5, label="y = 0 真:  L(z) = ln(1 + e$^{ z}$)")

    # 渐近线提示: 大 z 时 y=1 接近 0, y=0 线性趋近 z
    ax.plot(z, np.maximum(z, 0), color="#d62728", lw=0.8, linestyle=":", alpha=0.6)
    ax.plot(z, np.maximum(-z, 0), color="#1f77b4", lw=0.8, linestyle=":", alpha=0.6)
    ax.text(4.5, 4.7, "渐近线 y=z", color="#d62728", fontsize=9)
    ax.text(-5.8, 4.7, "渐近线 y=-z", color="#1f77b4", fontsize=9)

    # 中点
    ax.plot(0, np.log(2), "ko", markersize=6, zorder=5)
    ax.annotate("z=0 → p=0.5 → L=ln2", xy=(0, np.log(2)), xytext=(0.8, 1.5),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax.set_xlim(-6, 6); ax.set_ylim(0, 6.5)
    ax.set_xlabel("z  (logit, 直接送 sigmoid 之前)")
    ax.set_ylabel("L  (BCE 损失)")
    ax.set_title("图 3: Sigmoid + BCE = BCEWithLogitsLoss\n"
                 "PyTorch 内部就是直接用这个公式 (避开 sigmoid 数值溢出)")
    ax.legend(loc="upper center"); ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    draw_sigmoid(axes[0])
    draw_bce(axes[1])
    draw_sigmoid_plus_bce(axes[2])

    plt.tight_layout()
    savepath = "draw_sigmoid_BCE.png"
    plt.savefig(savepath, dpi=120, bbox_inches="tight")
    print(f"图已保存到 {savepath}")
    plt.show()


if __name__ == "__main__":
    main()
