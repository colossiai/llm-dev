"""
prerequisite/point_transform_intuition.py

把 `矩阵乘法几何直觉.md` 里的讨论可视化, 围绕同一个例子:

    M = [[2, 1],
         [0, 3]]    把点 v = (5, 6) 变换到 M@v = (16, 18)

4 个面板分别回答 4 个问题:
  1. 原点 (5, 6) 在旧基底下怎么分解?              → 5·e_1 + 6·e_2
  2. 新点 (16, 18) 在新基底下怎么分解?            → 5·Ae_1 + 6·Ae_2  (同一份"配方")
  3. 非对角元素 1 到底干了啥?                      → 把 (10,18) 推到 (16,18)
  4. y 坐标 = 18 和 6·Ae_2 的长度 6√10 是一回事吗?   → 不是, 一个是高度一个是斜距
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =================== 绘图小工具 ===================
def arrow(ax, vec, color, label=None, start=(0.0, 0.0), width=0.012):
    ax.quiver(start[0], start[1], float(vec[0]), float(vec[1]),
              angles="xy", scale_units="xy", scale=1,
              color=color, label=label, width=width)


def parallelogram(ax, v1, v2, color, alpha=0.12):
    """画由 v1, v2 张成的平行四边形 (背景填色)。"""
    pts = np.array([
        [0.0, 0.0],
        [float(v1[0]),               float(v1[1])],
        [float(v1[0] + v2[0]),       float(v1[1] + v2[1])],
        [float(v2[0]),               float(v2[1])],
    ])
    ax.add_patch(Polygon(pts, alpha=alpha, color=color))


def fmt(ax, title, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(title, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=7)


# =================== 主流程 ===================
def main():
    # 例子里的矩阵和点
    M       = torch.tensor([[2.0, 1.0],
                            [0.0, 3.0]])
    M_diag  = torch.tensor([[2.0, 0.0],     # 同样的对角, 砍掉非对角
                            [0.0, 3.0]])
    v       = torch.tensor([5.0, 6.0])
    e1      = torch.tensor([1.0, 0.0])
    e2      = torch.tensor([0.0, 1.0])

    Ae1     = M @ e1            # (2, 0)
    Ae2     = M @ e2            # (1, 3)  <-- 倾斜
    Mv      = M @ v             # (16, 18)
    Mv_diag = M_diag @ v        # (10, 18)

    print(f"v             = {v.tolist()}")
    print(f"Ae_1 = M @ e_1  = {Ae1.tolist()}")
    print(f"Ae_2 = M @ e_2  = {Ae2.tolist()}   ← 倾斜")
    print(f"M @ v         = {Mv.tolist()}      = 5·Ae_1 + 6·Ae_2")
    print(f"M_diag @ v    = {Mv_diag.tolist()}      (对角矩阵, 无串味)")
    print(f"差异          = {(Mv - Mv_diag).tolist()}   ← x 多出 6, 完全来自非对角元素 1")
    print(f"|6·Ae_2|        = {float(torch.norm(6 * Ae2)):.4f}   (= 6√10)")
    print(f"6·Ae_2 的 y 坐标 = {float((6 * Ae2)[1])}        ← 高度 ≠ 长度\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # -----------------------------------------------------------
    # Panel 1: 原点 (5, 6) = 5·e_1 + 6·e_2
    # -----------------------------------------------------------
    ax = axes[0, 0]
    parallelogram(ax, 5 * e1, 6 * e2, "tab:blue")
    arrow(ax, 5 * e1, "tab:red",   label="5·e_1 = (5, 0)")
    arrow(ax, 6 * e2, "tab:blue",  label="6·e_2 = (0, 6)", start=(5.0, 0.0))
    arrow(ax, v,      "tab:purple", label="v = 5e_1 + 6e_2 = (5, 6)")
    ax.plot(5, 6, "o", color="tab:purple", markersize=8)
    ax.text(5.2, 6.3, "(5, 6)", fontsize=10, color="tab:purple")
    fmt(ax, "1. 原点 (5, 6) 在标准基下: 走 5 步 e_1, 再走 6 步 e_2",
        xlim=(-1, 9), ylim=(-1, 9))

    # -----------------------------------------------------------
    # Panel 2: 新点 (16, 18) = 5·Ae_1 + 6·Ae_2  (同一份 5, 6 配方)
    # -----------------------------------------------------------
    ax = axes[0, 1]
    parallelogram(ax, 5 * Ae1, 6 * Ae2, "tab:orange")
    arrow(ax, 5 * Ae1, "tab:red",    label="5·Ae_1 = (10, 0)")
    arrow(ax, 6 * Ae2, "tab:blue",   label="6·Ae_2 = (6, 18)", start=(10.0, 0.0))
    arrow(ax, Mv,      "tab:purple", label="M@v = 5Ae_1 + 6Ae_2 = (16, 18)")
    # 也把单个 Ae_1、Ae_2 画出来当参考 (灰色细箭头)
    arrow(ax, Ae1, "gray", width=0.006)
    arrow(ax, Ae2, "gray", width=0.006)
    ax.text(2.1, 0.3, "Ae_1=(2,0)", fontsize=8, color="gray")
    ax.text(1.2, 3.1, "Ae_2=(1,3) ← 倾斜了", fontsize=8, color="gray")
    ax.plot(16, 18, "o", color="tab:purple", markersize=8)
    ax.text(16.3, 18.3, "(16, 18)", fontsize=10, color="tab:purple")
    fmt(ax, "2. 新点 = 同一份'5,6 配方', 但平行四边形被扭斜",
        xlim=(-1, 20), ylim=(-1, 22))

    # -----------------------------------------------------------
    # Panel 3: 对角矩阵 vs 非对角矩阵
    # -----------------------------------------------------------
    ax = axes[1, 0]
    arrow(ax, Mv_diag, "tab:cyan",
          label="diag([2,3]) @ v → (10, 18)   无串味")
    ax.plot(10, 18, "o", color="tab:cyan", markersize=10)
    ax.text(10.3, 18.8, "(10, 18)", fontsize=9, color="tab:cyan")

    arrow(ax, Mv, "tab:orange",
          label="[[2,1],[0,3]] @ v → (16, 18)   含非对角")
    ax.plot(16, 18, "o", color="tab:orange", markersize=10)
    ax.text(16.3, 18.8, "(16, 18)", fontsize=9, color="tab:orange")

    # 差异: 从 (10,18) 推到 (16,18) 的水平箭头
    ax.annotate("", xy=(16, 18), xytext=(10, 18),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2.5))
    ax.text(10.5, 18.6, "+6 (沿 x 漏出)", color="tab:red", fontsize=10,
            fontweight="bold")
    ax.text(10.5, 16.2, "= 6 份 Ae_2 × 各漏 1 到 x",
            color="tab:red", fontsize=9)

    fmt(ax, "3. 非对角元素 M[0,1]=1 把 (10,18) 推到 (16,18)",
        xlim=(-1, 22), ylim=(-1, 22))

    # -----------------------------------------------------------
    # Panel 4: y 坐标 ≠ 向量长度
    # -----------------------------------------------------------
    ax = axes[1, 1]
    six_Ae2 = 6 * Ae2     # (6, 18)
    arrow(ax, six_Ae2, "tab:blue", width=0.018)
    ax.plot(6, 18, "o", color="tab:blue", markersize=10)
    ax.text(6.3, 18.5, "6·Ae_2 = (6, 18)", fontsize=10, color="tab:blue",
            fontweight="bold")

    # 直角三角形的水平 / 竖直辅助线
    ax.plot([0, 6], [0, 0],  "k--", lw=0.8)
    ax.plot([6, 6], [0, 18], "k--", lw=0.8)
    # 右下角的直角小标记
    ax.plot([5.6, 6.0], [0, 0],   "k", lw=1.2)
    ax.plot([5.6, 5.6], [0, 0.4], "k", lw=1.2)

    length = float(torch.norm(six_Ae2))
    ax.text(3, -2,
            "x 分量 = 6 (水平漏出)",
            fontsize=10, color="tab:red", ha="center", fontweight="bold")
    ax.text(6.4, 9,
            "y 分量 = 18\n(向上高度)",
            fontsize=10, color="tab:green", fontweight="bold")
    ax.text(0.3, 13,
            f"长度 = √(36 + 324)\n     = √360 = 6√10\n     ≈ {length:.2f}\n(斜着走的总距离)",
            fontsize=10, color="tab:purple", fontweight="bold")

    fmt(ax, "4. 同一个 6·Ae_2: y 坐标=18, 长度=6√10≈18.97 (两件事!)",
        xlim=(-2, 11), ylim=(-4, 22))

    plt.tight_layout()
    plt.savefig("point_transform_intuition.png", dpi=120)
    print("图已保存: point_transform_intuition.png")

    print("\n--- 4 个面板对应的 4 条结论 ---")
    print("  1. 原点 (5, 6) = 5·e_1 + 6·e_2          (旧基底下的'配方')")
    print("  2. 新点 (16, 18) = 5·Ae_1 + 6·Ae_2      (同一份配方, 新基底)")
    print("  3. 非对角元素 1 = '旧 y 漏 1 到新 x' 的耦合系数")
    print("  4. y 坐标 (分量) ≠ 长度 (斜距), 只有纯竖直向量才相等")


if __name__ == "__main__":
    main()
