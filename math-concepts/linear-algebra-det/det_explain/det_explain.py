# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
用四张图给初学者讲清楚: 为什么 2x2 矩阵的行列式是 ad - bc ?

    A = | a  b |
        | c  d |

一句话直觉:
    把矩阵看成一个"空间变换", 行列式就是它把面积放大/缩小的倍数。
    单位正方形 (面积=1) 经过 A 变换后, 变成一个平行四边形,
    这个平行四边形的 (带符号) 面积 = ad - bc, 这就是行列式。

四张图分别回答四个问题:
    图 1  变换在做什么?      单位正方形 → 平行四边形, 面积从 1 变成 |ad-bc|
    图 2  ad-bc 从哪来?      用"大矩形 - 周边碎片"把 ad-bc 一块块拼出来 (核心)
    图 3  负号是什么意思?    ad-bc<0 表示定向翻转 (像照镜子, 左右手互换)
    图 4  退化情况           ad-bc=0 表示平行四边形被压扁成一条线, 面积=0

运行 (依赖已通过 PEP 723 内联元数据声明, uv 会自动装好):
    uv run det_explain.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 全局使用一组具体数字, 让初学者能对着算 ----
#   两个列向量:  u = (a, c) = 第一列,  v = (b, d) = 第二列
#   A 的列向量正是"单位正方形的两条边被变换后落到哪里"
A, B, C, D = 3, 1, 1, 2          # a=3, b=1, c=1, d=2
DET = A * D - B * C              # = 3*2 - 1*1 = 5

U = np.array([A, C])             # 第一列 (i 帽 -> 这里)
V = np.array([B, D])             # 第二列 (j 帽 -> 这里)

CU = "#1f77b4"   # u 的颜色 (蓝)
CV = "#d62728"   # v 的颜色 (红)
CFILL = "#ffd166"


def arrow(ax, start, end, color, label=None, lw=2.5):
    """画一个带箭头的向量。"""
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                        color=color, lw=lw, zorder=5)
    ax.add_patch(a)
    if label:
        mid = (np.array(start) + np.array(end)) / 2
        ax.annotate(label, mid, color=color, fontsize=12, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points", zorder=6)


def setup(ax, xlim, ylim, title):
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)


# ============================================================
# 图 1: 变换在做什么 —— 单位正方形 → 平行四边形
# ============================================================
def fig1_transform(ax):
    # 原始单位正方形 (浅灰虚线)
    unit = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], closed=True,
                   fill=True, facecolor="lightgray", alpha=0.4,
                   edgecolor="gray", linestyle="--", lw=1.5)
    ax.add_patch(unit)
    ax.text(0.5, 0.5, "原正方形\n面积 = 1", ha="center", va="center",
            fontsize=9, color="dimgray")

    # 变换后的平行四边形: 由 u, v 张成
    para = Polygon([(0, 0), U, U + V, V], closed=True,
                   fill=True, facecolor=CFILL, alpha=0.55,
                   edgecolor="orange", lw=2)
    ax.add_patch(para)

    arrow(ax, (0, 0), U, CU, r"$u=(a,c)=(3,1)$  第一列")
    arrow(ax, (0, 0), V, CV, r"$v=(b,d)=(1,2)$  第二列")

    cx, cy = (U + V) / 2
    ax.text(cx, cy, f"新面积 = |ad-bc| = {DET}", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#8a5a00")

    setup(ax, (-0.6, 4.6), (-0.6, 3.6),
          "图1  矩阵 = 空间变换: 正方形被拉成平行四边形\n行列式 = 面积放大倍数 (这里 1 → 5)")


# ============================================================
# 图 2: 核心! 用"大矩形 - 周边碎片"证明 ad - bc
# ============================================================
def fig2_proof(ax):
    """
    把平行四边形塞进一个 (a+b) x (c+d) 的大矩形里。
        大矩形面积       = (a+b)(c+d) = ac + ad + bc + bd
        减去周边 6 块碎片 = 2 个三角形(u) + 2 个三角形(v) + 2 个小矩形(bc)
        剩下             = ad - bc   <- 就是平行四边形
    """
    w, h = A + B, C + D            # 大矩形 宽 x 高 = 4 x 3

    # 大矩形边框
    rect = Polygon([(0, 0), (w, 0), (w, h), (0, h)], closed=True,
                   fill=False, edgecolor="black", lw=2, linestyle="-")
    ax.add_patch(rect)
    ax.text(w / 2, h + 0.15, f"大矩形 = (a+b)(c+d) = {w}×{h} = {w*h}",
            ha="center", fontsize=10, fontweight="bold")

    # 中间的平行四边形 (要求的目标)
    para = Polygon([(0, 0), U, U + V, V], closed=True,
                   fill=True, facecolor=CFILL, alpha=0.75,
                   edgecolor="orange", lw=2, zorder=4)
    ax.add_patch(para)
    cx, cy = (U + V) / 2
    ax.text(cx, cy, f"ad-bc\n={DET}", ha="center", va="center",
            fontsize=11, fontweight="bold", color="#8a5a00", zorder=5)

    # ---- 周边 6 块要减掉的碎片 ----
    # 平行四边形只碰到大矩形的左下角(0,0)和右上角(w,h)=u+v。
    # 剩下的部分, 被下方边(0,0)->u->u+v 与 上方边(0,0)->v->u+v 切成两半,
    # 每半各是 "三角形 ac/2 + 矩形 bc + 三角形 bd/2", 两半 180° 中心对称。
    tri = dict(alpha=0.35, lw=1)
    # 两个 "ac/2 三角形" (蓝): 下半一个, 上半一个(中心对称)
    ax.add_patch(Polygon([(0, 0), (A, 0), U], closed=True,
                         facecolor=CU, edgecolor=CU, **tri))          # 下: (0,0)(a,0)(a,c)
    ax.add_patch(Polygon([U + V, (B, h), V], closed=True,
                         facecolor=CU, edgecolor=CU, **tri))          # 上: (w,h)(b,h)(b,d)
    # 两个 "bd/2 三角形" (红): 下半一个, 上半一个
    ax.add_patch(Polygon([U, (w, C), U + V], closed=True,
                         facecolor=CV, edgecolor=CV, **tri))          # 下: (a,c)(w,c)(w,h)
    ax.add_patch(Polygon([V, (0, D), (0, 0)], closed=True,
                         facecolor=CV, edgecolor=CV, **tri))          # 上: (b,d)(0,d)(0,0)
    # 两个小矩形 (面积各 = b*c, 绿): 下半一个, 上半一个
    ax.add_patch(Polygon([(A, 0), (w, 0), (w, C), (A, C)], closed=True,
                         facecolor="green", edgecolor="green", alpha=0.3, lw=1))
    ax.add_patch(Polygon([(0, D), (B, D), (B, h), (0, h)], closed=True,
                         facecolor="green", edgecolor="green", alpha=0.3, lw=1))

    # 碎片标注
    ax.text(A - 0.55, 0.28, "ac/2", color=CU, fontsize=9, ha="center")
    ax.text(B + 0.35, h - 0.3, "ac/2", color=CU, fontsize=9, ha="center")
    ax.text(0.28, D - 0.55, "bd/2", color=CV, fontsize=9, ha="center")
    ax.text(w - 0.28, C + 0.55, "bd/2", color=CV, fontsize=9, ha="center")
    ax.text((A + w) / 2, C / 2, "bc", color="green", fontsize=9,
            ha="center", va="center", fontweight="bold")
    ax.text((0 + B) / 2, (D + h) / 2, "bc", color="green", fontsize=9,
            ha="center", va="center", fontweight="bold")

    setup(ax, (-0.6, 4.8), (-0.6, 3.9),
          "图2 (核心)  大矩形 减掉 周边6块碎片 = 中间平行四边形\n"
          "(a+b)(c+d) - 2·(ac/2) - 2·(bd/2) - 2·bc = ad - bc")


# ============================================================
# 图 3: 负号的含义 —— 定向翻转
# ============================================================
def fig3_sign(ax):
    # 正定向: 从 u 转到 v 是逆时针 (det > 0)
    para1 = Polygon([(0, 0), U, U + V, V], closed=True,
                    fill=True, facecolor="#a8dadc", alpha=0.6, edgecolor="teal", lw=2)
    ax.add_patch(para1)
    arrow(ax, (0, 0), U, CU, "u")
    arrow(ax, (0, 0), V, CV, "v")
    # 逆时针弧线示意
    ax.annotate("", xy=V * 0.6, xytext=U * 0.6,
                arrowprops=dict(arrowstyle="->", color="teal",
                                connectionstyle="arc3,rad=0.4", lw=1.8))
    ax.text(1.3, 0.55, "u→v 逆时针\ndet>0", color="teal", fontsize=9)

    # 交换两列 (等价于翻转): det 变号
    para2 = Polygon([(0, 0), V, U + V, U], closed=True,
                    fill=True, facecolor="#f4a5a5", alpha=0.4, edgecolor="crimson",
                    lw=2, linestyle="--")
    ax.add_patch(para2)
    ax.text(2.6, 2.4, "若交换两列 (镜像)\nad-bc → -(ad-bc)\ndet<0",
            color="crimson", fontsize=9)

    setup(ax, (-0.6, 4.0), (-0.6, 3.6),
          "图3  正负号 = 定向: det>0 保持左右手,\ndet<0 表示空间被翻转 (照镜子)")


# ============================================================
# 图 4: 退化情况 —— det = 0
# ============================================================
def fig4_degenerate(ax):
    # 两个共线的向量: u=(2,1), v=(4,2) => det = 2*2-4*1 = 0
    u = np.array([2, 1])
    v = np.array([4, 2])
    det0 = u[0] * v[1] - v[0] * u[1]

    # "平行四边形" 被压成一条线
    ax.plot([0, v[0]], [0, v[1]], color=CFILL, lw=8, alpha=0.7, solid_capstyle="round")
    arrow(ax, (0, 0), u, CU, "u=(2,1)")
    arrow(ax, (0, 0), v, CV, "v=(4,2)")

    ax.text(2.2, 0.55, f"两向量共线\n面积被压扁\nad-bc = {det0}",
            fontsize=10, color="#333", fontweight="bold")

    setup(ax, (-0.6, 4.8), (-0.6, 3.0),
          "图4  det=0: 两列向量共线, 平行四边形塌成一条线\n"
          "面积=0 → 变换不可逆, 空间被降维压扁")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1_transform(axes[0, 0])
    fig2_proof(axes[0, 1])
    fig3_sign(axes[1, 0])
    fig4_degenerate(axes[1, 1])

    fig.suptitle("为什么 2×2 行列式 = ad - bc ?  (行列式 = 单位正方形被变换后的带符号面积)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "det_explain.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"已保存: {out}")

    # 顺手在终端把账算给你看
    print("\n--- 用具体数字验证 图2 的切割证明 ---")
    print(f"矩阵 A = [[{A},{B}],[{C},{D}]]")
    print(f"大矩形    (a+b)(c+d) = {(A+B)*(C+D)}")
    print(f"减 2 个 ½ac 三角形    = {2*0.5*A*C}")
    print(f"减 2 个 ½bd 三角形    = {2*0.5*B*D}")
    print(f"减 2 个 bc  小矩形    = {2*B*C}")
    print(f"剩下平行四边形面积    = {(A+B)*(C+D) - 2*0.5*A*C - 2*0.5*B*D - 2*B*C}")
    print(f"直接算 ad - bc        = {DET}   ✓ 两者相等")

    plt.show()


if __name__ == "__main__":
    main()
