# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""
用一个真实（不是玩具）的应用给初学者讲清楚: 特征值 / 特征向量到底是什么?

    定义:   A v = λ v
    一句话: 大多数向量被矩阵 A 一乘, 既会被"拉伸"又会被"转向";
            但有那么几个特殊方向 v, 乘完之后【方向不变, 只是长度变了 λ 倍】。
            这个特殊方向就是"特征向量", 那个倍数 λ 就是"特征值"。

真实应用: Google 的 PageRank
    2000 年前后, 全世界的网页链接构成一个巨大的矩阵。
    "哪个网页最重要"这个问题, 数学上的答案就是:
        这个矩阵【特征值 = 1】所对应的那个特征向量。
    Google 靠这一个特征向量起家 —— 这是特征值改变世界最有名的例子, 绝非玩具。

四张图分别回答四件事:
    图1  特征向量是什么?   大多数向量会转向, 特征向量只被拉伸 (A v = λ v)
    图2  怎么把它算出来?   幂迭代: 反复乘 A, 任意向量都会"倒向"最大的那个特征向量
    图3  真实的应用长啥样?  5 个网页互相链接, 构成一张真实网络
    图4  答案是什么?       网页重要度 = 特征值1 对应的特征向量 (= PageRank)

运行 (依赖已通过 PEP 723 内联元数据声明, uv 会自动装好):
    uv run eigen_pagerank.py
    # 若 uv 拉取 PyPI 报证书错误, 加 --native-tls: uv run --native-tls eigen_pagerank.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 概念部分用的 2x2 矩阵 (对称, 方便手算, 特征值/向量都是整齐的)
#   A = | 2  1 |    特征值 λ1 = 3, 特征向量 (1, 1)   —— 沿此方向拉伸 3 倍
#       | 1  2 |    特征值 λ2 = 1, 特征向量 (1, -1)  —— 沿此方向长度不变
# ============================================================
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])

C_ORD = "#9aa0a6"   # 普通向量 (灰)
C_IN = "#1f77b4"    # 输入 (蓝)
C_OUT = "#d62728"   # 输出 A·v (红)
C_EIG = "#2ca02c"   # 特征方向 (绿)
CFILL = "#ffd166"


def arrow(ax, start, end, color, label=None, lw=2.5, ls="-", z=5):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                        color=color, lw=lw, linestyle=ls, zorder=z)
    ax.add_patch(a)
    if label:
        ax.annotate(label, end, color=color, fontsize=10, fontweight="bold",
                    xytext=(6, 6), textcoords="offset points", zorder=z + 1)


def setup(ax, xlim, ylim, title):
    ax.set_title(title, fontsize=12.5, pad=8)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)


# ============================================================
# 图 1: 特征向量是什么 —— 大多数向量转向, 特征向量只被拉伸
# ============================================================
def fig1_concept(ax):
    # (a) 一个普通方向: 乘完 A 会"转向"
    v_ord = np.array([1.0, 0.0])
    Av_ord = A @ v_ord                       # = (2, 1), 明显转了向
    arrow(ax, (0, 0), v_ord, C_IN)
    arrow(ax, (0, 0), Av_ord, C_OUT)
    ax.text(1.02, -0.28, "普通向量 v=(1,0)", color=C_IN, fontsize=10, fontweight="bold")
    ax.text(2.05, 0.75, "A·v=(2,1)\n转向了!", color=C_OUT, fontsize=10, fontweight="bold")

    # (b) 特征方向 1: v=(1,1), A·v=(3,3) —— 方向不变, 长度 ×3
    v1 = np.array([1.0, 1.0])
    Av1 = A @ v1                             # = (3, 3)
    arrow(ax, (0, 0), Av1, C_OUT, lw=3)
    arrow(ax, (0, 0), v1, C_EIG, lw=3)
    ax.text(2.55, 3.12, "A·v=(3,3)=3·v", color=C_OUT, fontsize=10, fontweight="bold")
    ax.text(0.05, 1.35, "特征向量 v=(1,1)\nλ=3 (拉长 3 倍)", color=C_EIG,
            fontsize=10, fontweight="bold")

    # (c) 特征方向 2: v=(1,-1), A·v=(1,-1) —— 完全不变, λ=1
    v2 = np.array([1.0, -1.0])
    Av2 = A @ v2                             # = (1, -1)
    arrow(ax, (0, 0), v2, C_EIG, lw=3)
    ax.text(1.05, -1.05, "特征向量 v=(1,-1),  λ=1 (原地不动)", color=C_EIG,
            fontsize=10, fontweight="bold")

    ax.text(-2.7, 2.9, "绿色 = 特征方向: 乘完 A 只变长, 不转向\n"
                       "蓝→红 = 普通方向: 乘完 A 转了向",
            fontsize=9.5, color="#333",
            bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060"))

    setup(ax, (-3.0, 3.6), (-1.8, 3.6),
          "图1  A·v = λ·v 是什么?\n特征向量 = 乘完 A【只被拉伸、不转向】的特殊方向")


# ============================================================
# 图 2: 幂迭代 —— 反复乘 A, 任何向量都会倒向"最大特征值"的方向
#        (这正是真实世界算 PageRank 的办法)
# ============================================================
def fig2_power(ax):
    v = np.array([1.0, -0.35])              # 随便挑一个起始方向
    dirs = [v / np.linalg.norm(v)]
    for _ in range(6):
        v = A @ v
        dirs.append(v / np.linalg.norm(v))  # 每步归一化, 只看"方向"

    n = len(dirs)
    for i, d in enumerate(dirs):
        frac = i / (n - 1)
        color = plt.cm.viridis(frac)
        lw = 1.6 + 2.4 * frac
        arrow(ax, (0, 0), d, color, lw=lw, z=4 + i)

    # 真正的主特征方向 (1,1) 归一化
    tgt = np.array([1.0, 1.0]) / np.sqrt(2)
    arrow(ax, (0, 0), tgt, C_EIG, "主特征向量 (1,1)", lw=1.2, ls="--", z=3)

    ax.text(-1.15, 1.15, "起点 (随便挑)", fontsize=9, color="#440154")
    ax.text(0.55, 0.28, "每乘一次 A\n就更靠近\n最大特征方向", fontsize=9,
            color="#21908c", fontweight="bold")

    setup(ax, (-1.3, 1.5), (-0.8, 1.4),
          "图2  怎么把它算出来? 幂迭代\n反复乘 A → 任意向量都「倒向」最大特征值的方向")


# ============================================================
# 图 3 + 图 4 用的真实应用: PageRank
#   5 个网页, 谁链接谁 (from -> to)。这是一张真实的有向网络。
# ============================================================
PAGES = ["A", "B", "C", "D", "E"]
# 邻接: LINKS[i] = i 指向的页面下标列表
LINKS = {
    0: [1, 2, 3],   # A -> B, C, D
    1: [0, 3],      # B -> A, D
    2: [0],         # C -> A          (C 把全部"投票"都投给了 A)
    3: [1, 2],      # D -> B, C
    4: [0, 3],      # E -> A, D       (没有任何人链接到 E)
}
DAMP = 0.85         # Google 的阻尼系数 (随机上网者有 15% 概率随机跳转)


def build_google_matrix():
    """构造列随机的 Google 矩阵 G, 其【特征值=1】的特征向量就是 PageRank。"""
    n = len(PAGES)
    M = np.zeros((n, n))
    for src, outs in LINKS.items():
        if outs:
            for dst in outs:
                M[dst, src] = 1.0 / len(outs)   # 列 src 平均分给它指向的页面
        else:
            M[:, src] = 1.0 / n                 # 没有出链的页面 -> 均匀跳转
    G = DAMP * M + (1 - DAMP) / n * np.ones((n, n))
    return G


def node_positions(n):
    """把 n 个节点均匀摆在一个圆上。"""
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def fig3_graph(ax, rank):
    pos = node_positions(len(PAGES))
    R = 0.20  # 节点半径

    # 先画有向边 (带一点弧度, 避免双向边重叠)
    for src, outs in LINKS.items():
        for dst in outs:
            p0, p1 = pos[src], pos[dst]
            d = p1 - p0
            L = np.linalg.norm(d)
            u = d / L
            start = p0 + u * R
            end = p1 - u * R
            ax.add_patch(FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=14,
                color="#888", lw=1.4, zorder=2,
                connectionstyle="arc3,rad=0.15"))

    # 再画节点: 圆的大小 = PageRank 大小 (直观!)
    order = np.argsort(-rank)
    top = order[0]
    for i, (x, y) in enumerate(pos):
        r = 0.12 + 0.9 * rank[i]
        fc = "#ff7f0e" if i == top else "#a8d0e6"
        ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor="#333",
                            lw=1.8, zorder=3))
        ax.text(x, y, PAGES[i], ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=4)

    ax.text(0, -1.55, "箭头 = 链接 (谁指向谁)   圆越大 = PageRank 越高",
            ha="center", fontsize=9.5, color="#333")
    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.8, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("图3  真实应用: 5 个网页互相链接构成的网络\n"
                 "谁最重要? = 谁被「重要的页面」指向", fontsize=12.5, pad=8)


def fig4_result(ax, rank, lam):
    order = np.argsort(-rank)
    names = [PAGES[i] for i in order]
    vals = rank[order]
    colors = ["#ff7f0e" if k == 0 else "#a8d0e6" for k in range(len(order))]

    bars = ax.bar(names, vals, color=colors, edgecolor="#333", lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006,
                f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("PageRank (重要度)", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.text(0.98, 0.95,
            f"这一列 = Google 矩阵 G\n特征值 λ={lam:.4f}≈1 对应的特征向量\n"
            f"(G·r = 1·r, 归一化后加起来=1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#eaf6ea", ec="#8ac48a"))
    ax.text(0.02, 0.95,
            "A 排第1: 被多个页面指向,\n且 C 把全部票都投给 A。\n"
            "E 垫底: 没有任何人链接它。",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            color="#555")
    ax.set_title("图4  答案 = 特征值1 的特征向量\n"
                 "这就是 Google 给网页排序的 PageRank", fontsize=12.5, pad=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)


def compute_pagerank():
    """用两种办法算 PageRank, 互相印证: (1) 直接特征分解  (2) 幂迭代。"""
    G = build_google_matrix()

    # 办法1: 特征分解, 取特征值最接近 1 的那个
    vals, vecs = np.linalg.eig(G)
    idx = np.argmin(np.abs(vals - 1.0))
    lam = vals[idx].real
    r_eig = np.abs(vecs[:, idx].real)
    r_eig = r_eig / r_eig.sum()

    # 办法2: 幂迭代 (真实工程里就是这么算的, 因为网页矩阵太大)
    n = len(PAGES)
    r = np.ones(n) / n
    for _ in range(100):
        r = G @ r
    r_power = r / r.sum()

    return G, lam, r_eig, r_power


def main():
    G, lam, r_eig, r_power = compute_pagerank()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig1_concept(axes[0, 0])
    fig2_power(axes[0, 1])
    fig3_graph(axes[1, 0], r_eig)
    fig4_result(axes[1, 1], r_eig, lam)

    fig.suptitle("特征值到底有什么用? —— 从 A·v=λ·v 到 Google 的 PageRank",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = "eigen_pagerank.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"已保存: {out}\n")

    # 把账在终端算给你看
    print("=== 概念: A = [[2,1],[1,2]] 的特征值/特征向量 ===")
    w, V = np.linalg.eig(A)
    for k in range(2):
        lam_k = w[k].real
        v_k = V[:, k].real
        print(f"  λ = {lam_k:.0f},  特征向量 v ≈ {np.round(v_k, 3)}"
              f"   验证 A·v = {np.round(A @ v_k, 3)},  λ·v = {np.round(lam_k * v_k, 3)}  ✓")

    print("\n=== 应用: PageRank (5 个网页) ===")
    print("  链接关系:  A→B,C,D   B→A,D   C→A   D→B,C   E→A,D")
    print(f"  Google 矩阵最大特征值 λ = {lam:.6f}  (理论上正好 = 1)")
    order = np.argsort(-r_eig)
    print("\n  排名   网页    PageRank(特征分解)   PageRank(幂迭代)")
    for rk, i in enumerate(order, 1):
        print(f"   {rk}     {PAGES[i]}        {r_eig[i]:.4f}"
              f"               {r_power[i]:.4f}")
    print(f"\n  两种算法结果一致 (最大差异 {np.abs(r_eig - r_power).max():.2e}) ✓")
    print("  结论: 网页重要度 = Google矩阵「特征值1」对应的特征向量。")

    plt.show()


if __name__ == "__main__":
    main()
