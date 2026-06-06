"""
02 - 矩阵乘法 · 线性变换 · 基底

核心直觉: 矩阵乘法 = 对向量做"空间变换"。
本脚本演示:
  - 矩阵 × 向量、矩阵 × 矩阵 的维度规则
  - 旋转、缩放、剪切 三种经典 2D 线性变换
  - "矩阵的列 = 变换后的新基底" 这一核心直觉
"""

import math

import matplotlib.pyplot as plt
import torch

# 让 matplotlib 能正常显示中文 (macOS)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def apply(M, points):
    """对一组点 (N, 2) 应用 2x2 矩阵 M, 返回变换后的 (N, 2)。"""
    # points: (N, 2)  → 转置 (2, N)  → 左乘 M 得 (2, N)  → 再转回 (N, 2)
    return (M @ points.T).T


def main():
    # =========================================================
    # 1. 维度规则: (m, k) @ (k, n) → (m, n)
    #    A 的"列数" 必须等于 B 的"行数"
    # =========================================================
    A = torch.randn(2, 3)
    B = torch.randn(3, 4)
    print("A.shape       =", A.shape)        # (2, 3)
    print("B.shape       =", B.shape)        # (3, 4)
    print("(A @ B).shape =", (A @ B).shape)  # (2, 4)

    # =========================================================
    # 2. 矩阵 × 向量: 把一个向量"搬"到一个新位置
    # =========================================================
    M = torch.tensor([[2.0, 0.0],
                      [0.0, 3.0]])           # x 拉 2 倍, y 拉 3 倍
    v = torch.tensor([1.0, 1.0])
    print("\nM @ v =", M @ v)                # [2, 3]

    # =========================================================
    # 3. "矩阵的列 = 变换后的新基底"
    #    把标准基 e1=[1,0]、e2=[0,1] 经过 M 变换, 结果就是 M 的两列。
    # =========================================================
    e1 = torch.tensor([1.0, 0.0])
    e2 = torch.tensor([0.0, 1.0])
    print("\n--- 基底 (basis) ---")
    print("M @ e1 =", (M @ e1).tolist(), "  ← 等于 M 的第 1 列")
    print("M @ e2 =", (M @ e2).tolist(), "  ← 等于 M 的第 2 列")
    print("M 的两列分别是:", M[:, 0].tolist(), M[:, 1].tolist())

    # =========================================================
    # 4. 三种经典线性变换, 作用在单位正方形 4 个角上
    # =========================================================
    square = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],   # 回到起点, 形成闭合图形
    ])

    theta = math.radians(30)
    R = torch.tensor([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])   # 旋转 30°
    S = torch.tensor([[1.5, 0.0],
                      [0.0, 0.5]])                            # 缩放
    H = torch.tensor([[1.0, 1.0],
                      [0.0, 1.0]])                            # 剪切

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, M_t, title in [
        (axes[0], torch.eye(2), "原图 (单位正方形)"),
        (axes[1], R,            "旋转 30°"),
        (axes[2], S,            "缩放 (1.5, 0.5)"),
        (axes[3], H,            "剪切"),
    ]:
        pts = apply(M_t, square)
        ax.plot(pts[:, 0], pts[:, 1], "-o")
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, 3)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("matrix_transform.png", dpi=120)
    print("\n图已保存到 matrix_transform.png")
    plt.show()


if __name__ == "__main__":
    main()
