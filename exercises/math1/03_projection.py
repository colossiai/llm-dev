"""
03 - 投影 (Projection)

把向量 a 沿着向量 b 的方向"投影"下来:
    proj_b(a) = (a · b / b · b) * b

直观理解: 想象光从垂直于 b 的方向照下来, a 在 b 上的"影子"就是 proj_b(a)。
任何向量 a 都可以唯一分解成:
    a = proj_b(a)   (与 b 平行的部分)
      + a_perp       (与 b 垂直的部分)

点积的几何含义就是: a 在 b 方向上的影子长度 × |b|。
"""

import matplotlib.pyplot as plt
import torch

# matplotlib 中文显示 (macOS)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def project(a, b):
    """向量 a 在向量 b 方向上的投影, 返回与 b 同向的向量。"""
    return (torch.dot(a, b) / torch.dot(b, b)) * b


def main():
    a = torch.tensor([3.0, 4.0])
    b = torch.tensor([5.0, 0.0])    # 沿 x 轴方向, 便于直观

    proj = project(a, b)
    perp = a - proj                  # 垂直分量

    print("a         =", a.tolist())
    print("b         =", b.tolist())
    print("proj_b(a) =", proj.tolist())   # 应当是 [3, 0]
    print("a_perp    =", perp.tolist())   # 应当是 [0, 4]

    # 验证 1: a_perp 与 b 垂直 → 点积接近 0
    print("\n验证 a_perp · b =", torch.dot(perp, b).item(), "  (应 ≈ 0)")
    # 验证 2: proj + perp == a
    print("验证 proj + perp =", (proj + perp).tolist(), "  (应 = a)")

    # =========================================================
    # 可视化:  b (绿)、a (蓝)、proj_b(a) (橙)、a_perp (红虚线)
    # =========================================================
    fig, ax = plt.subplots(figsize=(7, 7))
    origin = torch.zeros(2)

    def arrow(vec, color, label, start=origin):
        ax.quiver(start[0], start[1], vec[0], vec[1],
                  angles="xy", scale_units="xy", scale=1,
                  color=color, label=label)

    arrow(b,    "tab:green",  "b")
    arrow(a,    "tab:blue",   "a")
    arrow(proj, "tab:orange", "proj_b(a)")
    # a_perp 画在 proj 末端 → 末端正好落在 a 上, 构成直角三角形
    ax.plot([proj[0], a[0]], [proj[1], a[1]],
            color="tab:red", linestyle="--", label="a_perp")

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.legend(loc="upper right")
    ax.set_title("向量 a 在 b 方向上的投影")

    plt.tight_layout()
    plt.savefig("projection.png", dpi=120)
    print("\n图已保存到 projection.png")
    plt.show()


if __name__ == "__main__":
    main()
