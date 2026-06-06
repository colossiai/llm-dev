"""
02 - 为什么梯度代表"上山的斜度"?

回忆导数: f'(x) 告诉你"x 往右挪一点, f 涨多少"
        f'(x) > 0 → 函数在上升; f'(x) < 0 → 函数在下降; |f'(x)| 越大 → 越陡。

推广到多元: 梯度 ∇f(w, b) = [∂f/∂w, ∂f/∂b]
  - 这是个向量, 既有方向 也有 长度。
  - 方向: 在 (w, b) 这点, 朝哪个方向走一小步 f 涨得最快 → 就是它指的方向。
  - 长度: 那个最快方向上 f 的上升速率 = 最陡的坡度。
  - 所以 "梯度 = 当前位置上山最陡的方向 + 斜率"。

下面: 还是上一节的 loss 山, 我们在地形上撒一堆点,
      用 PyTorch autograd 算出每个点的 ∇loss, 把它当箭头画出来。
      你会看到所有箭头都齐刷刷地指向山顶 (远离山谷底)。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    torch.manual_seed(0)
    x_data = torch.linspace(-2, 2, 50)
    y_data = 2 * x_data + 1 + 0.1 * torch.randn_like(x_data)

    def loss_fn(w, b):
        y_pred = w * x_data + b
        return ((y_pred - y_data) ** 2).mean()

    # =========================================================
    # 1. 在一个点上手算 vs autograd, 验证"梯度就是斜率"
    #    loss(w, b) = mean((w·x + b - y)²)
    #    ∂loss/∂w = 2·mean( x · (w·x + b - y) )
    #    ∂loss/∂b = 2·mean(     w·x + b - y   )
    # =========================================================
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    loss = loss_fn(w, b)
    loss.backward()

    with torch.no_grad():
        # err = 残差 = y_pred - y_data, 即 (w·x + b) - y
        # 此处 w=0, b=0, 所以 y_pred = 0·x + 0 = 0
        # 写成 "0 * x_data + 0 - y_data" 是为了跟公式 "w·x + b - y" 一一对应,
        # 让读者一眼看出 "把 w、b 代入 0 之后" 的样子; 化简后其实就是 -y_data
        err = 0 * x_data + 0 - y_data
        manual_dw = 2 * (x_data * err).mean().item()   # 对应 ∂loss/∂w = 2·mean(x · err)
        manual_db = 2 * err.mean().item()              # 对应 ∂loss/∂b = 2·mean(err)

    print("--- 在 (w=0, b=0) 处验证 ---")
    print(f"autograd: ∇loss = [{w.grad.item():.4f}, {b.grad.item():.4f}]")
    print(f"手算:     ∇loss = [{manual_dw:.4f}, {manual_db:.4f}]")
    print("→ 完全一致。这两个数字就是当前点 'w 方向斜率' 和 'b 方向斜率'")
    print("→ 都是负的, 因为山谷底 (2, 1) 在 (0, 0) 的右上方, 朝右上走 loss 是降的,")
    print("  所以 '上山最陡' (= 梯度方向) 自然指向左下 → 两个分量都是负数")

    # =========================================================
    # 2. 在 (w, b) 平面网格上算每个点的梯度, 当箭头画出来
    # =========================================================
    ws = torch.linspace(-2, 6, 80)
    bs = torch.linspace(-3, 5, 80)
    W, B = torch.meshgrid(ws, bs, indexing="ij")
    L = torch.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            L[i, j] = loss_fn(W[i, j], B[i, j])

    # 箭头用粗一点的网格, 不然画面糊掉
    aw = torch.linspace(-2, 6, 14)
    ab = torch.linspace(-3, 5, 14)
    AW, AB = torch.meshgrid(aw, ab, indexing="ij")
    GW = torch.zeros_like(AW)
    GB = torch.zeros_like(AB)
    for i in range(AW.shape[0]):
        for j in range(AW.shape[1]):
            wi = AW[i, j].clone().requires_grad_(True)
            bi = AB[i, j].clone().requires_grad_(True)
            loss_fn(wi, bi).backward()
            GW[i, j] = wi.grad
            GB[i, j] = bi.grad

    # =========================================================
    # 3. 画图: 等高线 + 梯度箭头 (指向上山方向)
    # =========================================================
    fig, ax = plt.subplots(figsize=(9, 7))
    cs = ax.contour(W.numpy(), B.numpy(), L.numpy(), levels=25, cmap="terrain")
    ax.clabel(cs, inline=True, fontsize=7)

    # 注意: quiver 画的就是梯度向量 (GW, GB), 朝向 "上山"
    ax.quiver(AW.numpy(), AB.numpy(), GW.numpy(), GB.numpy(),
              color="red", alpha=0.7, scale=400, width=0.003,
              label="∇loss 箭头 (上山最陡方向)")

    ax.plot(2, 1, "g*", markersize=20, label="山谷底 (w=2, b=1)")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("梯度场: 红箭头 = 当前点'上山最陡'的方向与坡度\n"
                 "(注意所有箭头都背离山谷底, 越远越长 = 越陡)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("02_gradient_is_uphill_slope.png", dpi=120)
    print("\n图已保存到 02_gradient_is_uphill_slope.png")
    plt.show()


if __name__ == "__main__":
    main()
