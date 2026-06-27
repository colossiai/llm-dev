# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "statsmodels"]
# ///
"""
03 ARMA (AR + MA) —— "既受历史值影响, 也受历史冲击影响"
=====================================================

一句话直觉:
    ARMA = 惯性 (AR) + 冲击余波 (MA) 合体。
    ARMA(p,q): y_t = c + Σ φ_i·y_{t-i} + ε_t + Σ θ_j·ε_{t-j}

为什么要合体?
    现实里的平稳序列, 通常两种成分都有:
      - 有"惯性"  → 需要 AR
      - 有"冲击的余波" → 需要 MA
    单用 AR 或单用 MA 往往要很高阶才能拟合好; 合体后用很低的 (p,q) 就够了 (更省参数)。

怎么定阶 (p, q)? —— Box-Jenkins 三件套:
    | 工具  | 看 AR 阶数 p     | 看 MA 阶数 q     |
    | ---- | --------------- | --------------- |
    | ACF  | 拖尾 (不截断)      | lag=q 后截断      |
    | PACF | lag=p 后截断      | 拖尾 (不截断)      |
    口诀: "看 PACF 定 AR(p), 看 ACF 定 MA(q), 截断的那个给阶数。"
    纯 ARMA 两边都拖尾 → 实务里常用 AIC/BIC 网格搜索来选。

金融场景:
    平稳化后的收益率/价差序列, 常用低阶 ARMA 捕捉短期可预测成分;
    它也是 ARIMA、GARCH 均值方程的骨架。

这张图做什么:
    图 1  ARMA(1,1) 路径
    图 2  ACF + PACF 并排 → 学会看指纹
    图 3  用 AIC 网格搜索自动定阶, 并拟合
"""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2024)


def simulate_arma(ar: list[float], ma: list[float], n: int) -> np.ndarray:
    """模拟 ARMA。statsmodels 约定: ar/ma 系数含 lag0=1, AR 项要取负号。"""
    ar_poly = np.r_[1, -np.array(ar)]
    ma_poly = np.r_[1, np.array(ma)]
    return ArmaProcess(ar_poly, ma_poly).generate_sample(nsample=n, distrvs=rng.standard_normal)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- 图 1: ARMA(1,1) 路径 ----
    y = simulate_arma(ar=[0.7], ma=[0.5], n=400)
    axes[0].plot(y, lw=1.0)
    axes[0].set_title("图1: ARMA(1,1) 路径  φ=0.7, θ=0.5")
    axes[0].set_xlabel("时间 t")
    axes[0].set_ylabel("y_t")
    axes[0].grid(alpha=0.3)

    # ---- 图 2: ACF + PACF (这里把 PACF 画在同一坐标里对比) ----
    # 为了同图对比, 手动取值
    from statsmodels.tsa.stattools import acf, pacf

    lags = 20
    a = acf(y, nlags=lags)
    p = pacf(y, nlags=lags)
    x = np.arange(lags + 1)
    width = 0.4
    axes[1].bar(x - width / 2, a, width, label="ACF (定 MA 阶 q)", alpha=0.8)
    axes[1].bar(x + width / 2, p, width, label="PACF (定 AR 阶 p)", alpha=0.8)
    axes[1].axhline(0, color="k", lw=0.6)
    ci = 1.96 / np.sqrt(len(y))
    axes[1].axhline(ci, color="red", ls="--", alpha=0.4)
    axes[1].axhline(-ci, color="red", ls="--", alpha=0.4)
    axes[1].set_title("图2: ACF vs PACF —— 看哪个先截断定阶")
    axes[1].set_xlabel("滞后 lag")
    axes[1].legend()

    # ---- 图 3: AIC 网格搜索定阶 ----
    best = (None, np.inf)
    table = []
    for pp in range(3):
        for qq in range(3):
            try:
                aic = ARIMA(y, order=(pp, 0, qq)).fit().aic
            except Exception:
                aic = np.inf
            table.append((pp, qq, aic))
            if aic < best[1]:
                best = ((pp, qq), aic)
    (bp, bq), _ = best
    res = ARIMA(y, order=(bp, 0, bq)).fit()
    axes[2].plot(y[:150], label="数据", lw=1.0, alpha=0.7)
    axes[2].plot(res.fittedvalues[:150], label=f"ARMA({bp},{bq}) 拟合", lw=1.2)
    axes[2].set_title(f"图3: AIC 选出 ARMA({bp},{bq})  (真值 1,1)")
    axes[2].set_xlabel("时间 t")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.suptitle("ARMA = AR(惯性) + MA(冲击余波): 低阶就能拟合好平稳序列", fontsize=14)
    fig.tight_layout()
    out = __file__.replace(".py", ".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"图已保存: {out}")
    print("AIC 网格 (p, q, aic):")
    for pp, qq, aic in sorted(table, key=lambda r: r[2])[:4]:
        print(f"  ARMA({pp},{qq})  AIC={aic:.1f}")
    print(f"一句话: 看 PACF 定 AR(p), 看 ACF 定 MA(q); 纯 ARMA 用 AIC 选 → 选中 ({bp},{bq})。")


if __name__ == "__main__":
    main()
