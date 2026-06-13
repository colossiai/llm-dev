# 随机过程 (Stochastic Process) — 做市视角讲解

## 一句话总结

> **随机过程 = 一族随时间演化的随机变量** `{X_t}_{t≥0}`。
> 在做市里，它建模两类东西：**价格** (`mid-price 的扩散`) 和 **订单流** (`订单到达的离散事件`)。
> 做市策略本质就是「在这两类随机过程的冲击下，动态调整买卖报价」。

---

## 直觉：为什么做市需要随机过程？

普通函数 `y = f(t)` 给你一条确定曲线。但市场不是确定的：
- 下一秒中间价是涨是跌？— 不知道，只知道**分布**
- 下一笔订单什么时候来？— 不知道，只知道**到达率**

随机过程提供的语言：**用概率分布刻画时间演化**。

> 类比：神经网络是「确定函数 + 学到的权重」；随机过程是「概率分布 + 时间维度」。
> 你已经会的 LSTM/Transformer 在做"给定历史序列预测下一个 token"；随机过程在做"给定历史路径，描述下一时刻状态的分布"。

---

## 知识地图（按做市重要性排序）

| 过程 | 用来建模 | 关键参数 | 文件 |
|---|---|---|---|
| **布朗运动 (Brownian Motion)** | 中间价的对称扩散（加性） | 漂移 μ、波动率 σ | `01_brownian_motion.py` |
| **几何布朗运动 (GBM)** | 中间价的乘性扩散（保证非负） | μ、σ | `01_brownian_motion.py` |
| **泊松过程 (Poisson)** | 订单到达的计数 | 到达率 λ | `02_poisson_process.py` |
| **复合泊松 (Compound Poisson)** | 订单到达 + 随机订单量 | λ + 量分布 | `02_poisson_process.py` |
| **Hawkes 过程** | 订单流的"自激发聚集" | λ₀, α, β | `03_hawkes_process.py` |
| **跳跃扩散 (Jump-Diffusion)** | 平时扩散 + 偶尔跳变（如新闻冲击） | GBM 参数 + 跳跃 λ_J | `04_jump_diffusion.py` |
| **综合：Avellaneda-Stoikov** | 完整做市策略 | 全部组合 | `05_market_making_demo.py` |

---

## 三类过程的根本区别（一表看懂）

| 过程类型 | 时间是否连续 | 状态是否连续 | 路径形状 | 例子 |
|---|---|---|---|---|
| 布朗运动 / GBM | 连续 | 连续 | 处处不光滑但连续 | mid-price |
| 泊松过程 | 连续 | 离散（整数） | 阶梯状跳变 | 订单计数 |
| 跳跃扩散 | 连续 | 连续 + 跳变 | 大部分连续 + 偶尔跳 | 含新闻的价格 |
| Hawkes | 连续 | 离散 | 阶梯，但聚集 | 高频订单流 |

---

## 做市映射：哪个过程对应哪个变量？

```
做市状态空间
├── 中间价 S_t          ← GBM（或 Jump-Diffusion）
├── 买单到达 N_t^buy    ← Poisson 或 Hawkes
├── 卖单到达 N_t^sell   ← Poisson 或 Hawkes
├── 订单量 V_i          ← Compound Poisson 的 V
└── 我的库存 q_t        ← 由订单到达累积驱动
```

**做市最优控制问题** = 「给定上述随机过程的动态，选 (买价, 卖价) 让效用最大」
→ 这就引出了 HJB 方程和 Avellaneda-Stoikov 解（见 `05_market_making_demo.py`）。

---

## 学习顺序建议

1. `01_brownian_motion.py` — **必须先看**，所有连续随机过程的基础
2. `02_poisson_process.py` — 离散事件建模，订单流的起点
3. `03_hawkes_process.py` — 在 Poisson 之上升级，理解"自激发"
4. `04_jump_diffusion.py` — 在 BM 之上加跳跃，理解肥尾
5. `05_market_making_demo.py` — 把上面全部组合成真实做市

---

## 怎么运行

每个脚本都用了 **PEP 723 inline metadata**，依赖会被 `uv` 自动管理。直接：

```bash
cd math-concepts/market-making/stochastic_process

uv run 01_brownian_motion.py
uv run 02_poisson_process.py
uv run 03_hawkes_process.py
uv run 04_jump_diffusion.py
uv run 05_market_making_demo.py
```

每个脚本会在当前目录生成对应 PNG 图。
