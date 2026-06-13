# 做市策略 (Market Making) 所需数学知识学习路线

## Context

用户背景：金融系统研发工程师，已学完 LLM 原理，目标方向 AI + 金融工程化。
现在想进入高频做市策略 (HFT Market Making) 领域，需要补齐数学基础。

做市的本质：**在订单簿两侧持续报价赚价差，同时管理库存风险和被"逆向选择"的风险**。
这意味着核心数学问题是：「在随机订单流冲击下，如何动态选择最优买卖报价」。
所以学习重点不是"通用金融数学"，而是 **随机控制 + 微观结构 + 高频统计** 三件套。

---

## 一句话总结（先看这个）

> **做市 = 随机控制问题**：状态是 (库存, 价格, 订单簿)，控制是 (买价, 卖价)，目标是终端财富的效用最大化。整条学习链都围绕这个公式展开。

---

## 知识地图（按优先级 P0 > P1 > P2）

| 优先级 | 模块 | 解决什么问题 | 类比 |
|---|---|---|---|
| **P0** | 随机过程 | 描述价格 / 订单到达的随机性 | 像 LLM 里的 token 序列建模 |
| **P0** | 随机最优控制 | 在不确定性下做动态决策 | 像 RL 的 Bellman 方程 |
| **P0** | 市场微观结构 | 理解订单簿、价差、逆向选择 | 系统的"硬件层" |
| **P1** | 高频统计 | 处理 tick 数据特殊性 | 像处理不等长 / 噪声序列 |
| **P1** | 凸优化 / 数值方法 | HJB 方程求解、参数标定 | 模型训练里的 optimizer |
| **P2** | 强化学习 | 用数据驱动替代解析解 | 接你已有的 LLM/RL 背景 |
| **P2** | 博弈论 | 多家做市商竞争分析 | 多 agent 系统 |

---

## P0 — 必须掌握的核心三件套

### 1. 随机过程 (Stochastic Processes)

**直觉**：做市要先回答"下一笔订单什么时候来、什么方向、什么价格"，这就是随机过程。

| 知识点 | 在做市里干什么用 |
|---|---|
| 布朗运动 / 几何布朗运动 | 中间价 mid-price 的建模基线 |
| 泊松过程 / 复合泊松 | 订单到达时间建模 |
| **Hawkes 过程**（重要） | 订单流自激发、聚集效应（成交后短时间内更多成交） |
| 跳跃扩散 (Jump-Diffusion) | 价格突变事件建模 |
| 鞅、停时、Itô 引理 | 推导价格动态的工具 |

**推荐入门**：Shreve《Stochastic Calculus for Finance II》前 5 章。

### 2. 随机最优控制 (Stochastic Optimal Control)

**直觉**：你已经懂 RL 的 Bellman 方程了 → 把它从离散时间扩展到连续时间，就是 HJB 方程。做市的"解析做市理论"全在这里。

| 知识点 | 关键点 |
|---|---|
| 动态规划原理 | Bellman → HJB 的连续时间版本 |
| **HJB 方程** | 求解最优报价的偏微分方程 |
| 值函数 / 控制变量 | 值函数 = 期望效用；控制 = (δ_bid, δ_ask) |
| 库存惩罚项 | -γ·q² 这种风险厌恶项 |
| **Avellaneda-Stoikov (2008)** | 做市领域的"Attention is All You Need"，必读论文 |

**类比**：HJB 之于做市，就像 GPT 之于 NLP — 不读这篇论文等于没入门。

### 3. 市场微观结构 (Market Microstructure)

**直觉**：随机控制告诉你"应该报什么价"，微观结构告诉你"为什么会被吃单、为什么会亏"。

| 知识点 | 解决什么 |
|---|---|
| 限价订单簿 (LOB) 动力学 | 多档报价的随机演化 |
| 买卖价差分解 | 价差 = 订单处理成本 + 库存成本 + 逆向选择成本 |
| **Glosten-Milgrom 模型** | 解释为什么知情交易者吃掉做市商 |
| **Kyle 模型** | 知情交易 + 流动性交易的均衡 |
| 订单流不平衡 (OFI) | 短时价格预测的关键 alpha |
| 价格冲击模型 (Almgren-Chriss) | 大单冲击建模 |

**推荐**：Cartea, Jaimungal, Penalva《Algorithmic and High-Frequency Trading》(本领域圣经)。

---

## P1 — 工程化必备

### 4. 高频统计与时间序列

| 知识点 | 用途 |
|---|---|
| Realized Volatility | 从 tick 估波动率（替代 GARCH） |
| 微观结构噪声 | tick 价格 ≠ 真实价格 |
| 不规则采样 | 事件驱动而非等距时间 |
| HMM / Regime Switching | 识别市场状态（趋势/震荡/高波动） |

### 5. 凸优化 / 数值方法

- 凸优化：Boyd《Convex Optimization》前 5 章足够
- HJB 数值解：有限差分、policy iteration
- 蒙特卡洛模拟：策略回测必备
- 参数标定：MLE / GMM / Kalman 滤波

---

## P2 — 进阶 & 你的差异化优势

### 6. 强化学习做市（你的 LLM 背景能直接迁移）

**直觉**：解析的 Avellaneda-Stoikov 假设太强（线性中间价、指数效用），真实市场用 RL 拟合。

- DQN / PPO / SAC 在做市中的应用
- State 设计：库存 + LOB 特征 + 时间
- Reward shaping：PnL - λ·库存方差
- Sim-to-Real：训练环境的 LOB 模拟器
- **关键论文**：Spooner et al. "Market Making via Reinforcement Learning" (2018)

### 7. 博弈论（多做市商竞争）

- 不完全信息博弈
- Mean Field Games（多 agent 做市的连续极限）
- Cartea 等近年的 MFG 做市论文

---

## 推荐学习顺序（约 3-4 个月）

| 阶段 | 时长 | 内容 | 验收标准 |
|---|---|---|---|
| 1 | 3-4 周 | Shreve II 前 5 章 + Itô 引理 | 能推导 GBM 下的 BSM 公式 |
| 2 | 2 周 | Avellaneda-Stoikov 原文逐行推 | 能从 HJB 推到最优报价闭式解 |
| 3 | 4 周 | CJP 那本书第 1-7 章 | 理解 LOB、Hawkes、价格冲击 |
| 4 | 3 周 | 实现 AS 模型 + LOB 模拟器 | Python 跑出一条 PnL 曲线 |
| 5 | 持续 | RL 做市 + 真实 tick 数据 | 在你公司数据上跑出 alpha |

---

## 关键资源清单

**论文**（按重要性）
1. Avellaneda & Stoikov (2008) - High-frequency trading in a limit order book
2. Guéant, Lehalle, Fernandez-Tapia (2013) - Dealing with the inventory risk
3. Cartea & Jaimungal (2015) - Risk metrics for HFT
4. Spooner et al. (2018) - Market Making via RL

**书**
1. **Cartea, Jaimungal, Penalva** - Algorithmic and High-Frequency Trading（必读）
2. Shreve - Stochastic Calculus for Finance II（数学基础）
3. Lehalle & Laruelle - Market Microstructure in Practice（业界视角）
4. Aldridge - High-Frequency Trading（入门概览）

**课程**
- Imperial College / Oxford 的 MathFin 课程公开 slides
- QuantStart / Hudson & Thames 的 ML 量化系列

---

## 与你已有背景的衔接

| 你已会的 | 在做市里对应 | 学习捷径 |
|---|---|---|
| Transformer / LSTM | LOB 序列预测、短期 alpha | 直接迁移 |
| RL 基础（如果有） | 数据驱动做市 | 接 RL 做市论文 |
| 系统工程 | 行情接入、低延迟撮合 | 你的核心优势 |
| Python / C++ | 策略实现 | 不用补 |
| **缺的 → ** | 随机控制 + 微观结构 | **本计划的重点** |

---

## 验证你学懂了

能独立完成下面任一项，就算入门：
1. 从零推导 Avellaneda-Stoikov 最优报价公式
2. 用 Python 实现一个 LOB 模拟器，跑通 AS 策略并画出库存/PnL 曲线
3. 解释「为什么价差越窄反而越容易亏钱」（逆向选择）
4. 在 BitMEX/Binance tick 数据上拟合 Hawkes 过程

---

## 不在本计划范围

- 通用金融数学（期权定价、利率模型）— 与做市相关性低
- 宏观/基本面分析 — 与 HFT 完全无关
- 传统因子投资 — 时间尺度不同

如果某天你想做"中低频统计套利"再补这些，做市阶段先聚焦。
