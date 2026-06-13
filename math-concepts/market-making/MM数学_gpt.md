如果你的目标是从**金融系统研发工程师**进一步走向**高频交易（HFT）Market Making Strategy Engineer**，那么需要的数学和做 LLM 不太一样。

对于做市（Market Making）来说：

> **概率论 + 随机过程 + 最优化 + 时间序列**
>
> 比深度学习数学重要得多。

---

# 第一层：必须掌握（核心）

## 1. 概率论（Probability）

这是所有量化交易的基础。

需要熟悉：

* 条件概率
* 贝叶斯公式
* 随机变量
* 期望（Expectation）
* 方差（Variance）
* 协方差（Covariance）
* 正态分布

例如：

市场价格未来变化：

[
\Delta P
]

可能：

| 价格变化    | 概率  |
| ------- | --- |
| +1 Tick | 40% |
| 0       | 20% |
| -1 Tick | 40% |

那么：

[
E[\Delta P]
]

就是未来价格期望。

做市策略本质：

> 预测未来几秒价格的概率分布。

---

## 2. 统计学（Statistics）

需要会：

* MLE（Maximum Likelihood Estimation）
* 假设检验
* p-value
* 置信区间

例如：

你发现一个信号：

```
Order Imbalance > 0.7
```

价格未来上涨：

```
52%
```

问题：

这是真 Alpha？

还是随机噪音？

统计学回答这个问题。

---

## 3. 线性代数

不需要像 LLM 那么深。

主要：

* 向量
* 矩阵
* 特征值
* PCA

例如：

100个特征：

* spread
* imbalance
* queue position
* trade flow

可能高度相关。

PCA 可以降维。

---

# 第二层：Market Making 最重要

## 4. 随机过程（Stochastic Process）

这是做市核心中的核心。

价格：

不是普通函数。

而是随机过程。

例如：

### Random Walk

[
P_t = P_{t-1}+\epsilon_t
]

---

### Brownian Motion

[
dS_t = \sigma dW_t
]

dS_t = \sigma dW_t

这是现代量化金融基础模型之一。

---

### Poisson Process

订单到达：

[
N(t)
]

通常建模成泊松过程。

做市策略大量使用。

例如：

* 买单到达率
* 卖单到达率
* 成交率

---

## 5. 马尔可夫链（Markov Chain）

假设：

当前订单簿状态：

```
State A
```

未来：

```
P(A -> B)
```

只与当前状态有关。

不与历史有关。

这就是：

Markov Chain

很多订单簿模型使用。

---

## 6. Queueing Theory（排队论）

HFT 特别重要。

因为：

订单簿本质是排队系统。

例如：

Bid:

```
100.00
-----------
你
前面500手
```

你什么时候成交？

取决于：

* 前面撤单速度
* 对手成交速度

需要：

* M/M/1
* Birth-Death Process
* Queue Dynamics

这是很多普通量化都不会的。

但 HFT 必须懂。

---

# 第三层：策略数学

## 7. 时间序列（Time Series）

重点：

### AR

[
X_t=\phi X_{t-1}
]

---

### ARIMA

---

### GARCH

波动率预测。

例如：

未来10秒：

```
波动率增加
```

那么：

应该扩大 spread。

---

## 8. 信号处理

很多 HFT 都在用。

例如：

订单流：

```
买买买买买
```

是不是趋势？

需要：

* Moving Average
* Exponential Moving Average
* Kalman Filter
* Fourier Transform（了解）

特别：

Kalman Filter

在 HFT 中非常常见。

---

# 第四层：高级做市模型

## 9. 最优化（Optimization）

做市本质：

平衡：

* 盈利
* 库存风险

---

例如：

库存：

[
q
]

过大：

风险增加。

目标：

[
\max E[PnL]-\lambda Risk
]

这就是优化问题。

---

需要：

* Convex Optimization
* Lagrange Multiplier
* Dynamic Programming

---

## 10. 随机控制（Stochastic Control）

高频做市最高级数学。

经典模型：

Avellaneda–Stoikov Model

核心思想：

同时决定：

* Bid Price
* Ask Price

使：

[
\text{Expected Profit} - \text{Inventory Risk}
]

最大。

几乎所有现代做市教材都会讲。

---

# 对于 HFT 做市，知识重要性排序

如果按实际工作价值排序：

| 排名 | 数学    |
| -- | ----- |
| 1  | 概率论   |
| 2  | 统计学   |
| 3  | 时间序列  |
| 4  | 随机过程  |
| 5  | 排队论   |
| 6  | 最优化   |
| 7  | 马尔可夫链 |
| 8  | 随机控制  |
| 9  | 线性代数  |
| 10 | 微积分   |

---

# 从金融系统研发转向 Strategy 的最小学习路径

你已经是金融系统研发工程师的话，我建议：

### Phase 1

* 概率论
* 统计学
* 时间序列

（1~2个月）

---

### Phase 2

* Order Book Microstructure
* Queue Theory
* Market Impact

（1个月）

---

### Phase 3

* 随机过程
* Brownian Motion
* Poisson Process

（1个月）

---

### Phase 4

* Avellaneda–Stoikov
* Inventory Management
* Optimal Market Making

（1个月）

完成以上内容后，你对做市策略的理解会超过很多只会调参数的量化开发工程师，并且能够读懂大部分 HFT 公司（如 Jane Street、Citadel Securities、Optiver）常见的市场微观结构和做市模型论文。
