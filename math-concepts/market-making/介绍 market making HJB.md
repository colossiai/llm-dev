# 介绍 market making HJB


如果你学到 **Avellaneda–Stoikov Market Making**，很快就会遇到一个核心概念：

> **HJB（Hamilton–Jacobi–Bellman Equation）**
>
> 它是“最优控制（Optimal Control）”问题的核心工具。

对于做市来说，HJB 的作用是：

> **在每个时刻决定最优 Bid 和 Ask 报价。**

---

# 1. 做市商面临什么问题？

假设你在做 BTC 做市。

当前市场：

```text
Mid Price = 100
```

你可以挂：

```text
Bid = 99.9
Ask = 100.1
```

赚：

```text
Spread = 0.2
```

---

问题来了：

如果突然上涨：

```text
100
101
102
103
```

你的卖单被成交：

```text
卖出 @100.1
```

随后价格：

```text
103
```

你亏惨了。

这叫：

**Inventory Risk**

（库存风险）

---

因此做市目标不是：

```text
赚最多 Spread
```

而是：

```text
Spread 收益
-
库存风险
```

---

# 2. 定义状态（State）

通常定义：

价格：

```math
S_t
```

库存：

```math
q_t
```

现金：

```math
X_t
```

---

当前状态：

```math
(X_t,S_t,q_t)
```

例如：

```text
Cash = 10000

Inventory = +50

Price = 100
```

---

# 3. Value Function

定义：

```math
V(t,S,q,X)
```

表示：

> 当前状态下，从现在做到结束能够获得的最大期望收益。

---

例如：

```text
今天下午 3 点

库存 = 0
```

价值：

```text
100
```

---

而：

```text
库存 = 5000
```

价值：

```text
-500
```

因为风险更大。

---

# 4. Bellman 思想

核心思想：

> 最优策略 = 当前最优决策 + 未来最优策略

即：

```math
V(t)
=
\max_u
\left(
\text{即时收益}
+
\text{未来价值}
\right)
```

其中：

```math
u
```

就是控制变量。

---

在做市里：

```math
u=
(\delta^b,\delta^a)
```

即：

```text
Bid离Mid多远

Ask离Mid多远
```

---

# 5. HJB 的来源

对极小时间：

```math
dt
```

考虑。

Bellman：

```math
V(t)
=
\max_u
E
\left[
V(t+dt)
+
dProfit
\right]
```

---

展开：

```math
V(t+dt)
=
V
+
V_tdt
+
V_sdS
+
\frac12V_{ss}(dS)^2
```

---

利用：

```math
dS
=
\sigma dW
```

dS = \sigma dW

以及：

```math
(dW)^2=dt
```

---

整理后得到：

```math
0
=

V_t
+
\frac12\sigma^2V_{ss}
+
\max_u
\left(
\text{Order Arrival Gain}
\right)
```

这就是 HJB。

---

# 6. 在做市里的 HJB

Avellaneda–Stoikov 推导后得到：

```math
\partial_tV
+
\frac12\sigma^2\partial_{SS}V
+
\max_{\delta_b,\delta_a}
{
\lambda_b(\delta_b)\Delta V_b
+
\lambda_a(\delta_a)\Delta V_a
}
=0
```

其中：

---

成交率：

```math
\lambda_b
```

买单成交强度

---

```math
\lambda_a
```

卖单成交强度

---

通常假设：

```math
\lambda(\delta)
=
Ae^{-k\delta}
```

```math
\lambda(\delta)=Ae^{-k\delta}
```

意思：

```text
挂得越远

成交概率越低
```

---

例如：

| 距离Mid | 成交率 |
| ----- | --- |
| 0.01  | 90% |
| 0.05  | 40% |
| 0.10  | 10% |

---

# 7. HJB 真正在做什么

本质是在寻找：

```math
\delta_b^*
```

```math
\delta_a^*
```

即：

最优买价

最优卖价

---

你可以理解成：

```text
Spread 开太大

赚得多
成交少

Spread 开太小

赚得少
成交多
```

---

HJB自动计算：

```text
哪个距离最赚钱
```

---

# 8. 最经典结果

Avellaneda–Stoikov 解出来：

Reservation Price：

```math
r
=
S
-
q\gamma\sigma^2(T-t)
```

---

这是整个模型最著名的公式。

---

解释：

库存：

```text
q > 0
```

持仓太多。

那么：

```math
r<S
```

即：

```text
主动降低报价

想赶紧卖掉
```

---

库存：

```text
q < 0
```

空仓太多。

则：

```math
r>S
```

即：

```text
提高报价

想买回来
```

---

这就是：

**Inventory Skew**

（库存倾斜）

现代做市系统几乎都会有类似逻辑。

---

# 9. 对工程师的直觉理解

你不必把 HJB 当成偏微分方程。

可以把它理解成：

```text
输入：

价格
波动率
库存
剩余时间

↓

优化器(HJB)

↓

输出：

最佳 Bid
最佳 Ask
```

即：

```text
Market State
      ↓
   HJB
      ↓
Optimal Quotes
```

---

对于从金融系统研发转做市策略的人，建议按下面顺序学习：

1. 市场微观结构（LOB、Queue）
2. Poisson Process
3. Dynamic Programming（Bellman）
4. HJB
5. Avellaneda–Stoikov
6. Inventory Control
7. Optimal Execution（Almgren–Chriss）

学完这些，你就能读懂大部分 HFT 做市论文中出现的 HJB 推导，并理解为什么库存、波动率和成交强度会直接影响最优报价。
