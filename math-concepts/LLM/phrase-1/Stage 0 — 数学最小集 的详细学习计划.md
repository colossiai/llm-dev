如果目标是：

> **学会调试 LLM 模型（debugging / finetuning / 看 loss / 看 attention / 排查训练问题）**

那么 Stage 0 的原则是：

> **只学“够用数学”**
> 不搞线代证明，不刷大学题库，不做形式化推导。

目标是建立：

> **几何直觉 + 优化直觉 + 概率直觉**

最终达到：

看到：

```text
QK^T / √d
cross entropy
gradient update
```

不会恐惧。

我建议 **3 周（每天 45~90 分钟）**。

---

# Stage 0 总目标

学完后你应该能理解：

1. embedding 为什么是高维向量
2. attention 为什么用点积
3. 为什么：

```math
y=Wx+b
```

是在做空间变换

4. loss 为什么下降
5. gradient 为什么会 explode/vanish
6. cross entropy 在干嘛

---

# Stage 0 结构

分三块：

```text
Week1 线性代数（最重要）
Week2 微积分（优化直觉）
Week3 概率统计（loss直觉）
```

时间占比：

```text
线代 50%
微积分 30%
概率 20%
```

---

# Week 1 — 线性代数（重点）

目标：

> 建立“向量空间脑子”

每天约 60–90 分钟。

---

## Day 1：向量是什么

学：

* vector
* norm
* direction
* similarity

重点问题：

> 为什么 embedding 是向量？

必须理解：

> token 是高维空间里的点。

看：

理解：

* magnitude
* direction
* distance

练习：

想象：

```text
cat dog tiger lion
```

在空间里聚类。

你脑内要有：

> semantic space

直觉。

---

## Day 2：点积（极重要）

LLM 核心。

学：

理解：

> dot product ≈ 相似度

问题：

为什么：

```text
king queen
```

更接近？

理解：

* cosine similarity
* angle

练习：

解释：

> 为什么 attention 用 dot product

一句话：

> “谁和 query 更方向一致”

---

## Day 3：矩阵乘法 = 空间变换

你现在就在学。

学：

* matrix multiplication
* scaling
* rotation
* shear

理解：

> matrix = reshape space

必须理解：

```math
y=Wx
```

意思：

> 一个向量进入某空间后被重新表达。

练习：

解释：

> 为什么一个 Linear layer 能抽 feature？

---

## Day 4：Basis（核心）

学：

> basis intuition

理解：

> 矩阵列 = 新坐标轴

为什么重要？

因为：

LLM 的 hidden state：

> 在不断换坐标系。

直觉：

```text
happy axis
technical axis
politics axis
```

不同方向代表 feature。

---

## Day 5：Projection（极重要）

很多教程忽略。

学：

理解：

> attention ≈ 投影

问题：

> token 为什么更关注某词？

因为：

> “它更投影到当前方向”

---

## Day 6：矩阵维度 / tensor shape

学：

```text
(batch, seq_len, hidden_dim)
```

例如：

```text
(32, 2048, 4096)
```

必须能 mentally parse。

例如：

```text
Q.shape = (B, T, D)
K.shape = (B, T, D)
```

知道：

```text
QK^T
```

shape 为什么变。

这是 debug 基础。

---

## Day 7：复盘

自己解释：

```text
embedding
attention
linear layer
```

背后的线代。

标准：

能解释：

> “Attention 本质是在向量空间里做相似度与信息聚合。”

---

# Week 2 — 微积分（优化直觉）

目标：

> 理解 loss 如何驱动学习。

---

## Day 8：导数是什么

理解：

> slope

不是算题。

理解：

> “往哪里走 loss 降最快”

---

## Day 9：梯度 gradient

理解：

abla L

问题：

> 为什么叫 gradient descent？

因为：

> 顺坡下山。

脑内模型：

```text
mountain surface
```

找最低点。

---

## Day 10：链式法则（够用版）

理解：

> error backward

不要推导。

只理解：

```text
prediction wrong
↓
loss
↓
gradient
↓
update
```

---

## Day 11：学习率

理解：

```text
lr too large → explode
lr too small → no learning
```

这是 debug 高频。

---

## Day 12：vanishing/exploding gradient

理解：

> 为什么 transformer 要 residual

为什么：

```text
x + f(x)
```

更稳定。

---

## Day 13：Optimization intuition

知道：

* SGD
* AdamW

区别。

不用推导。

理解：

> 为什么 Adam 更稳。

---

## Day 14：复盘

解释：

> loss 是怎么让模型变聪明的？

---

# Week 3 — 概率（LLM 真正在干嘛）

目标：

> 理解 token prediction。

---

## Day 15：概率

理解：

```text
P(token)
```

例如：

输入：

```text
I drink hot ___
```

模型：

```text
tea = 0.6
coffee = 0.3
car = 0.0001
```

---

## Day 16：softmax

理解：

作用：

> 变成概率。

理解：

> logits → probability

---

## Day 17：log probability

理解：

为什么 log。

只要直觉：

> easier optimization

---

## Day 18：entropy

理解：

> uncertainty

例如：

```text
0.5 / 0.5
```

高 entropy。

---

## Day 19：cross entropy（极重要）

理解：

直觉：

> 惩罚模型猜错。

如果：

正确答案：

```text
tea
```

模型：

```text
tea=0.9
```

loss 小。

如果：

```text
tea=0.01
```

loss 爆炸。

---

## Day 20：temperature

理解：

为什么：

```text
temperature=0.1
```

更 deterministic。

---

## Day 21：复盘

解释：

> LLM 为什么是 next-token predictor？

---

# 每天学习模板（推荐）

45–90 分钟：

```text
20 min 看概念
20 min 看图像化解释
20 min 自己讲出来
20 min 小练习
```

最重要：

> **自己解释**

例如：

今天学点积。

你必须能说：

> “attention 用点积，因为它衡量 query 和 key 的方向相似度。”

---

# 推荐资源（最适合 LLM）

数学直觉：

[3Blue1Brown Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra?utm_source=chatgpt.com)

微积分：

[3Blue1Brown Essence of Calculus](https://www.3blue1brown.com/topics/calculus?utm_source=chatgpt.com)

Transformer：

[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/?utm_source=chatgpt.com)

训练直觉：

[Karpathy Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html?utm_source=chatgpt.com)

---

最终标准（Stage 0 毕业）：

看到：

```text
QK^T / √d
```

你能说：

> “这是在比较 token 间相似度。”

看到：

```text
loss.backward()
```

你知道：

> “梯度开始传播了。”

看到：

```text
nan loss
```

你脑子开始想：

```text
learning rate?
softmax overflow?
grad explosion?
dtype?
```

而不是 panic。
