# #介绍LLM chain-rule


如果你已经学过导数和梯度，那么 **Chain Rule（链式法则）** 是理解 LLM 训练的核心数学工具。

因为：

> **LLM 的本质是一个超长的复合函数（nested function）**
>
> 而训练时需要知道：
>
> > Loss 对每一个参数的影响有多大？

这个问题就是靠 Chain Rule 解决的。

---

# 1. 什么是 Chain Rule

先看最简单例子：

```math
y = (x^2)^3
```

可以拆成两层：

```math
u = x^2
```

```math
y = u^3
```

那么：

```math
\frac{dy}{dx}
=
\frac{dy}{du}
\cdot
\frac{du}{dx}
```

即：

```math
3u^2 \cdot 2x
```

代入：

```math
6x^5
```

这就是链式法则：

> 外层导数 × 内层导数

---

# 2. 为什么需要 Chain Rule

假设：

```math
x → f → g → loss
```

例如：

```math
x
→ linear
→ relu
→ softmax
→ loss
```

Loss 并不直接依赖 x。

而是：

```math
loss = h(g(f(x)))
```

你想知道：

```math
\frac{d loss}{dx}
```

必须一层层传回来：

```math
\frac{d loss}{dx}
=
\frac{d loss}{dg}
\cdot
\frac{dg}{df}
\cdot
\frac{df}{dx}
```

这就是 Chain Rule。

---

# 3. 图像理解

假设：

```text
x
│
▼
Linear
│
▼
ReLU
│
▼
Softmax
│
▼
Loss
```

Forward：

```text
x → prediction → loss
```

Backward：

```text
loss
← gradient
← gradient
← gradient
← x
```

梯度会沿着图反向传播。

所以：

> Backpropagation = Chain Rule 的工程实现

---

# 4. LLM 里的真实情况

一个 Transformer Block：

```text
embedding
   │
   ▼
QKV projection
   │
   ▼
attention
   │
   ▼
linear
   │
   ▼
MLP
   │
   ▼
output
```

假设参数：

```math
W_Q
```

你要更新它。

目标：

```math
\frac{\partial Loss}{\partial W_Q}
```

但是：

```math
Loss
```

并不直接依赖：

```math
W_Q
```

而是：

```text
W_Q
↓
Q
↓
attention score
↓
softmax
↓
context
↓
output
↓
loss
```

因此：

```math
\frac{\partial Loss}{\partial W_Q}
=
\frac{\partial Loss}{\partial output}
\cdot
\frac{\partial output}{\partial context}
\cdot
\frac{\partial context}{\partial attention}
\cdot
\frac{\partial attention}{\partial Q}
\cdot
\frac{\partial Q}{\partial W_Q}
```

这就是 Chain Rule。

---

# 5. 一个最小神经网络例子

假设：

```math
z = wx+b
```

```math
a = ReLU(z)
```

```math
Loss = (a-y)^2
```

结构：

```text
w
│
▼
z
│
▼
a
│
▼
loss
```

求：

```math
\frac{dLoss}{dw}
```

链式法则：

```math
\frac{dLoss}{dw}
=
\frac{dLoss}{da}
\cdot
\frac{da}{dz}
\cdot
\frac{dz}{dw}
```

分别计算：

### 第一步

```math
Loss=(a-y)^2
```

```math
\frac{dLoss}{da}
=
2(a-y)
```

---

### 第二步

```math
a=ReLU(z)
```

```math
\frac{da}{dz}
=
\begin{cases}
1 & z>0\\
0 & z<0
\end{cases}
```

---

### 第三步

```math
z=wx+b
```

```math
\frac{dz}{dw}=x
```

---

最终：

```math
\frac{dLoss}{dw}
=
2(a-y)
\cdot
ReLU'(z)
\cdot
x
```

---

# 6. 在 PyTorch 中发生什么

你写：

```python
loss.backward()
```

PyTorch 会自动构建计算图：

```text
x
 ↓
matmul
 ↓
relu
 ↓
softmax
 ↓
loss
```

然后自动应用 Chain Rule：

```math
\frac{\partial Loss}{\partial parameter}
```

对图中每个参数都求出来。

所以：

```python
loss.backward()
```

本质上等于：

> 沿着整个 Transformer 图，自动执行数十亿次 Chain Rule。

---

# 7. 为什么 Chain Rule 是 LLM 的核心

Transformer 的训练其实只有三件事：

### Forward

```text
token
→ embedding
→ attention
→ MLP
→ logits
```

---

### Loss

```text
预测词
vs
真实词
```

计算误差。

---

### Backward

把误差传回去：

```text
loss
← logits
← MLP
← attention
← embedding
```

这一整个过程完全依赖：

```math
\frac{\partial Loss}{\partial \theta}
```

而计算这个梯度的方法就是：

```math
Chain\ Rule
```

---

从 LLM 学习路径来看：

```text
导数
  ↓
偏导数
  ↓
梯度
  ↓
Chain Rule
  ↓
Backpropagation
  ↓
PyTorch Autograd
  ↓
Transformer Training
```

其中 **Chain Rule 是连接“梯度”和“反向传播”的桥梁**。理解了它，你就能明白为什么误差能从最后一个 token 一直传回到最前面的 Embedding 和 Q/K/V 矩阵。
