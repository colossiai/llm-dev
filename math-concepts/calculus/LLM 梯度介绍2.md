
如果你已经理解了：

* 向量（Vector）
* 矩阵（Matrix）
* 线性变换（Wx）
* Embedding 空间
* 导数（Derivative）

那么理解 LLM 里的梯度（Gradient）就只差一步了。

---

# 一句话定义

梯度就是：

> **告诉模型参数应该往哪个方向修改，才能让错误（Loss）下降得最快。**

---

# 从最简单的一元函数开始

假设：

```math
f(x)=x^2
```

在：

```math
x=3
```

时：

```math
f'(x)=2x=6
```

这里导数 6 的意思：

> x 增加一点点，
> f(x) 会增加约 6 倍那么多。

---

几何上：

```text
          /
         /
        /
       /
------*--------
      x=3
```

切线斜率很陡。

---

# 导数升级成梯度

现实中的神经网络参数不是一个数。

而是：

```text
θ = [w1,w2,w3,...]
```

例如：

```text
θ = [1,2]
```

Loss：

```text
L(w1,w2)
```

此时不能再用单个导数。

因为：

```text
∂L/∂w1
∂L/∂w2
```

都存在。

于是组合成：

```math
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}
\end{bmatrix}
```

\nabla L=\begin{bmatrix}\frac{\partial L}{\partial w_1}\\frac{\partial L}{\partial w_2}\end{bmatrix}

这就是梯度。

---

# 梯度的几何意义

想象 Loss 是一座山：

```text
            ^
           / \
          /   \
         /     \
        /       \
-------*---------
      当前参数
```

梯度指向：

```text
上山最快方向
```

而训练想做的是：

```text
下山
```

所以更新参数：

```math
\theta_{new}
=
\theta
-\eta \nabla L
```

其中：

* θ = 参数
* η = Learning Rate
* ∇L = 梯度

这就是 Gradient Descent。

\theta_{new}=\theta-\eta\nabla L

---

# 在 LLM 里面梯度是什么

假设输入：

```text
I love cats
```

真实答案：

```text
very much
```

模型预测：

```text
I love cats and
```

于是计算 Loss：

```text
预测 ≠ 真实
```

Loss 较大。

---

接下来会问：

```text
到底是哪些参数导致预测错了？
```

这就是梯度干的事。

---

# 举例

Transformer 某层参数：

```text
Wq
```

形状：

```text
4096 × 4096
```

有：

```text
16,777,216
```

个参数。

对于其中一个元素：

```text
Wq[123][456]
```

梯度可能是：

```text
+0.08
```

意思：

> 如果这个参数继续增大，
> Loss 会变大。

所以应该减小它。

---

另一个参数：

```text
-0.03
```

意思：

> 如果这个参数增大，
> Loss 反而会减小。

所以应该增大它。

---

# 一个 Transformer Layer 的梯度

参数包括：

```text
Wq
Wk
Wv
Wo

W1
W2
```

每个都是矩阵。

例如：

```text
Wq: 4096×4096
Wk: 4096×4096
Wv: 4096×4096
```

梯度也是同样大小矩阵：

```text
∂L/∂Wq
∂L/∂Wk
∂L/∂Wv
```

例如：

```text
∂L/∂Wq
=
[
 [0.01, -0.03, ...]
 [0.08,  0.02, ...]
 ...
]
```

梯度矩阵中的每个数字：

> 都表示 Loss 对该参数的敏感程度。

---

# 为什么梯度能算出来

核心是链式法则（Chain Rule）。

你最近在学 LLM，链式法则是理解反向传播的关键。

Transformer 前向传播：

```text
x
 ↓
Embedding
 ↓
Q,K,V
 ↓
Attention
 ↓
MLP
 ↓
Logits
 ↓
Loss
```

Loss 实际上是：

```math
L=f(g(h(...(W))))
```

很多函数嵌套。

链式法则告诉我们：

```math
\frac{dL}{dW}
=
\frac{dL}{dh}
\cdot
\frac{dh}{dg}
\cdot
\frac{dg}{dW}
```

\frac{dL}{dW}=\frac{dL}{dh}\cdot\frac{dh}{dg}\cdot\frac{dg}{dW}

因此可以从 Loss 一层层往回传。

这就是：

```text
Backpropagation
反向传播
```

---

# 从几何角度理解 LLM 梯度

你已经在学习 Embedding 空间。

其实：

```text
Embedding
↓
线性变换
↓
Attention
↓
Loss
```

整个模型可以看成：

> 在超高维空间中不断变换向量。

训练时：

* Forward：计算预测结果
* Backward：计算梯度

梯度回答的是：

> 为了让正确答案概率更高，
> 应该如何扭曲这些高维空间？

因此：

```text
参数 W
决定空间形状

梯度 ∇L
决定如何修改空间形状
```

这是理解梯度最接近 LLM 本质的视角。

---

# 学习 LLM 时对梯度最重要的理解

不要把梯度理解成一个公式。

把它理解成：

```text
梯度 =
Loss 对参数空间的方向指示器
```

或者：

```text
梯度 =
告诉模型如何修改
Q/K/V 投影空间
Embedding 空间
MLP 空间
使下一次预测更准确
```

当你接下来学习：

* Backpropagation
* Computational Graph（计算图）
* Jacobian
* Softmax Gradient
* Cross Entropy

你会看到：

```text
Loss
  ↓
梯度
  ↓
反向传播
  ↓
更新 Wq/Wk/Wv
  ↓
模型学到知识
```

这条链就是整个 LLM 训练过程的核心。
