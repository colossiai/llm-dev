# 如何手写 MLP

如果你已经理解了：

* 向量（Vector）
* 矩阵乘法（Matrix Multiplication）
* 导数（Derivative）
* 梯度（Gradient）
* Embedding
* Attention 基础

那么下一步最好的练习就是：

> **手写一个最小 MLP（Multi-Layer Perceptron，多层感知机）**

因为：

* 它是 Transformer 的基础组件
* 里面已经包含了神经网络最核心的东西

  * 线性层
  * 激活函数
  * Loss
  * 反向传播
  * 梯度下降

---

# 1. MLP 长什么样

最简单的：

```text
输入 x

   ↓

Linear
(W1,b1)

   ↓

ReLU

   ↓

Linear
(W2,b2)

   ↓

输出 y
```

数学表达：

```math
h = ReLU(W_1x+b_1)

y = W_2h+b_2
```

例如：

```text
x = [身高,体重]

↓

MLP

↓

预测体脂率
```

---

# 2. 一个具体例子

输入：

```math
x =
\begin{bmatrix}
2 \\
3
\end{bmatrix}
```

第一层：

```math
W_1 =
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
```

计算：

```math
z_1 = W_1x
```

结果：

```math
=
\begin{bmatrix}
8\\
18
\end{bmatrix}
```

---

# 3. 激活函数

如果只有矩阵乘法：

```math
y=W_2W_1x
```

其实还是一个大矩阵：

```math
y=Wx
```

模型没有变强。

所以需要：

```math
ReLU(x)=max(0,x)
```

例如：

```text
[-3, 2, -1, 5]

↓

[0, 2, 0, 5]
```

加入非线性。

---

# 4. Forward Pass

完整流程：

```math
z_1 = W_1x+b_1
```

```math
h = ReLU(z_1)
```

```math
y = W_2h+b_2
```

这叫：

> Forward Propagation（前向传播）

---

# 5. Loss

假设真实答案：

```math
t = 10
```

模型预测：

```math
y = 8
```

使用 MSE：

MSE=(y-t)^2

结果：

```math
(8-10)^2=4
```

Loss 越小越好。

---

# 6. 为什么需要梯度

目标：

```text
Loss ↓
```

例如：

```math
Loss(W)
```

我们需要知道：

```math
\frac{\partial Loss}{\partial W}
```

表示：

> W 往哪个方向调，Loss 会下降最快。

---

# 7. Backpropagation

链式法则开始登场：

```math
\frac{\partial Loss}{\partial W_1}
=
\frac{\partial Loss}{\partial y}
\cdot
\frac{\partial y}{\partial h}
\cdot
\frac{\partial h}{\partial z_1}
\cdot
\frac{\partial z_1}{\partial W_1}
```

这就是：

> Backpropagation（反向传播）

本质：

```text
Loss

↑

Linear2

↑

ReLU

↑

Linear1
```

从后往前求导。

---

# 8. 梯度下降更新参数

得到梯度：

```math
\frac{\partial Loss}{\partial W}
```

更新：

```math
W_{new}=W-\eta\frac{\partial Loss}{\partial W}
```


其中：

```math
η
```

是学习率（Learning Rate）。

例如：

```text
W = 5

gradient = 2

lr = 0.1
```

更新：

```text
W = 5 - 0.1×2

  = 4.8
```

---

# 9. 用 NumPy 手写

```python
import numpy as np

# 输入
x = np.array([[2.0],
              [3.0]])

# 参数
W1 = np.random.randn(4,2)
b1 = np.zeros((4,1))

W2 = np.random.randn(1,4)
b2 = np.zeros((1,1))

# forward
z1 = W1 @ x + b1

h = np.maximum(0, z1)

y = W2 @ h + b2

print(y)
```

这里已经完成：

```text
Linear
↓
ReLU
↓
Linear
```

的前向传播。

---

# 10. 推荐学习顺序

对于「从数学到手写 LLM」路线：

```text
✓ 向量

✓ 矩阵

✓ 点积

✓ 线性变换

✓ 导数

✓ 梯度

↓
手写 MLP
    ↓
手写 Backprop
    ↓
手写 SGD
    ↓
手写 Softmax
    ↓
手写 Cross Entropy
    ↓
手写 Embedding
    ↓
手写 Self-Attention
    ↓
手写 Multi-Head Attention
    ↓
手写 Transformer Block
    ↓
手写 Mini GPT
```

如果只能做一个练习来真正理解神经网络，我建议：

> **不用 PyTorch，纯 NumPy 手写一个“两层 MLP + MSE + Backprop + SGD”**。

当你能亲手推导并实现：

```math
Linear → ReLU → Linear → MSE
```

整个反向传播后，再去看 Transformer 里的 Attention 和 FFN（Feed Forward Network），会非常自然。事实上，Transformer Block 里的 FFN 本质上就是一个更大的 MLP。
