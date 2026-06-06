# 如何手写 Linear Regression

如果你的目标是从数学基础走到“手写一个最简 LLM”，那么 **Linear Regression（线性回归）** 是第一个值得亲手实现的机器学习模型，因为它会把你已经学过的：

* 向量
* 矩阵乘法
* 导数
* 梯度

全部串起来。

---

# 1. 问题定义

假设有数据：

| 面积 x | 房价 y |
| ---- | ---- |
| 1    | 3    |
| 2    | 5    |
| 3    | 7    |
| 4    | 9    |

观察发现：

```math
y = 2x + 1
```

但我们并不知道这个公式。

Linear Regression 的目标是学习：

```math
y = wx + b
```

其中：

* w = weight
* b = bias

需要通过训练求出来。

---

# 2. 模型

模型非常简单：

```math
y=wx+b
```

例如：

```text
w = 0.5
b = 1
```

输入：

```text
x = 4
```

预测：

```text
ŷ = 0.5×4 + 1
   = 3
```

真实值：

```text
y = 9
```

显然预测错了。

---

# 3. Loss（损失函数）

最常用：

MSE（均方误差）

```math
\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
```

例如：

```text
预测 = 3
真实 = 9
```

误差：

```text
3 - 9 = -6
```

平方：

```text
36
```

Loss = 36

目标：

```text
让 Loss 越来越小
```

---

# 4. 求梯度

现在来到机器学习最核心的一步：

```text
Loss 对 w 的导数
Loss 对 b 的导数
```

即：

```math
∂Loss/∂w
\\
∂Loss/∂b
```

它们告诉我们：

> w 和 b 应该往哪个方向调整。

---

# 5. 梯度下降

更新公式：

```math
w = w - lr * dw
\\
b = b - lr * db
```

其中：

```text
lr = learning rate
```

例如：

```text
w = 0.5

dw = -10

lr = 0.01
```

更新：

```text
w = 0.5 - 0.01*(-10)
  = 0.6
```

因为梯度为负：

```text
说明应该增大 w
```

---

# 6. 手写 Python（不依赖 NumPy）

```python
data = [
    (1, 3),
    (2, 5),
    (3, 7),
    (4, 9)
]

w = 0.0
b = 0.0

lr = 0.01

for epoch in range(1000):

    dw = 0.0
    db = 0.0
    loss = 0.0

    n = len(data)

    for x, y in data:

        y_hat = w * x + b

        error = y_hat - y

        loss += error ** 2

        dw += 2 * error * x
        db += 2 * error

    loss /= n
    dw /= n
    db /= n

    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(epoch, loss)

print("w =", w)
print("b =", b)
```

输出接近：

```text
w = 2.0
b = 1.0
```

成功学出：

```math
y = 2x + 1
```

---

# 7. 用矩阵形式理解（LLM必经之路）

刚才写的是：

```math
y = wx + b
```

实际上机器学习喜欢写成：

```math
y = XW + b
```

例如：

```text
X =
[
 [1]
 [2]
 [3]
 [4]
]
```

```text
W =
[
 [2]
]
```

矩阵乘法：

```text
XW =
[
 [2]
 [4]
 [6]
 [8]
]
```

再加上：

```text
b = 1
```

得到：

```text
[
 [3]
 [5]
 [7]
 [9]
]
```

---

# 8. 与 LLM 的关系

Linear Regression 实际上就是最简单的神经网络：

```text
输入
 ↓
Linear Layer
 ↓
输出
```

即：

```math
y = Wx + b
```

而 Transformer 中大量存在：

```math
Q = XW_Q

K = XW_K

V = XW_V
```

本质上也是：

```text
输入向量
    ↓
矩阵乘法
    ↓
新的语义空间
```

所以如果你能完全手写 Linear Regression，并理解：

1. Forward

```text
y_hat = wx + b
```

2. Loss

```text
MSE
```

3. Gradient

```text
dLoss/dw
```

4. Update

```text
w = w - lr*dw
```

那么你已经掌握了：

```text
Linear Layer
↓
Loss
↓
Backprop
↓
Gradient Descent
```

这正是从数学走向 Transformer 和 LLM 训练的第一块完整拼图。
