```
解释这个反向传播

【参数】
    W1 (4×2), b1 (4)      第 1 层
    W2 (1×4), b2 (1)      第 2 层

【前向 (forward) — 数据从左走到右, 算损失】
    h_pre = W1 · x + b1        ← 第 1 层的线性变换
    h     = ReLU(h_pre)        ← 第 1 层的激活
    z     = W2 · h + b2        ← 第 2 层的线性变换
    y_hat = sigmoid(z)         ← 把任意 z 压到 (0, 1) 当概率
    L     = -```mathy·log(y_hat) + (1-y)·log(1-y_hat)```   ← BCE 交叉熵

【反向 (backward) — 损失反着传, 算每个参数的"梯度"】
  "梯度" = 损失 L 对该参数的偏导数, 告诉我们"这个参数往哪边动一点 L 会变小"。
  规则: 链式法则 (chain rule)
      dL/dW = dL/d(输出) · d(输出)/dW

  从后往前一层一层算:
    dL/dz   = y_hat - y                  # (sigmoid + BCE 化简的"魔法"结果, 干净到不像话)
    dL/dW2  = (dL/dz) · h.T              # 链式: L → z → W2
    dL/db2  = dL/dz                      # 链式: L → z → b2
    dL/dh   = W2.T · (dL/dz)             # 把误差"反传"到隐藏层
    dL/dh_pre = dL/dh * relu'(h_pre)     # 穿过 ReLU 激活 (逐元素相乘)
    dL/dW1  = (dL/dh_pre) · x.T          # 链式: 一路传到 W1
    dL/db1  = dL/dh_pre
```


---


这个例子其实已经把神经网络反向传播的核心都写出来了。理解它的关键是：

> **每一层只负责两件事：**
>
> 1. 接收后面传回来的误差（gradient）
> 2. 根据链式法则继续往前传

我们用一个具体数字跑一次。

---

# 1. 网络结构

输入：

```math
x =
\begin{bmatrix}
1\
2
\end{bmatrix}
```

第1层：

```math
W_1=
\begin{bmatrix}
0.5 & 0.2\\
0.1 & 0.4\\
0.3 & 0.7\\
0.6 & 0.8
\end{bmatrix}
```

```math
b_1=
\begin{bmatrix}
0\\
0\\
0\\
0
\end{bmatrix}
```

第2层：

```math
W_2=
\begin{bmatrix}
0.2 & 0.3 & 0.4 & 0.5
\end{bmatrix}
```

```math
b_2=0
```

真实标签：

```math
y=1
```

---

# 2. Forward

## 第一层

```math
h_{pre}=W_1x+b_1
```

计算：

```math
\begin{bmatrix}
0.5\times1+0.2\times2\\
0.1\times1+0.4\times2\\
0.3\times1+0.7\times2\\
0.6\times1+0.8\times2
\end{bmatrix}
=
\begin{bmatrix}
0.9\\
0.9\\
1.7\\
2.2
\end{bmatrix}
```

---

## ReLU

因为全部大于0：

```math
h=
\begin{bmatrix}
0.9\\
0.9\\
1.7\\
2.2
\end{bmatrix}
```

---

## 第二层

```math
z=W_2h+b_2
```

```math
=0.2(0.9)+0.3(0.9)+0.4(1.7)+0.5(2.2)
```

```math
=2.23
```

---

## Sigmoid

```math
\hat y=\sigma(z)
```

```math
=\frac1{1+e^{-2.23}}
```

```math
\approx0.903
```

---

## BCE Loss

因为

```math
y=1
```

所以

```math
L=-\log(0.903)
```

```math
\approx0.102
```

---

# 3. Backward

反向传播从 Loss 开始。

---

## Step 1

著名公式：

```math
\frac{\partial L}{\partial z}
=
\hat y-y
```

所以：

```math
dL/dz
=
0.903-1
=
-0.097
```

---

### 这是什么意思？

表示：

> z 增大一点，
>
> Loss 会下降。

因为梯度是负数。

---

# 4. 求 W2 的梯度

公式：

```math
\frac{\partial L}{\partial W_2}
=
(dL/dz)\cdot h^T
```

即：

```math
-0.097
\times
\begin{bmatrix}
0.9&0.9&1.7&2.2
\end{bmatrix}
```

得到：

```math
dL/dW_2
=
\begin{bmatrix}
-0.0873 &
-0.0873 &
-0.1649 &
-0.2134
\end{bmatrix}
```

---

### 为什么是这样？

因为

```math
z
=

w_1h_1+w_2h_2+w_3h_3+w_4h_4
```

例如：

```math
\frac{\partial z}{\partial w_3}
=h_3
=
1.7
```

链式法则：

```math
\frac{\partial L}{\partial w_3}
=
\frac{\partial L}{\partial z}
\frac{\partial z}{\partial w_3}
```

```math
=(-0.097)(1.7)
```

```math
=-0.1649
```

---

# 5. 求 b2 梯度

因为

```math
z=W_2h+b_2
```

所以

```math
\frac{\partial z}{\partial b_2}=1
```

因此：

```math
dL/db_2
=dL/dz
=
-0.097
```

---

# 6. 把误差传回隐藏层

公式：

```math
dL/dh
=
W_2^T(dL/dz)
```

即：

```math

\begin{bmatrix}
0.2\\
0.3\\
0.4\\
0.5
\end{bmatrix}
(-0.097)
```

得到：

```math
dL/dh=
\begin{bmatrix}
-0.0194\\
-0.0291\\
-0.0388\\
-0.0485
\end{bmatrix}
```

---

### 直觉

这里是在回答：

> 第3个隐藏神经元如果变大一点，
>
> Loss 会变化多少？

因此误差被分配回每个隐藏神经元。

---

# 7. 穿过 ReLU

ReLU：

```math
ReLU(x)=\max(0,x)
```

导数：

```math
ReLU'(x)
=

\begin{cases}
1 & x>0\\
0 & x\le0
\end{cases}
```

由于：

```math
h_{pre}=[0.9,0.9,1.7,2.2]
```

全部大于0。

所以：

```math
relu'(h_{pre})=[1,1,1,1]
```

因此：

```math
dL/dh_{pre}=dL/dh
```

完全不变。

---

# 8. 求 W1 梯度

公式：

```math
dL/dW_1=(dL/dh_{pre})x^T
```

这里是外积（outer product）。

```math
\begin{bmatrix}
-0.0194\\
-0.0291\\
-0.0388\\
-0.0485
\end{bmatrix}
\begin{bmatrix}
1&2
\end{bmatrix}
```

得到：

```math
dL/dW_1=
\begin{bmatrix}
-0.0194 & -0.0388\\
-0.0291 & -0.0582\\
-0.0388 & -0.0776\\
-0.0485 & -0.0970
\end{bmatrix}
```

---

# 9. 求 b1 梯度

因为

```math
h_{pre}=W_1x+b_1
```

对偏置求导：

```math
dL/db_1=dL/dh_{pre}
```

即：

```math
\begin{bmatrix}
-0.0194\\
-0.0291\\
-0.0388\\
-0.0485
\end{bmatrix}
```

---

# 10. 梯度下降更新

假设学习率：

```math
lr=0.1
```

更新规则：

```math
W:=W-lr\times gradient
```

例如：

```math
W_{2,1}=0.2-0.1(-0.0873)
```

# ```math

0.20873
```

变大了。

为什么？

因为梯度是负数。

增加这个权重会让 Loss 下降。

---

# 整个反向传播可以浓缩成一句话

前向传播：

```text
x
→ W1+b1
→ ReLU
→ W2+b2
→ sigmoid
→ Loss
```

反向传播：

```text
Loss
→ dL/dz
→ dL/dW2
→ dL/dh
→ dL/dh_pre
→ dL/dW1
```

每一步都只是不断应用链式法则：

```math
\frac{\partial L}{\partial A}
=
\frac{\partial L}{\partial B}
\cdot
\frac{\partial B}{\partial A}
```

你会发现 PyTorch 的 `.backward()` 本质上就是自动帮你完成这一串链式法则计算，把所有：

```math
dL/dW_2,\quad dL/db_2,\quad dL/dW_1,\quad dL/db_1
```

全部算出来并存到 `.grad` 里面。
