# 如何手写 single-head attention


如果你已经理解了：

* 向量（Vector）
* 矩阵乘法（Matrix Multiply）
* 点积（Dot Product）
* 投影（Projection）
* Softmax

那么手写一个 **Single-Head Self-Attention** 已经不远了。

---

# Step 1：输入

假设一句话：

```text
"I love AI"
```

已经被 embedding 成：

```python
X = [
    [1, 0, 1, 0],  # I
    [0, 2, 0, 1],  # love
    [1, 1, 1, 1],  # AI
]
```

shape:

```python
(seq_len, d_model)

(3, 4)
```

即：

```text
3 个 token
每个 token 4 维
```

---

# Step 2：生成 Q K V

Attention 的第一步：

```math
Q = XW_Q
K = XW_K
V = XW_V
```

Q=XW_Q

例如：

```python
WQ = np.random.randn(4, 4)
WK = np.random.randn(4, 4)
WV = np.random.randn(4, 4)

Q = X @ WQ
K = X @ WK
V = X @ WV
```

shape：

```python
Q: (3,4)
K: (3,4)
V: (3,4)
```

这里本质上只是：

```text
Embedding空间
    ↓
三个不同线性变换
    ↓
Q空间
K空间
V空间
```

---

# Step 3：计算 Attention Score

核心公式：

QK^T

```python
scores = Q @ K.T
```

shape：

```python
(3,4) @ (4,3)

↓

(3,3)
```

得到：

```text
        I   love  AI

I       5    2    7
love    1    9    4
AI      3    5    8
```

---

# 这里发生了什么？

例如：

```text
scores[0][2]
```

是：

```math
Q_I \cdot K_{AI}
```

即：

```text
"I" 看 "AI" 的相关度
```

Attention 本质：

```text
Query
去匹配
所有 Key
```

所以：

```python
scores[i][j]
```

表示：

```text
token i
关注
token j

的程度
```

---

# Step 4：缩放

Transformer 原论文：

```math
\frac{QK^T}{\sqrt{d_k}}
```

```python
scores = scores / np.sqrt(d_k)
```

例如：

```python
d_k = 4

sqrt(4)=2
```

变成：

```text
2.5 1.0 3.5
0.5 4.5 2.0
1.5 2.5 4.0
```

---

# 为什么除 √dk

因为：

```text
维度越高
点积越大
```

例如：

```python
64维
```

点积可能：

```text
120
200
300
```

Softmax 会爆炸：

```python
exp(300)
```

导致：

```text
几乎变成 one-hot
```

梯度消失。

所以：

```text
先缩放
再 Softmax
```

---

# Step 5：Softmax

```python
weights = softmax(scores)
```

对每一行做：

```text
2.5 1.0 3.5
```

变：

```text
0.25 0.06 0.69
```

得到：

```text
        I   love  AI

I      0.25 0.06 0.69
love   0.02 0.88 0.10
AI     0.06 0.16 0.78
```

每行：

```text
和 = 1
```

表示：

```text
注意力分布
```

---

# Step 6：加权求和 Value

最终：

```math
Attention(Q,K,V)=Softmax(QK^T/\sqrt{d_k})V
```

Attention(Q,K,V)=$Softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

代码：

```python
output = weights @ V
```

shape：

```python
(3,3) @ (3,4)

↓

(3,4)
```

---

# 几何意义

以 token "I" 为例：

权重：

```text
[0.25, 0.06, 0.69]
```

表示：

```text
输出(I)

=
0.25 * V(I)
+
0.06 * V(love)
+
0.69 * V(AI)
```

即：

```text
在 Value 空间里

做加权平均
```

所以：

```text
Attention
=
根据相关度

重新混合所有 token 信息
```

---

# 最简 NumPy 实现

```python
import numpy as np

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=-1, keepdims=True)

def attention(X, WQ, WK, WV):
    Q = X @ WQ
    K = X @ WK
    V = X @ WV

    scores = Q @ K.T

    dk = K.shape[-1]

    scores = scores / np.sqrt(dk)

    weights = softmax(scores)

    output = weights @ V

    return output
```

---

# 一眼看懂 Single-Head Attention

```text
X
│
├── XWQ → Q
├── XWK → K
└── XWV → V

QKᵀ
│
├─ token 与 token 相似度
│
▼

Softmax
│
├─ 注意力权重
│
▼

权重 × V
│
▼

Output
```

如果下一步继续学习 Transformer，最自然的顺序是：

```text
Single-Head Attention
    ↓
Masked Attention
    ↓
Multi-Head Attention
    ↓
Residual Connection
    ↓
LayerNorm
    ↓
Feed Forward Network
    ↓
Transformer Block
```

这样你就能逐步手写一个最小 GPT。
