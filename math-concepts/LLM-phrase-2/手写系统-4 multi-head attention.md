# 如何手写 multi-head attention

如果你已经理解了：

* 向量
* 矩阵乘法
* 点积
* Linear Layer
* Single-Head Attention

那么 Multi-Head Attention 本质上只是：

> **把 Single-Head Attention 并行做 N 次，然后拼接起来。**

---

# Step 1: 回顾 Single-Head Attention

输入：

```python
x.shape = (batch, seq_len, d_model)
```

例如：

```python
(batch=2, seq_len=4, d_model=8)
```

---

三个线性层：

```python
Q = x @ Wq
K = x @ Wk
V = x @ Wv
```

得到：

```python
Q.shape = (2,4,8)
K.shape = (2,4,8)
V.shape = (2,4,8)
```

计算 attention：

```python
scores = Q @ K.transpose(-2,-1)
```

shape：

```python
(2,4,8)
@
(2,8,4)

=

(2,4,4)
```

即：

```text
token 对 token 的关注程度
```

---

softmax：

```python
weights = softmax(scores)
```

得到：

```python
(2,4,4)
```

---

加权 V：

```python
out = weights @ V
```

shape：

```python
(2,4,4)
@
(2,4,8)

=

(2,4,8)
```

这就是 single-head attention。

---

# Step 2: 为什么要多个 Head

Single Head 只有一种关注方式。

例如：

```text
Tom gave Mary a book because she wanted it.
```

可能：

Head1：

```text
she -> Mary
```

Head2：

```text
it -> book
```

Head3：

```text
Tom -> gave
```

Head4：

```text
主谓关系
```

不同 head 学不同模式。

---

# Step 3: 切分 embedding

假设：

```python
d_model = 8
num_heads = 2
```

那么：

```python
head_dim = 4
```

即：

```text
8维
=
2个head
×
4维
```

---

原来：

```python
Q.shape = (2,4,8)
```

reshape：

```python
Q =
Q.view(
    batch,
    seq_len,
    num_heads,
    head_dim
)
```

变成：

```python
(2,4,2,4)
```

---

再交换维度：

```python
Q = Q.transpose(1,2)
```

变成：

```python
(2,2,4,4)
```

含义：

```text
(batch,
 heads,
 seq_len,
 head_dim)
```

即：

```python
(batch=2,
 heads=2,
 seq=4,
 dim=4)
```

K、V 同理。

---

# Step 4: Attention 现在变成四维

Q：

```python
(2,2,4,4)
```

K：

```python
(2,2,4,4)
```

计算：

```python
scores =
Q @ K.transpose(-2,-1)
```

K：

```python
(2,2,4,4)
```

转置：

```python
(2,2,4,4)
```

最后两维交换：

```python
(2,2,4,4)
```

得到：

```python
scores.shape
=
(2,2,4,4)
```

含义：

```text
batch=2

head=0:
    4×4 attention

head=1:
    4×4 attention
```

实际上：

```text
每个 head
都有自己的一张 attention 图
```

---

# Step 5: Softmax

```python
weights =
softmax(scores, dim=-1)
```

shape：

```python
(2,2,4,4)
```

---

# Step 6: 乘 V

```python
out = weights @ V
```

shape：

```python
(2,2,4,4)
@
(2,2,4,4)

=

(2,2,4,4)
```

得到：

```python
(batch,
 heads,
 seq_len,
 head_dim)
```

---

# Step 7: 拼回去

现在：

```python
out.shape
=
(2,2,4,4)
```

交换回来：

```python
out =
out.transpose(1,2)
```

变成：

```python
(2,4,2,4)
```

---

再 reshape：

```python
out =
out.reshape(
    batch,
    seq_len,
    d_model
)
```

得到：

```python
(2,4,8)
```

即：

```text
head0 的结果
+
head1 的结果

拼接
```

---

# Step 8: 最后一个线性层

Transformer 还有：

```python
Wo
```

```python
out =
out @ Wo
```

shape：

```python
(2,4,8)
```

不变。

作用：

```text
融合所有 head 的信息
```

---

# 最简 PyTorch 手写版

```python
import torch
import torch.nn.functional as F

def mha(x, Wq, Wk, Wv, Wo, num_heads):

    B, S, D = x.shape

    head_dim = D // num_heads

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv

    Q = Q.view(B, S, num_heads, head_dim).transpose(1, 2)
    K = K.view(B, S, num_heads, head_dim).transpose(1, 2)
    V = V.view(B, S, num_heads, head_dim).transpose(1, 2)

    scores = (
        Q @ K.transpose(-2, -1)
    ) / (head_dim ** 0.5)

    weights = F.softmax(scores, dim=-1)

    out = weights @ V

    out = out.transpose(1, 2)

    out = out.reshape(B, S, D)

    out = out @ Wo

    return out
```

---

如果你的目标是**从零手写一个最小 GPT**，推荐学习顺序：

1. Embedding
2. Linear Layer
3. Single-Head Attention
4. Multi-Head Attention ← 你现在这里
5. Residual Connection
6. LayerNorm
7. Transformer Block
8. Causal Mask
9. Cross Entropy Loss
10. Backpropagation
11. AdamW
12. 训练一个 Tiny GPT

从数学角度看，Multi-Head Attention 最重要的一行就是：

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```


而 Multi-Head 只是把这套计算并行做很多次：

```text
head1 = Attention(Q1,K1,V1)

head2 = Attention(Q2,K2,V2)

...

headN = Attention(QN,KN,VN)

output = Concat(head1,...,headN) @ Wo
```

本质上没有新的数学，只是增加了一个 `heads` 维度。
