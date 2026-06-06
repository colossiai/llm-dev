# 理解 pytorch einsum


`torch.einsum()`（Einstein Summation）是 PyTorch 里一个非常强大的张量运算工具。

很多 Attention、Transformer、矩阵运算都可以用它一行表达。

---

# 1. 最简单理解

假设：

```python
A.shape = (2, 3)
B.shape = (3, 4)
```

普通矩阵乘法：

```python
C = A @ B
```

结果：

```python
C.shape = (2, 4)
```

对应的 einsum：

```python
C = torch.einsum('ik,kj->ij', A, B)
```

含义：

```text
i = A 的行
k = A 的列 = B 的行
j = B 的列
```

公式：

```math
C_{ij} = \sum_k A_{ik} B_{kj}
```

即：

```text
重复出现的维度 k
=> 自动求和
```

这是 einsum 最核心规则。

---

# 2. einsum 语法

格式：

```python
torch.einsum(
    "输入1索引,输入2索引,...->输出索引",
    tensor1,
    tensor2,
    ...
)
```

例如：

```python
torch.einsum('ab,bc->ac', A, B)
```

表示：

```text
A(a,b)
B(b,c)

b 被消掉（求和）

得到

C(a,c)
```

---

# 3. 维度字母只是标签

字母没有特殊意义：

```python
'ab,bc->ac'
```

和

```python
'xy,yz->xz'
```

完全一样。

---

# 4. 向量点积

向量：

```python
a = torch.tensor([1,2,3])
b = torch.tensor([4,5,6])
```

普通写法：

```python
torch.dot(a,b)
```

einsum：

```python
torch.einsum('i,i->', a, b)
```

公式：

```math
\sum_i a_i b_i
```

结果：

```text
1×4 + 2×5 + 3×6
= 32
```

---

# 5. 求和

```python
x.shape = (3,4)
```

全部求和：

```python
torch.einsum('ij->', x)
```

等价：

```python
x.sum()
```

---

# 6. 行求和

```python
torch.einsum('ij->i', x)
```

意思：

```text
保留 i
消掉 j
```

等价：

```python
x.sum(dim=1)
```

---

# 7. 转置

```python
x.shape = (2,3)
```

转置：

```python
torch.einsum('ij->ji', x)
```

等价：

```python
x.T
```

---

# 8. Batch Matrix Multiplication

假设：

```python
A.shape = (B, M, K)
B.shape = (B, K, N)
```

普通：

```python
torch.bmm(A, B)
```

einsum：

```python
torch.einsum(
    'bmk,bkn->bmn',
    A,
    B
)
```

这里：

```text
b = batch
m = 行
k = 内积维度
n = 列
```

---

# 9. Attention 里的 einsum

这是最常见场景。

假设：

```python
Q.shape = (batch, seq, d)
K.shape = (batch, seq, d)
```

---

计算：

```math
QK^T
```

普通：

```python
scores = Q @ K.transpose(-2, -1)
```

einsum：

```python
scores = torch.einsum(
    'bsd,btd->bst',
    Q,
    K
)
```

其中：

```text
b = batch
s = query token
t = key token
d = hidden dimension
```

公式：

```math
score_{bst}
=
\sum_d
Q_{bsd}
K_{btd}
```

结果：

```python
(batch, seq, seq)
```

这就是 Attention Score Matrix。

---

# 10. Multi-Head Attention

Transformer源码经常写：

```python
Q.shape = (B, H, S, D)
K.shape = (B, H, T, D)
```

计算 Attention Score：

```python
scores = torch.einsum(
    'bhsd,bhtd->bhst',
    Q,
    K
)
```

得到：

```python
(B, H, S, T)
```

即：

```text
每个 head
每个 query token
对所有 key token
计算相似度
```

---

再计算：

```math
Attention(Q,K,V)
=
softmax(QK^T)V
```

假设：

```python
attn.shape = (B,H,S,T)
V.shape    = (B,H,T,D)
```

那么：

```python
out = torch.einsum(
    'bhst,bhtd->bhsd',
    attn,
    V
)
```

得到：

```python
(B,H,S,D)
```

这正是 Transformer 核心计算。

---

# 11. 一个记忆口诀

看到：

```python
'bhsd,bhtd->bhst'
```

不要想着代码。

直接念出来：

```text
保留：

b h s t

消掉：

d
```

因为：

```text
d 出现在输入
但没出现在输出
```

所以：

```text
自动沿 d 求和
```

---

# 12. 一眼看懂 einsum

规则只有两条：

### ① 输出中存在的字母 → 保留

例如：

```python
'ij->i'
```

保留：

```text
i
```

---

### ② 输入中重复但输出没有的字母 → 求和

例如：

```python
'ik,kj->ij'
```

```text
k 被消掉
=> 沿 k 求和
```

即：

```math
C_{ij}
=
\sum_k
A_{ik}B_{kj}
```

---

如果你正在学习 Transformer，我推荐按下面顺序掌握：

```text
matmul
 ↓
batch matmul
 ↓
einsum
 ↓
single-head attention
 ↓
multi-head attention
 ↓
flash attention
```

因为 Attention 的所有核心公式，其实都可以用两条 einsum 表达：

```python
scores = einsum('bhsd,bhtd->bhst', Q, K)
out    = einsum('bhst,bhtd->bhsd', P, V)
```

理解了这两行，Transformer 的张量维度基本就通了。
