# LLM学习中, tensor.shape, broadcasting, transpose / reshape / view, batch matrix multiplication 算是 PyTorch 使用基础技巧吗


是的，而且对于“学懂 LLM + 能调试代码”来说，这些基本上属于 **PyTorch 张量操作最小集合（minimum toolkit）**。

---

如果你现在的目标是：

> **从线代/attention 数学 → 手写一个最简 LLM → 看懂 PyTorch 模型**

那么这些知识的重要性是这样的：

## 第一层：必须掌握（直接决定你能不能看懂 LLM）

### 1. `tensor.shape`

这是最重要的。

因为 LLM 代码几乎全是在：

> **“张量维度变来变去”**

例如：

```python
x.shape
# (batch, seq_len, hidden_dim)
```

你必须做到：

> 一眼看到 shape 就知道空间意义。

例如：

```python
(32, 128, 768)
```

不是数字。

而是：

* `32` → batch size（32个样本）
* `128` → token 数
* `768` → embedding / hidden vector 长度

即：

> 32 句话，每句 128 token，每个 token 一个 768 维向量。

这是读 attention code 的前提。

---

### 2. transpose / permute

LLM 里极其常见。

因为 attention 要做：

```math
QK^T
```

本质：

> 为了让矩阵乘法维度对齐。

例如：

```python
Q.shape
(B, heads, T, d_k)

K.shape
(B, heads, T, d_k)
```

为了点积：

```python
Q @ K.transpose(-2, -1)
```

变成：

```python
(B, heads, T, d_k)
@
(B, heads, d_k, T)
=
(B, heads, T, T)
```

这里得到：

> 每个 token 和其他 token 的相关性矩阵。

因为这是核心公式，建议直接看一下：

QK^T

---

### 3. reshape / view

也是必须。

因为 multi-head attention 要拆 head。

例如：

原来：

```python
(B, T, hidden_dim)
```

变：

```python
(B, T, heads, head_dim)
```

再换顺序：

```python
(B, heads, T, head_dim)
```

典型代码：

```python
q = q.view(B, T, n_heads, head_dim)
q = q.transpose(1, 2)
```

为什么？

因为 attention 希望 head 维提前：

```python
(B, heads, T, d)
```

便于 batch matrix multiplication。

---

### 4. batch matrix multiplication（非常重要）

你之前学矩阵乘法：

```python
A @ B
```

是二维：

```python
(M, P)
@
(P, N)
```

但 LLM：

> 一次算很多矩阵。

例如：

```python
(B, heads, T, d)
@
(B, heads, d, T)
```

PyTorch 自动理解：

> 前两维 `(B, heads)` 当 batch

对每个 head 分别做矩阵乘法。

所以：

```python
torch.matmul(A, B)
```

其实是在做：

```text
for b in batch:
    for h in heads:
        A[b,h] @ B[b,h]
```

这是理解 attention 的关键直觉。

---

### 5. broadcasting（必须）

这个很多人卡住。

因为 mask、bias、scale 都在 broadcast。

例如：

attention mask：

```python
scores.shape
(B, heads, T, T)

mask.shape
(T, T)
```

直接：

```python
scores + mask
```

居然成立。

因为 broadcasting：

PyTorch 自动扩展：

```python
(T,T)

→

(1,1,T,T)

→

(B,heads,T,T)
```

即：

> 小 tensor 自动复制到大 tensor 的维度。

所以 broadcasting 本质：

> “自动补维 + 自动重复”。

---

## 第二层：很快要掌握（写 LLM 会频繁见）

### `unsqueeze()`

加维度：

```python
x.shape
(128,768)

x.unsqueeze(0)
(1,128,768)
```

常见于：

```python
mask.unsqueeze(0).unsqueeze(0)
```

把：

```python
(T,T)
```

变：

```python
(1,1,T,T)
```

方便 broadcast。

---

### `squeeze()`

删掉 size=1 的维度。

---

### indexing / slicing

比如：

```python
x[:, -1, :]
```

取最后 token。

GPT sampling 常见。

---

### `contiguous()`

和 `transpose/view` 有关系。

因为：

```python
transpose
```

很多时候只是改 memory view。

所以：

```python
x.transpose(...).view(...)
```

可能报错。

需要：

```python
x.transpose(...).contiguous().view(...)
```

不过现在先不用深究。

---

## 一个学习优先级（推荐）

按顺序：

1. `shape`
2. `reshape/view`
3. `transpose/permute`
4. batch matrix multiplication (`matmul`)
5. broadcasting
6. indexing/slicing
7. unsqueeze/squeeze
8. contiguous

---

如果目标是 **手写 mini GPT**，我会说：

> 这几个比学复杂数学更先卡住人。

因为很多人不是不懂 attention 数学，而是：

> 看不懂 tensor shape 怎么流动。

典型障碍：

```python
q = q.view(B,T,H,D).transpose(1,2)
attn = q @ k.transpose(-2,-1)
```

脑子直接炸：

> “这 shape 到底变成什么了？”

而一旦能 mentally trace shape：

```text
(B,T,C)
→
(B,T,H,D)
→
(B,H,T,D)
→
(B,H,T,T)
```

Transformer 会突然变得非常清晰。
