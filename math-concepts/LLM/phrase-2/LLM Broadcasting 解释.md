在 LLM / PyTorch / 张量运算里，**broadcasting（广播）** 是：

> **“让不同 shape 的张量自动扩展维度，使它们能做逐元素运算。”**

本质是：

> **不用真的复制数据，框架假装把小张量扩展成大张量。**

---

## 1. 先从最简单例子理解

假设：

```python
x = [1, 2, 3]
y = 10
```

做：

```python
x + y
```

结果：

```python
[11, 12, 13]
```

为什么一个长度 3 的 vector 能和一个 scalar 相加？

因为发生了 **broadcasting**：

框架偷偷把：

```python
10
```

当成：

```python
[10,10,10]
```

于是：

```python
[1,2,3]
+
[10,10,10]
=
[11,12,13]
```

但注意：

> **实际上没有真的复制三个 10**

只是逻辑上扩展。

---

## 2. 在矩阵里

例如：

```python
X.shape = (3,4)
```

表示：

```text
[
 [1 2 3 4]
 [5 6 7 8]
 [9 10 11 12]
]
```

现在：

```python
b.shape = (4,)
```

即：

```text
[100,200,300,400]
```

做：

```python
X + b
```

broadcasting：

```text
[
 [1 2 3 4]
 [5 6 7 8]
 [9 10 11 12]
]

+

[
 [100 200 300 400]
 [100 200 300 400]
 [100 200 300 400]
]
```

得到：

```text
[
 [101 202 303 404]
 [105 206 307 408]
 [109 210 311 412]
]
```

这里：

```python
(3,4) + (4,)
```

自动变成：

```python
(3,4) + (1,4)
```

再扩展：

```python
(3,4)
```

---

## 3. Broadcasting 规则（核心）

从**右往左看维度**：

满足任一条件即可：

1. 相同
2. 有一个是 1
3. 有一个不存在（自动补 1）

例如：

```text
(2,3,4)
(    3,4)
```

自动补：

```text
(2,3,4)
(1,3,4)
```

可以 broadcast。

---

不能 broadcast：

```text
(2,3,4)
(2,5,4)
```

因为：

```text
3 != 5
```

而且都不是 1。

---

## 4. 在 LLM 里哪里出现？

非常多。

例如 attention mask。

假设：

hidden states：

```python
(batch, seq_len, hidden_dim)

(2, 5, 768)
```

mask：

```python
(batch, seq_len)

(2,5)
```

为了加到 attention score：

可能会 reshape：

```python
(2,1,1,5)
```

然后 broadcast 到：

```python
(batch, heads, seq, seq)

(2,12,5,5)
```

意思是：

> **同一个 mask 自动复制到所有 head、所有 query token**

你不用手工：

```python
repeat(12 times)
```

框架自动做。

---

例如：

attention logits：

```python
scores.shape

(batch, heads, seq, seq)

(2,12,5,5)
```

mask：

```python
mask.shape

(2,1,1,5)
```

相加：

```python
scores + mask
```

发生：

```text
(2,12,5,5)
(2, 1,1,5)
```

broadcast：

```text
(2,12,5,5)
(2,12,5,5)
```

于是每个 head 都加同样 mask。

这在 transformer 极其常见。

---

## 5. 为什么 broadcasting 对 LLM 很重要？

因为：

Transformer 全是批量矩阵运算。

比如：

* bias 加法
* attention mask
* positional encoding
* layer norm 参数 γ β
* multi-head scaling
* batch 训练

全部依赖 broadcasting。

否则你会写很多：

```python
repeat()
expand()
tile()
```

又慢又耗内存。

---

一个很好的心智模型：

> **broadcasting = 自动把低维张量“铺满空间”以对齐 shape**

比如：

```text
(5,)
```

像一条线。

broadcast 到：

```text
(3,5)
```

像复制成 3 行。

或者：

```text
(1,1,5)
```

扩成：

```text
(8,12,5)
```

像把一个模板贴满整个 batch/head 空间。

如果你已经在学 LLM tensor shape，broadcasting 基本是下一步必须掌握的东西，因为很多 `(batch, heads, seq_len, hidden_dim)` 的代码看不懂，本质都是 shape + broadcasting。
