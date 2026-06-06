
# LLM attention中， QK^T是token矩阵， 那么这是针对一句话，还是针对几万字的文章？如何文章太长怎么办


这是一个非常核心的问题，因为它直接关系到：

> **Transformer 为什么会有 context window（上下文长度限制）**

先说结论：

> 在标准 Transformer / LLM 里，`QK^T` 是**针对当前 context window 内所有 token** 做的，而不是无限长文章。

比如：

* 一句话：10 个 token → attention 是 `10×10`
* 一篇文章：2000 token → attention 是 `2000×2000`
* 超长上下文：128k token → attention 是 `128k×128k`

---

## 1. 回忆一下 `QK^T` 在算什么

假设输入一句话：

> “I love machine learning”

token 化后：

```text
[I] [love] [machine] [learning]
```

4 个 token。

embedding 后形成矩阵：

```text
X ∈ R^(seq_len × hidden_dim)

(4 × hidden_dim)
```

经过线性变换：

```text
Q = XW_Q
K = XW_K
V = XW_V
```

假设：

```text
hidden_dim = 4096
head_dim = 128
```

某个 head：

```text
Q: (4 × 128)
K: (4 × 128)
```

然后：

QK^T

维度变成：

```text
(4 × 128)
×
(128 × 4)
=
(4 × 4)
```

意思：

> **每个 token 和所有 token 做相似度比较**

矩阵：

```text
      I love machine learning
I      *
love    *
machine       *
learning
```

每个位置：

```text
score(i,j)
```

代表：

> 第 i 个 token 应该关注第 j 个 token 多少。

---

## 2. 如果是一篇文章呢？

假设：

```text
seq_len = 5000 token
```

则：

```text
Q: (5000 × 128)
K: (5000 × 128)
```

attention score：

```text
5000 × 5000
```

即：

```text
25,000,000
```

个 attention 分数。

因为：

n^2

attention 复杂度：

```text
O(n²)
```

这是 Transformer 最大的问题之一。

因为：

> token 越长，成本平方增长。

例如：

| token 长度 | attention matrix |
| -------: | ---------------: |
|       1k |               1M |
|      10k |             100M |
|     100k |              10B |

所以：

> **不能无限把文章塞进去。**

---

## 3. 那 LLM 是怎么处理“几万字文章”的？

关键：

> **它只看 context window 内的 token**

比如模型：

* 8k
* 32k
* 128k
* 1M（极少数）

context window：

> 当前模型一次能看到的 token 数。

例如：

假设 32k token。

一本书：

```text
300k token
```

模型不会：

```text
QK^T over 300k token
```

而是：

只看：

```text
最近 32k token
```

超出部分：

> 看不到。

所以你常说：

> “模型忘了前面讲什么”

本质就是：

> 前面的 token 被 context window 挤掉了。

---

## 4. 为什么说 GPT 会“遗忘”

比如：

你写了：

```text
第1章：主角叫 John
...
（50k token）
...
第20章
```

若模型窗口：

```text
32k
```

第1章已经不在窗口里。

所以：

```text
John 是谁？
```

模型可能乱答。

因为：

> attention 根本没有看到。

它没有做：

```text
QK^T(全部历史)
```

而是：

```text
QK^T(当前窗口)
```

---

## 5. 那超长文章怎么办？

工业界有几种办法。

### 方法1：滑动窗口（最常见）

切块：

```text
chunk1
chunk2
chunk3
```

比如每块：

```text
4k token
```

然后：

```text
summarize(chunk1)
summarize(chunk2)
```

再汇总。

很多 RAG 系统这么做。

---

### 方法2：RAG（检索增强）

不是让模型读完整本书。

而是：

先检索：

> 和问题相关的段落

例如问：

> John 的父亲是谁？

向量检索找到：

```text
第2章相关段落
```

只把相关内容塞进 context。

而不是：

```text
全部 300k token
```

---

### 方法3：Sparse Attention（稀疏 attention）

不是：

```text
所有 token 两两比较
```

即不是：

```text
n × n
```

而是：

只关注局部：

```text
附近 token
```

例如：

```text
token i
只看 i±256
```

复杂度下降。

像：

* Google 的 long-context 模型
* 一些长上下文 Transformer

会做类似优化。

---

### 方法4：KV Cache（推理阶段）

聊天时不是每次重算全部：

之前 token 的：

```text
K,V
```

缓存起来。

新 token 来时：

```text
只算新的 Q
```

这是为什么 GPT 打字看起来很快。

但：

> KV cache 不解决超长上下文问题。

只是减少重复计算。

---

## 6. 从线代视角理解

你可以把：

QK^T

理解成：

> **一个 seq_len × seq_len 的 token 关系图**

若：

```text
n = 5
```

就是：

```text
5×5
```

若：

```text
n = 100000
```

就是：

```text
100000×100000
```

本质：

> 每个 token 都问：

> “我应该关注谁？”

所以 attention 本质是：

> **token 对 token 的全连接关系矩阵**

也因此：

> 长文本非常昂贵。

一句话总结：

> **标准 LLM 并不是对“整篇无限长文章”做 `QK^T`，而是对“当前 context window 中的全部 token”做 attention；窗口越长，`QK^T` 越大，成本按 `n²` 爆炸。**
