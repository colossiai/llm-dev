# LLM入门需要的线代知识

如果目标是**学习 LLM（尤其是理解 Transformer、Attention、Embedding、训练过程）**，你不需要成为数学系线代高手，但有一部分线性代数是“高频核心”，有一部分只是“看论文时补”。

我会按重要性排序。

---

# 1. 必学（80%收益）

## A. 向量（Vector）——最重要

必须理解：

* 向量是什么
* 向量空间直觉
* 高维向量
* 长度（norm）
* 方向
* 相似度

为什么？

因为在 LLM：

> **一切几乎都是向量**

例如：

token → embedding vector

一个单词：

```text
"cat"
```

会变成：

```text
[0.12, -0.4, 0.77, ...]
```

768维、4096维之类。

你需要建立直觉：

> “单词是高维空间里的点”

比如：

* king 和 queen 很近
* cat 和 dog 很近
* Singapore 和 Malaysia 较近

---

## B. 矩阵乘法（空间变换）

你刚刚问的这个。

```math
y=Wx+b
```

必须理解：

* 矩阵乘法意义
* 矩阵 = 空间变换
* 为什么矩阵能“抽取特征”
* 多层矩阵组合

因为：

> 神经网络基本就是疯狂做矩阵变换。

例如：

```text
embedding
   ↓
Linear layer
   ↓
attention
   ↓
MLP
```

本质：

```text
x → Wx+b
```

你需要看到：

> “向量经过空间扭曲后，移动到另一个语义区域”

例如：

```text
apple(公司)
```

和

```text
apple(水果)
```

上下文经过变换后会落到不同区域。

---

## C. 点积（dot product）

这是 Attention 的核心。

必须理解：

* dot product 几何意义
* 相似度
* cosine similarity

为什么？

Transformer attention：

QK^T

其实就是：

> “谁和谁更相关？”

例如：

```text
The animal didn't cross the street because it was tired
```

"it"

应该关注：

```text
animal
```

attention 用点积算：

> query 和 key 是否方向相近

即：

> 语义是否接近

---

## D. 线性变换（Linear Transformation）

理解：

> 为什么矩阵是变换

以及：

* rotation
* scaling
* projection
* basis change

不需要证明。

要有直觉：

> “一个 layer 就是在重新表达特征”

---

## E. 维度（dimension）

非常重要。

你要习惯：

```text
(batch, seq_len, hidden_dim)
```

例如：

```text
(32, 2048, 4096)
```

能理解：

* shape
* broadcasting
* tensor dimension

否则读代码会很痛苦。

---

# 2. 第二优先级（理解 Attention/Embedding 很有帮助）

## F. 基底（basis）

理解：

> 坐标系变化

因为 embedding 经常可以理解成：

> 换一种表达空间

例如：

“快乐程度”

“政治倾向”

“技术性”

可能对应某些方向。

理解 basis 后：

> “feature direction”

直觉会很强。

---

## G. 投影（projection）

非常重要但常被忽略。

Attention 可理解为：

> 往某方向投影

比如：

> “句子里哪些 token 更像当前位置关心的语义？”

---

## H. 特征值 / 特征向量（Eigen）

很多人以为必学。

其实：

> **不是学习 LLM 的阻塞点。**

更像：

> 深入 ML 才会遇到。

例如：

* stability
* optimization intuition
* PCA

可以后学。

---

## I. 矩阵分解（SVD）

理解即可，不必推导。

比如：

因为：

* LoRA
* low rank adaptation
* embedding compression

会遇到。

但不是入门门槛。

---

# 3. 可以几乎不学（先跳过）

* 行列式 determinant
* 余子式
* 克拉默法则
* 手算逆矩阵
* 高斯消元细节
* 线代证明题

这些对：

> “看懂 LLM”

帮助极低。

---

# 推荐学习路线（最实用）

顺序：

1. 向量
2. dot product
3. 矩阵乘法 = 空间变换
4. linear transformation
5. basis
6. projection
7. tensor shape
8. eigen intuition
9. SVD intuition

---

一个判断标准：

如果你能直觉理解：

```text
Q = XWq
K = XWk
V = XWv
Attention(Q,K,V)
```

是在做：

> “把 token 映射到不同语义空间，再计算相似度并聚合信息”

那你的线代已经够 LLM 入门了。
