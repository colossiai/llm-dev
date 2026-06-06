# 在LLM Transformer 里：QK^T, 这里 K^T 是什么意思

在 Transformer 的 attention 里：

[
QK^T
]

这里的 **(K^T)** 意思是：

> **把 Key 矩阵做转置（transpose）**

也就是：

> **行列互换**

---

## 1. 什么是 transpose（转置）

如果：

[
K=
\begin{bmatrix}
1 & 2 & 3 \
4 & 5 & 6
\end{bmatrix}
]

那么：

[
K^T=
\begin{bmatrix}
1 & 4 \
2 & 5 \
3 & 6
\end{bmatrix}
]

你可以理解成：

> 横着写变竖着写

或者：

> 行(row) ↔ 列(column)

---

## 2. 为什么要 (K^T)

先看 shape（维度）。

假设：

* seq_len = 3（3 个 token）
* hidden_dim = 4
* attention_dim = 2

那么：

Query 和 Key 大概是：

[
Q \in \mathbb R^{3 \times 2}
]

[
K \in \mathbb R^{3 \times 2}
]

含义：

* 3 = token 数
* 2 = q/k embedding 维度

即：

[
Q=
\begin{bmatrix}
q_1 \
q_2 \
q_3
\end{bmatrix}
]

[
K=
\begin{bmatrix}
k_1 \
k_2 \
k_3
\end{bmatrix}
]

每一行是一个 token 的向量。

Attention 的目标是：

> **每个 q 和所有 k 做相似度比较**

即：

[
q_i \cdot k_j
]

我们希望一次矩阵乘法算出：

[
\begin{bmatrix}
q_1\cdot k_1 & q_1\cdot k_2 & q_1\cdot k_3 \
q_2\cdot k_1 & q_2\cdot k_2 & q_2\cdot k_3 \
q_3\cdot k_1 & q_3\cdot k_2 & q_3\cdot k_3
\end{bmatrix}
]

---

## 3. 为什么一定要 transpose

矩阵乘法规则：

> 左矩阵列数 = 右矩阵行数

现在：

[
Q:(3\times2)
]

[
K:(3\times2)
]

直接乘：

[
QK
]

不合法：

[
(3\times2)(3\times2)
]

因为：

[
2\neq3
]

所以要转置：

[
K^T:(2\times3)
]

于是：

[
(3\times2)(2\times3)
]

合法。

结果：

[
3\times3
]

刚好是：

> **token 与 token 的相似度矩阵**

QK^T

---

## 4. 几何直觉：为什么 transpose = 点积机器

最关键的一步来了。

把：

[
Q=
\begin{bmatrix}
q_1\
q_2\
q_3
\end{bmatrix}
]

看成：

> 每一行是 query 向量

而：

[
K^T=
\begin{bmatrix}
| & | & | \
k_1 & k_2 & k_3\
| & | & |
\end{bmatrix}
]

变成：

> 每一列是 key 向量

这样矩阵乘法天然就变成：

> **行 × 列**

而“行 × 列”本质上就是：

> **向量点积**

因此：

[
(QK^T)_{ij}
===========

q_i\cdot k_j
]

意思：

> 第 i 个 query 对第 j 个 token 的关注分数。

---

### 一个非常形象的理解

不 transpose：

> 一堆横着的箭头 × 一堆横着的箭头
> 没法两两比。

transpose 后：

> query 横着放
> key 竖着放

于是：

> 横 × 竖

自动形成：

> **所有 token 两两点积表**

像这样：

|    |    k1 |    k2 |    k3 |
| -- | ----: | ----: | ----: |
| q1 | q1·k1 | q1·k2 | q1·k3 |
| q2 | q2·k1 | q2·k2 | q2·k3 |
| q3 | q3·k1 | q3·k2 | q3·k3 |

这就是 attention score matrix。

一句话记忆：

> **(K^T) 的目的 = 把“所有 key 向量”转成列，好让矩阵乘法一次完成所有 query-key 点积。**
