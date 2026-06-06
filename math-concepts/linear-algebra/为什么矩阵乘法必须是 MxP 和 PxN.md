# 为什么矩阵乘法必须是 MxP 和 PxN

因为**矩阵乘法本质是在做“连续线性变换（空间变换）”**，所以中间维度必须对得上。

先给一句话：

> `M×P` 和 `P×N` 能相乘，是因为前一个矩阵的“输出空间”必须等于后一个矩阵的“输入空间”。

---

## 1. 从最机械的计算规则看

假设：

```math
A \in \mathbb R^{M\times P}
```

```math
B \in \mathbb R^{P\times N}
```

结果：

```math
C = AB \in \mathbb R^{M\times N}
```

为什么？

因为：

矩阵乘法定义是：

```math
C_{ij}
=
\sum_{k=1}^{P}
A_{ik}B_{kj}
```

这里有一个关键：

> A 的每一行长度 = P
> B 的每一列长度 = P

只有长度一样，才能做点积。

比如：

```math
[2,1]
\cdot
[5,6]
=
2\times5+1\times6
```

如果一个长度 2，一个长度 3：

```math
[2,1]
\cdot
[5,6,9]
```

就根本不知道怎么乘。

所以：

> **矩阵乘法成立，本质要求：行向量长度 = 列向量长度**

这就是：

```math
(M\times P)(P\times N)
```

中间那个 `P` 必须一致。

---

## 2. 更重要：几何直觉（LLM最有用）

矩阵不是数字表，而是：

> **空间变换器**

比如：

一个矩阵

```math
W \in \mathbb R^{P\times N}
```

意思：

> 把一个 P 维向量，变换成 N 维向量

即：

```math
x\in\mathbb R^P
```

经过：

```math
Wx
```

得到：

```math
y\in\mathbb R^N
```

所以：

* 输入维度 = P
* 输出维度 = N

---

现在连续做两个变换。

先：

```math
B:\mathbb R^N\rightarrow\mathbb R^P
```

再：

```math
A:\mathbb R^P\rightarrow\mathbb R^M
```

那么：

```math
AB
```

就是：

> 先 B，再 A

因为：

```math
x
\rightarrow
Bx
\rightarrow
A(Bx)
```

中间结果必须能接起来：

```math
Bx\in\mathbb R^P
```

而：

```math
A
```

必须接受 P 维输入。

所以 A 的输入维度也必须是 P。

于是：

```math
(M\times P)(P\times N)=M\times N
```

中间 `P` 就是：

> **前一个变换的输出空间 = 后一个变换的输入空间**

像水管接口一样：

```text
N维输入
   ↓
B (P×N)
   ↓
P维输出
   ↓
A (M×P)
   ↓
M维输出
```

如果接口不一致：

```text
B 输出 5维
A 需要 3维输入
```

就接不上。

---

## 3. 用你熟悉的单位向量（basis）视角

矩阵可以看成：

> “新的基底向量集合”

例如：

```math
A=
\begin{bmatrix}
2&1\\
0&3
\end{bmatrix}
```

它有两列：

```math
a_1=
\begin{bmatrix}
2\\0
\end{bmatrix}
```

```math
a_2=
\begin{bmatrix}
1\\3
\end{bmatrix}
```

意味着：

> 输入空间必须是 2 维

因为：

任何输入向量：

```math
x=
\begin{bmatrix}
x_1\\x_2
\end{bmatrix}
```

其实是在说：

> 用多少份 (a_1)、多少份 (a_2)

即：

```math
Ax
=
x_1a_1+x_2a_2
```

所以：

> **矩阵有几列，就要求输入向量几维。**

这就是：

```math
M\times P
```

为什么输入必须是 P 维。

---

## 4. LLM 里为什么维度必须匹配

例如 embedding：

假设：

```math
x\in\mathbb R^{4096}
```

投影矩阵：

```math
W_Q
\in
\mathbb R^{4096\times512}
```

表示：

> 4096维 embedding → 512维 Q 空间

如果写错：

```math
W_Q
\in
\mathbb R^{300\times512}
```

就没法乘：

因为：

```text
x = 4096维
W_Q 要求 300维输入
```

接口不一致。

所以：

> **矩阵乘法的维度约束，本质是“空间接口必须兼容”。**

---

一句最终总结：

> `M×P` 能乘 `P×N`，不是数学家硬规定，而是因为：
>
> **点积长度要一致 + 空间变换的输入输出必须接得上。**
