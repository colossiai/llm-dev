# 理解LLM中的矩阵投影（Q/K）


理解 LLM 里的 **Q/K 矩阵投影（projection）**，最容易卡住的地方是：

> 为什么 embedding 好好的，还要乘一个矩阵变成 Q 和 K？

一句话先说：

> **Q/K projection = 把同一个 token embedding 投影到“找关系”的空间里。**

它不是为了“增加信息”，而是为了：

> **改变观察角度（coordinate system / basis）**，让 attention 更容易比较 token 之间的关系。

---

## 1. 从 embedding 开始

假设一句话：

> “The animal didn’t cross the street because it was tired”

模型看到 `it` 时，需要判断：

`it → animal` 还是 `street`？

先有 embedding：

每个 token 是一个向量：

[
x_i \in \mathbb R^{d_{model}}
]

例如：

[
x_i \in \mathbb R^{4096}
]

你可以想象：

```text
animal = [0.2, -1.3, 0.7, ...]
street = [1.1, 0.4, -0.9, ...]
it     = [0.8, -0.2, 0.1, ...]
```

这些 embedding 混着很多信息：

* 语法
* 语义
* 风格
* 位置关系
* 上下文

但 attention 要做的是：

> **谁跟谁相关？**

这时候原始 embedding 空间不一定适合直接比。

所以：

> **先投影到一个更适合“比较关系”的空间。**

---

## 2. 什么叫 projection？

Q/K 的公式：

[
Q=XW_Q
]

[
K=XW_K
]

其中：

* (X): embedding matrix
* (W_Q): Query projection matrix
* (W_K): Key projection matrix

这本质是：

> **线性空间变换**

即：

> **换 basis（换观察角度）**

你之前学的：

矩阵乘法 = 空间变换

这里完全一样。

假设：

[
x=
\begin{bmatrix}
1\
2
\end{bmatrix}
]

乘：

[
W=
\begin{bmatrix}
2&0\
0&0.5
\end{bmatrix}
]

得到：

[
Wx=
\begin{bmatrix}
2\
1
\end{bmatrix}
]

含义：

> x 所在空间被拉伸/压缩了。

有些方向被放大。

有些方向被弱化。

---

## 3. Q/K projection 在几何上是什么？

关键理解：

embedding 空间：

```text
"很多信息混在一起"
```

Q/K 空间：

```text
"为了 attention specially optimized"
```

例如模型学到：

做代词指代时：

```text
number agreement
subject relation
semantic actor
```

更重要。

于是：

[
W_Q, W_K
]

自动学会：

### 放大某些方向

例如：

```text
animate object
subjectness
grammar role
```

### 压缩某些方向

例如：

```text
style
emotion
punctuation preference
```

所以：

同一个 token：

```text
animal
```

经过 projection：

原来：

```text
animal = [语法,语义,风格,位置,...]
```

变成：

```text
animal_Q = [是否像主语,是否可指代,语义类别...]
```

注意：

> 不是人工定义的 feature

而是训练学出来。

---

## 4. 为什么说是“投影”？

因为矩阵乘法可以理解为：

> **向新 basis 做 projection**

如果把矩阵列向量看作 basis：

[
W_Q=
[b_1,b_2,b_3,...]
]

那么：

[
xW_Q
]

可以理解成：

> “x 在这些新方向上的坐标”

即：

> **x 投影到新的语义轴上**

比如：

原空间：

```text
[猫性, 狗性, 颜色, 情绪, 动词倾向]
```

attention 更关心：

```text
[主谓关系, 指代能力, 时间关系]
```

于是：

Q/K projection：

相当于：

```text
重新定义坐标轴
```

---

## 5. Attention 为什么是 Q·K？

因为：

点积：

[
q_i \cdot k_j
]

是在问：

> **方向是否一致**

你之前学过：

点积：

[
a\cdot b
========

|a||b|\cos\theta
]

所以：

attention score：

本质是：

> **“在 attention 空间中，两个 token 是否朝同一个方向”**

方向一致：

```text
相关
```

方向不同：

```text
不相关
```

这就是：

> “在几何空间中找相关方向”

---

## 6. 一个直觉图

想象：

原 embedding：

```text
         情绪
           ↑
           |
   风格 ← token → 语法
           |
           ↓
          语义
```

很混乱。

经过：

[
W_Q,W_K
]

后：

```text
      subjectness
            ↑
            |
coreference → token → actorness
            |
            ↓
       semantic role
```

现在：

`it`

更容易和：

`animal`

方向对齐。

---

## 7. 从线代视角完整表达

其实 attention 是：

[
QK^T
]

展开：

[
(XW_Q)(XW_K)^T
]

进一步：

# [

XW_QW_K^TX^T
]

这非常重要：

说明：

> **attention 是 embedding 空间上的一个双线性相似度(bilinear similarity)**

即：

不是直接：

[
XX^T
]

而是：

> **先换空间，再比较**

所以：

Q/K projection 的核心目的：

> **学习一种更好的 similarity metric**

像是：

从普通欧氏距离：

变成：

> “模型自己学出来的距离函数”

---

### 一句话总结

> **Q/K projection = 用训练出来的矩阵，把 embedding 投影到一个“更适合比较关系”的几何空间，再用点积判断 token 是否方向一致。**

真正的 mental model：

```text
embedding
    ↓
矩阵变换（换basis）
    ↓
attention空间(Q/K)
    ↓
方向相似度(Q·K)
    ↓
谁相关
```


# 这里说的矩阵投影（projection），其实就是点积里面的projection对吗？

对，但要小心：

> **Q/K 的“projection” 和点积里的 projection 有亲缘关系，但不是同一个概念。**

很多教材会把它们混着讲，所以容易绕。

先给结论：

```text
Q/K projection
= 矩阵线性投影 / basis transform（空间变换）

点积 projection
= 一个向量在另一个向量方向上的投影长度
```

它们关系是：

> **Q/K projection 先发生，然后 attention 的点积再利用“投影”思想算相似度。**

---

## 1. 点积里的 projection 是什么？

你之前学过：

[
a\cdot b
========

|a||b|\cos\theta
]

也可以写成：

> “a 在 b 方向上的投影长度 × |b|”

几何图：

```text
      a
     /
    /
   /θ
--/------------> b
```

a 投影到 b：

```text
      a
     /
    /
---+-----------> b
   ↑
 projection
```

公式：

[
\text{proj}_b(a)
================

|a|\cos\theta
]

所以：

[
a\cdot b
========

(\text{projection length})\times |b|
]

这个 projection：

> **是 vector-to-vector 的几何投影**

---

## 2. Q/K projection 不是这个

Q/K：

[
Q=XW_Q
]

这里的：

“projection”

不是：

```text
把向量投影到另一根向量上
```

而是：

> **投影到一个新的坐标系（subspace / basis）**

更像：

```text
原空间
↓
矩阵变换
↓
新空间
```

例如：

二维：

原坐标：

```text
x-axis
y-axis
```

变成：

```text
新轴1
新轴2
```

矩阵：

[
W_Q
]

在做：

> **定义新的语义轴**

比如：

原 embedding：

```text
[语法,语义,风格,位置]
```

投影后：

```text
[主谓关系,实体匹配,代词指代]
```

这是：

> **basis projection / linear projection**

不是点积投影。

---

## 3. 为什么大家都叫 projection？

因为线性代数里：

projection 有很多层意思。

### A. 几何 projection（点积）

向量投到另一向量：

[
\text{proj}_b(a)
]

例如：

影子。

---

### B. 线性 projection（矩阵）

把向量映射到：

> 一个子空间 / 新 basis

例如：

3D → 2D

或者：

高维 → attention head space

例如：

[
4096 \rightarrow 128
]

这是：

> projection matrix

Q/K 属于这个。

---

## 4. 但两者深层是有关联的

虽然不是同一个东西，

但 attention 里：

### Step 1

先：

[
Q=XW_Q
]

[
K=XW_K
]

把 embedding 投影到新空间。

（矩阵 projection）

---

### Step 2

再：

[
QK^T
]

做点积。

而点积本身：

又是在测：

> projection / alignment

即：

```text
Q 在 K 方向上的投影程度
```

所以：

attention 实际是：

```text
先做矩阵投影
再做向量投影
```

非常关键。

---

## 5. 一个最强 mental model

把 attention 想成：

```text
原始 embedding
        ↓
重新定义坐标轴（Q/K matrix projection）
        ↓
进入 attention 空间
        ↓
比较方向一致性（dot-product projection）
        ↓
得到相关性
```

所以：

> **Q/K 的 projection ≠ dot product projection**
>
> 但：
>
> **attention = 矩阵 projection + 点积 projection**

这是最准确的理解。

---

从数学上说：

Q/K 的 projection 更接近：

[
y = Wx
]

而点积 projection：

更接近：

[
\frac{x\cdot b}{|b|}
]

两者不是同一个公式。


# 所以矩阵 projection 其实就是: Wx 线性空间转换

对，**在 LLM / 深度学习语境里，你可以先这样理解**：

> **矩阵 projection ≈ (Wx) 的线性空间变换（linear transformation）**

但我会加一个更精确版本：

> **projection 是一种线性变换，但线性变换不一定是 projection。**

这是线代术语里容易混淆的地方。

---

## 1. 在 LLM 里，你这样理解完全 OK

Q/K：

[
q = W_Qx
]

[
k = W_Kx
]

本质：

> **用矩阵做空间变换**

即：

```text
原 embedding 空间
      ↓ Wx
attention 空间
```

你可以把它想成：

```text
拉伸
压缩
旋转
重新组合维度
换basis
```

例如：

原来：

```text
x =
[语法,语义,风格,位置]
```

经过：

[
Wx
]

变成：

```text
q =
[代词匹配,主语关系,实体相关性]
```

所以在工程语境：

> **projection = learned linear transformation**

这个理解是对的。

---

## 2. 为什么严格数学上不完全等价？

在线性代数里，

“projection” 有一个更严格定义：

它是：

> **投影到某个子空间**

例如：

3D 点投影到平面：

```text
      *
     /
    /
---/-------- plane
```

或者：

投影到 x 轴：

```text
(x,y)
 ↓
(x,0)
```

它满足一个性质：

[
P^2=P
]

叫：

> idempotent（幂等）

因为：

投影一次：

```text
平面上
```

再投影：

```text
还是同一个地方
```

例如：

P^2=P

这是严格数学 projection。

---

## 3. 但 Q/K matrix 不满足

Q/K：

[
W_Q
]

通常：

[
W_Q^2 \neq W_Q
]

因为它不是：

> “扔到一个平面”

而是：

> “任意 learned linear transform”

例如：

* 拉伸
* 压缩
* 旋转
* basis mixing
* 降维

所以严格说：

> Q/K 更像 **linear transform**

而不是数学上 strict projection。

---

## 4. 为什么 ML 圈还是叫 projection？

因为它强调：

> **把向量映射到另一个 representation space**

例如：

4096 → 128：

[
x\in\mathbb R^{4096}
]

[
Wx\in\mathbb R^{128}
]

大家口语上说：

> project x into attention space

意思是：

> **映射到一个新空间**

而不是数学课那个严格 definition。

---

## 5. 一个很好的 mental model

你现在可以这样记：

```text
数学严格定义:
projection = 特殊线性变换

ML/LLM语境:
projection ≈ learned linear transform (Wx)
```

所以：

当你看到：

```text
Q projection
K projection
V projection
```

脑内翻译：

```text
learned space transformation
```

即：

```text
embedding
   ↓ Wx
attention-friendly representation
```

这是非常准确的直觉。
