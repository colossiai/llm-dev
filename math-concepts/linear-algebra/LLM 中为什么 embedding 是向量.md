一句话：

> **因为 LLM 需要把“词语/句子”的语义变成机器可以计算的空间位置，而“向量”天然就是空间中的坐标。**

重点不是「为什么是数学上的向量」，而是：

> **为什么语义要放进一个空间（vector space）里。**

---

## 1. 先从“词不能直接算”开始

假设模型看到：

* cat
* dog
* king
* queen

计算机只能处理数字。

一种最直觉的方法是编号：

```text
cat = 1
dog = 2
king = 3
queen = 4
```

但这里有一个巨大问题：

数字之间没有语义。

模型会误以为：

```text
queen(4) 比 king(3) 大一点
dog(2) 是 cat(1) 的两倍
```

这些完全没意义。

---

## 2. One-hot encoding 也不够

另一种办法：

假设词表只有 4 个词：

```text
cat   = [1,0,0,0]
dog   = [0,1,0,0]
king  = [0,0,1,0]
queen = [0,0,0,1]
```

这叫 one-hot。

优点：

* 可计算
* 唯一标识词

但问题：

**词之间没有关系。**

看距离：

```text
cat ↔ dog
cat ↔ king
king ↔ queen
```

距离全一样。

模型不知道：

```text
cat 更像 dog
king 更像 queen
```

---

## 3. Embedding 的核心：把词放进“语义空间”

于是我们希望：

> **相似意思的词，在空间中靠近。**

比如二维空间（真实情况是几百维）：

```text
             queen
               ●

      king ●

dog ●

cat ●
```

这里：

* cat 和 dog 靠近
* king 和 queen 靠近
* animal 和 royalty 分开

此时每个词都有坐标：

```text
cat   = [1.2, -0.3]
dog   = [1.5, -0.1]
king  = [5.2, 3.0]
queen = [5.5, 3.4]
```

这些坐标：

> **就是 embedding 向量。**

本质：

> embedding = semantic coordinate（语义坐标）

---

## 4. 为什么一定是“向量”？

因为向量有几个特别强大的性质。

### (1) 可以算相似度

例如 cosine similarity：

如果方向很接近：

```text
cat ≈ dog
```

夹角小，相似度高。

如果差很多：

```text
cat ↔ democracy
```

夹角大。

于是模型能判断：

> 哪些词语义更接近。

---

### (2) 可以线性变换（超级关键）

Transformer 本质上疯狂做：

```text
矩阵 × 向量
```

即：

```text
y = Wx
```

```math
\mathbf{y} = W\mathbf{x}
```


如果 embedding 是向量：

模型就能：

* 旋转
* 拉伸
* 投影
* 压缩

在语义空间中移动它。

例如：

> “apple”

经过上下文后：

* fruit meaning
* company meaning

会移动到不同区域。

所以：

> **Transformer 是在不断变换 embedding 向量的位置。**

这就是你之前学的：

> 矩阵乘法 = 空间变换

在 LLM 中的真正落地。

---

### (3) 可以表达关系

经典例子：

```text
king - man + woman ≈ queen
```

不是魔法，而是：

空间里学出了方向。

例如：

```text
male → female
```

是一条稳定方向。

于是：

```text
king + (female direction)
≈ queen
```

即：

> **语义关系变成几何关系。**

---

## 5. 为什么是高维向量？

二维太弱。

现实语义很多维度：

比如：

“apple”可能同时包含：

* food
* sweetness
* tech
* luxury
* productivity
* western culture

所以需要很多维度。

真实模型：

* 768
* 1024
* 4096
* 8192

维 embedding 很常见。

每一维未必可解释：

```text
x1 = furry-ness?
x2 = political-ness?
x3 = abstractness?
```

通常不是人类可读。

而是：

> **模型自动学出的语义坐标轴。**

---

## 6. 更深层理解：为什么不是别的结构？

你可能会问：

> 为什么不是树？图？数据库？

因为训练目标是：

> **可微分 + 可优化。**

神经网络靠：

梯度下降学习。

向量：

* 连续（continuous）
* 可微（differentiable）
* 可矩阵运算
* GPU 极其擅长

所以：

> **向量空间是深度学习最自然的数据表示。**

---

一句非常重要的话：

> **Embedding 不是“词的编号”，而是“词在语义空间中的坐标”。**

而：

> **LLM 本质上是在高维语义空间中移动、组合、变换这些向量。**
