# LLM了解了基本attention的 Q,K,V, 那么下一步是学习 softmax吗？

是，**下一步非常适合学 softmax**，但最好是：

> **Q/K/V → attention score（QKᵀ）→ softmax → weighted sum(V)**

也就是说，不要把 softmax 单独学，而是放在 attention 流程里学。

你现在的位置大概是：

```text
Embedding
   ↓
Linear projection (Wq, Wk, Wv)
   ↓
Q, K, V
   ↓
QK^T  ← attention score
   ↓
Softmax   ← 下一步
   ↓
加权平均 V
   ↓
输出 token representation
```

attention 的核心公式：

\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V

你已经理解了前半段：

* Q/K = 在语义空间里找相关方向
* QKᵀ = token 与 token 的相关性打分（similarity score）

下一步就是：

> **为什么这些 score 需要 softmax？**

---

## 1. softmax 在 attention 里干什么？

假设：

```text
QK^T score:

[8, 2, -1]
```

意思：

```text
token A 对三个 token 的关注程度

8   → 很相关
2   → 一般相关
-1  → 不太相关
```

但问题是：

### score 不是概率

它只是“原始打分（logits）”。

不能直接当权重：

```text
8 + 2 + (-1) = 9
```

没有意义。

所以要变成：

```text
[0.997, 0.002, 0.001]
```

满足：

```text
全部 >= 0
总和 = 1
```

这样才像：

> “注意力预算分配”

比如：

```text
97% 注意第一词
2% 注意第二词
1% 注意第三词
```

---

## 2. softmax 本质是什么？

一句话：

> **把一组相关性分数，变成概率分布。**

公式：

\mathrm{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}

这里：

* exp(x)（指数）把大数放大、小数压小
* 再归一化（除总和）

例如：

```text
score = [2,1,0]
```

指数：

```text
[e², e¹, e⁰]
≈
[7.39, 2.71, 1]
```

归一化：

```text
[0.66, 0.24, 0.09]
```

于是：

> 分数差距被强化了。

这非常重要。

因为 attention 想做的是：

> **聚焦（focus）**

而不是平均。

---

## 3. 为什么 attention 不直接用 QKᵀ？

因为会出问题。

例如：

```text
score = [100, 50, -20]
```

直接乘：

```text
100V1 + 50V2 -20V3
```

问题：

* 可正可负
* 数值爆炸
* 不可解释
* 没有“注意力比例”的概念

softmax 后：

```text
[0.999..., 0.000..., 0]
```

变成：

> “主要关注谁”

而不是“任意线性组合”。

---

## 4. 几何直觉

你已经学过投影/点积，所以可以这样理解：

```text
Q·K
```

是：

> “方向相似度”

softmax 是：

> **把所有方向相似度做竞争（competition）**

例如：

```text
[8, 2, -1]
```

像：

> “在所有候选 token 中，谁最值得关注？”

softmax 之后：

```text
[0.99, 0.01, 0]
```

模型说：

> “几乎全部注意力给第一个。”

所以：

> **QKᵀ = similarity**
>
> **softmax = attention allocation**

---

## 5. 学习顺序建议

你现在最顺的路线是：

```text
1. Q/K/V 几何意义   ✅
2. QK^T attention score ✅
3. softmax（下一步） ←
4. 为什么除 √dk
5. weighted sum(V)
6. multi-head attention
7. residual + layernorm
8. FFN
9. transformer block
10. training + gradient
```

尤其建议你下一步深入：

> **softmax 为什么用 exp？为什么不用 normalize(x/sum(x))？**

这是很多人真正开始理解 attention 的节点。


# softmax 为什么用 exp？为什么不用 normalize(x/sum(x))

这是一个非常关键的问题。因为这里决定了：

> Transformer 的 attention 为什么能“聚焦（focus）”。

先说结论：

> **不用 `x/sum(x)`，因为它不能稳定地表达“竞争关系”，也不能处理负数，更无法制造强烈注意力。**
>
> **softmax 用 exp，是为了把“相对更相关”的 token 放大，让注意力形成明显偏向。**

看一个例子。

假设 attention score：

```text
[2, 1, 0]
```

这是：

```text
QK^T
```

的结果。

---

## 1. 如果直接 normalize：x/sum(x)

你提议的方法：

\frac{x_i}{\sum_j x_j}

得到：

```text
sum = 3

[2/3, 1/3, 0]
=
[0.67, 0.33, 0]
```

看起来像概率。

但问题来了。

### 问题1：不能处理负数

attention score 经常有负数。

例如：

```text
[8, 2, -1]
```

normalize：

```text
sum = 9

[8/9, 2/9, -1/9]
```

得到：

```text
[0.89, 0.22, -0.11]
```

坏了：

* 出现负概率
* 不再是 attention weight
* weighted sum(V) 会奇怪

attention 要的是：

> “关注程度”

不能是负的。

---

## 2. 问题2：没有“赢家通吃”能力

attention 本质：

> **竞争机制**

希望：

```text
8 比 2 大很多
```

能被放大。

normalize：

```text
[8,2,1]
→
[0.73,0.18,0.09]
```

差距并不大。

但 softmax：

先指数：

```text
[e^8,e^2,e^1]
```

约：

```text
[2980,7.4,2.7]
```

归一化：

```text
[0.997,0.002,0.001]
```

一下变成：

> “几乎全关注第一项”

这正是 attention 想要的效果。

---

## 3. 为什么 exp 会放大差异？

因为指数增长：

```text
1 → 2.7
2 → 7.4
3 → 20
4 → 54
5 → 148
```

增长越来越快。

差值被扩大。

例如：

原始：

```text
2 和 4
差 = 2
```

exp 后：

```text
7.4 和 54
差 ≈ 47
```

于是：

> **相对优势被强化**

而 attention 要做的就是：

> **在很多 token 中挑重点。**

---

## 4. softmax 的真正性质：只关心相对差值

这是最漂亮的一点。

softmax：

\mathrm{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}

如果整体加常数：

```text
[2,1,0]
```

变：

```text
[102,101,100]
```

结果完全一样。

因为：

```text
e^(x+c)
=
e^c e^x
```

公共项约掉。

所以：

> **attention 只关心相对关系，不关心绝对值。**

这是巨大优点。

normalize(x/sum(x)) 不具备这个性质。

例如：

```text
[2,1,0]
→ [0.67,0.33,0]

[102,101,100]
→ [0.34,0.33,0.33]
```

完全变了。

模型会非常不稳定。

---

## 5. 几何直觉

你现在可以这样理解：

```text
Q·K
```

得到：

> 每个 token 的相似度打分。

softmax：

不是简单归一化。

而是：

> **做一个“竞争性放大”**

像选举：

```text
score:
A = 8
B = 2
C = 1
```

normalize：

```text
73%
18%
9%
```

softmax：

```text
99.7%
0.2%
0.1%
```

意思：

> “明显最相关的人，拿走几乎全部注意力。”

这就是 Transformer 能聚焦上下文的原因。

---

所以一句话总结：

> `x/sum(x)` = 普通比例分配
> `softmax(exp)` = **竞争性概率分配（winner emphasis）**

attention 更需要后者，而不是前者。
