# 介绍LLM入门需要的微积分知识, 他们在LLM中的应用，为什么不能没有这些知识


如果目标是**学懂 LLM（尤其是调试模型、理解论文、训练机制）**，微积分不用学成数学系水平，但有一套“最小可用集”。

一句话先说结论：

> **LLM 里的微积分，本质是在回答：模型怎么知道自己错了，以及怎么一步步变好。**

线性代数负责“表示和变换”，微积分负责“优化和学习”。

---

# 1. 为什么 LLM 需要微积分？

训练 LLM 时，本质是：

> 调参数，让错误越来越小。

模型：

```text
输入 x → Transformer → 输出预测 ŷ
```

和真实答案：

```text
y
```

之间有误差：

```text
loss = 错多少
```

问题来了：

> **参数应该往哪个方向改，才能让 loss 下降？**

这就是微积分存在的理由。

因为：

> **导数 = 改一点后，结果变化多少**

---

## 一个直觉例子

想象：

你站在山里，闭着眼。

目标：

> 下山（loss 最小）

你怎么办？

摸一下脚边：

* 左边更低？
* 右边更低？

于是：

> 朝下降最快方向走。

这个“坡度”就是：

> **gradient（梯度）**

而梯度来自：

> **微积分（导数）**

---

# 2. LLM 微积分最小知识树

按重要程度排序：

1. 导数 derivative
2. 偏导数 partial derivative
3. 梯度 gradient
4. 链式法则 chain rule
5. 多变量函数
6. 优化（gradient descent）
7. 概率分布相关函数（softmax/log）

---

# 3. 导数：变化率（最核心）

先理解：

```math
f'(x)=\lim_{h\to0}\frac{f(x+h)-f(x)}{h}
```

本质：

> **改一点 x，结果变化多少**

例如：

```math
y=x^2
```

导数：

```math
\frac{dy}{dx}=2x
```


意思：

当：

```text
x = 10
```

时：

```text
导数 = 20
```

表示：

> x 多一点点，y 会快速增加。

---

## 在 LLM 中对应什么？

假设参数：

```text
w
```

loss：

```text
L(w)
```

我们想知道：

> 改一点 w，loss 怎么变？

所以求：

```math
\frac{dL}{dw}
```

意思：

> 参数变化，对错误的影响。

如果：

```text
dL/dw > 0
```

说明：

> w 再变大会更糟。

所以：

> 往反方向走。

这就是训练。

---

# 4. 偏导数：因为参数很多

LLM 不只有一个参数。

而是：

```text
10B
70B
100B+
```

参数。

loss 是：

```text
L(w1,w2,w3,...)
```

多变量函数。

于是：

> 对每个参数分别求导。

叫：

> 偏导数（partial derivative）

例如：

```math
L(x,y)=x^2+y^2
```

偏导：

```math
\frac{\partial L}{\partial x}=2x
```

和：

```math
\frac{\partial L}{\partial y}=2y
```

在 LLM：

```text
每个 weight 都有自己的梯度
```

例如：

```text
Wq
Wk
Wv
FFN weight
embedding weight
```

都会有：

```text
∂L/∂parameter
```

---

# 5. 梯度：最陡下降方向

把所有偏导组合：

```math
\nabla L=\left(\frac{\partial L}{\partial w_1},\frac{\partial L}{\partial w_2},...,\frac{\partial L}{\partial w_n}\right)
```


这叫：

> gradient

直觉：

> 山坡最陡方向。

因为它告诉：

> 参数往哪改，loss 降最快。

---

## 在 LLM 中

优化器（例如：

PyTorch 的 Adam）做：

```text
拿梯度
→ 更新参数
→ loss下降
```

核心公式：

```math
w_{new}=w-\eta\nabla L
```

其中：

```text
η = learning rate
```

意思：

> 朝反梯度方向走一点。

这是：

> gradient descent

---

# 6. 链式法则：Transformer 为什么能训练

这是最重要的。

因为：

> Transformer 是很多函数串起来。

例如：

```text
embedding
→ linear
→ attention
→ softmax
→ linear
→ FFN
→ loss
```

像流水线：

```text
x → f → g → h → L
```

你想知道：

```text
dL/dw
```

但：

```text
w
```

在很前面。

怎么办？

用：

> chain rule

```math
\frac{dL}{dw}=\frac{dL}{dh}\frac{dh}{dg}\frac{dg}{df}\frac{df}{dw}
```

意思：

> 一层层把“错误信号”传回来。

这就是：

> backpropagation（反向传播）

如果没有链式法则：

> 深度网络根本训不了。

因为：

> 不知道前面的参数该怎么改。

---

## 为什么不能没有？

因为你需要回答：

> attention 出问题，是 Wq 还是 Wk？

你最终会看：

```text
gradient exploding
gradient vanishing
```

这些全是微积分。

没有链式法则：

> 看论文像魔法。

---

# 7. Softmax 为什么需要指数与导数

attention：

```math
\text{softmax}(x_i)=\frac{e^{x_i}}{\sum_j e^{x_j}}
```

为什么用：

```text
e^x
```

因为：

### (1) 保证正数

概率不能负。

### (2) 放大差异

score 大一点：

```text
概率大很多
```

### (3) 可导

关键：

```math
\frac{d}{dx}e^x=e^x
```

非常容易训练。

如果函数不可导：

> gradient 无法传播。

模型学不了。

---

# 8. 为什么需要 log（交叉熵）

LLM 预测下一个 token：

例如：

```text
cat : 0.8
dog : 0.1
fish: 0.1
```

loss：

通常是：

> cross entropy

核心：

L=-\log p

为什么？

因为：

如果预测错：

```text
p=0.001
```

惩罚巨大。

预测对：

```text
p=0.99
```

loss 很小。

所以：

> 模型被强烈推动去提高正确 token 概率。

---

# 9. 如果完全不会微积分，会卡在哪里？

### 卡 1：不懂训练

不知道：

```text
loss.backward()
```

干什么。

---

### 卡 2：不懂 optimizer

例如：

* SGD
* Adam
* momentum

都是：

> gradient 更新规则。

---

### 卡 3：看不懂论文

常见公式：

```text
∂L/∂θ
argmin
expectation
log-likelihood
```

会像天书。

---

### 卡 4：不会 debug

例如：

```text
loss nan
gradient explode
learning stuck
```

需要微积分直觉。

---

# 10. LLM 入门微积分最小学习计划（建议）

按顺序：

### Stage 1 — 导数直觉

学会：

> 改一点 → 变化多少

内容：

* slope
* tangent
* derivative

目标：

> 理解 gradient 是坡度。

---

### Stage 2 — 多变量 + 偏导

目标：

> 理解 billions of parameters

内容：

* partial derivative
* gradient vector

---

### Stage 3 — 链式法则

目标：

> 理解 backprop

这是最关键。

---

### Stage 4 — 指数与 log

目标：

> 理解 softmax + cross entropy

---

### Stage 5 — 梯度下降

目标：

> 理解 optimizer

---

一句话总结：

> **线性代数回答“模型如何表示世界”，微积分回答“模型如何学习”。**

如果缺少微积分：

> 你能用 LLM，但很难真正理解“为什么它能训练起来”。
