# 帮我做一份机器学习 → 深度学习 → LLM 完整学习路径图


如果你的目标是**真正理解 LLM（而不只是调用 API）**，推荐按照下面的路线学习。每一阶段都尽量做到「理解原理 + 手写实现 + 使用 PyTorch 实践」。

# 第一阶段：Python 与数学基础

## Python

掌握：

* 函数
* 类与对象
* 列表推导式
* 生成器
* 装饰器（了解即可）
* Numpy

重点：

```python
import numpy as np

x = np.array([1,2,3])
W = np.array([[1,2],[3,4],[5,6]])

y = x @ W
```

理解：

* 向量
* 矩阵
* 张量

---

## 线性代数

必须掌握：

* 向量（Vector）
* 矩阵（Matrix）
* 矩阵乘法
* 转置（Transpose）
* 内积（Dot Product）
* 维度变换

重点理解：

```math
y = Wx + b
```

因为神经网络本质上就是不断做这个运算。

---

## 概率统计

掌握：

* 概率
* 条件概率
* 独立事件
* 期望
* 方差

重点：

```math
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
```


后面会用于语言模型概率预测。

---

## 微积分

掌握：

* 导数
* 偏导数
* 梯度
* 链式法则

重点：

理解：

> Loss 如何对参数求导。

这是反向传播的基础。

---

# 第二阶段：机器学习基础

目标：

理解模型如何从数据中学习。

---

## Linear Regression

手写：

```python
y = wx + b
```

学习：

* MSE
* Gradient Descent

重点公式：

```math
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2
```

---

## Logistic Regression

理解：

* 二分类
* Sigmoid

概率输出：

[
P(y=1|x)
]

---

## 模型评估

分类：

* Accuracy
* Precision
* Recall
* F1

回归：

* MSE
* MAE

---

## 过拟合

理解：

* Overfitting
* Underfitting
* Train/Test Split

学习：

* L1
* L2 Regularization

---

# 第三阶段：神经网络

这是进入 LLM 的真正入口。

---

## 感知机（Perceptron）

结构：

```text
x1
x2 ---> neuron ---> y
x3
```

本质：

[
Wx+b
]

然后经过激活函数。

---

## 激活函数

学习：

* Sigmoid
* Tanh
* ReLU

重点：

为什么不能只有线性层。

---

## MLP（多层感知机）

结构：

```text
Input
 ↓
Linear
 ↓
ReLU
 ↓
Linear
 ↓
Output
```

手写实现一次。

---

## Backpropagation

重点理解：

* 链式法则
* 梯度传播

这是深度学习最核心知识之一。

---

## PyTorch

掌握：

```python
tensor
```

以及：

```python
view()
reshape()
transpose()
permute()
```

---

还要掌握：

```python
autograd
optimizer
loss
```

例如：

```python
loss.backward()
optimizer.step()
```

---

# 第四阶段：深度学习核心

## Embedding

词变向量：

```text
cat
 ↓
[0.1, 0.5, -0.2, ...]
```

理解：

> GPT 不认识文字，只认识向量。

---

## Softmax

必须彻底理解。

```math
\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}
```


学习：

* 为什么用 exp
* Temperature

你最近已经在学这一部分。

---

## Cross Entropy

LLM 最重要损失函数之一。

理解：

* 概率预测
* NLL
* Cross Entropy

---

## Batch

理解：

```python
(B, T, D)
```

例如：

```python
(32, 128, 768)
```

---

## Broadcasting

必须熟练。

例如：

```python
x + bias
```

自动扩展维度。

---

# 第五阶段：Attention

这是 GPT 的灵魂。

---

## Q K V

理解：

```text
Query
Key
Value
```

计算：

[
QK^T
]

得到注意力分数。

---

## Scaled Dot Product Attention

核心公式：

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

这是 GPT 最重要的公式。

---

## Multi-Head Attention

学习：

```text
Head1
Head2
Head3
...
```

不同 Head 学不同关系。

---

## Masked Attention

理解：

为什么 GPT 看不到未来 Token。

---

# 第六阶段：Transformer

推荐阅读：

Attention Is All You Need

---

掌握：

## Positional Encoding

解决：

> Attention 不知道词序。

---

## Transformer Block

结构：

```text
Input
 ↓
Attention
 ↓
Add & Norm
 ↓
MLP
 ↓
Add & Norm
```

---

## Encoder

用于：

* BERT

---

## Decoder

用于：

* GPT

---

# 第七阶段：LLM

## GPT结构

```text
Embedding
 ↓
Transformer Block × N
 ↓
Linear
 ↓
Softmax
```

---

## Next Token Prediction

例如：

```text
I love machine
```

预测：

```text
learning
```

---

## Tokenization

理解：

* BPE
* WordPiece

例如：

```text
unbelievable
```

可能拆成：

```text
un
believ
able
```

---

## Context Window

理解：

```text
4K
8K
32K
128K
```

代表能看到多少 Token。

---

## KV Cache

理解：

为什么聊天越来越快。

---

# 第八阶段：LLM训练

## Pretraining

海量文本训练。

学习：

* 数据集
* Token
* Cross Entropy

---

## SFT

监督微调。

---

## RLHF

理解：

* 人类反馈
* Reward Model

---

# 第九阶段：推理优化

掌握：

* Temperature
* Top-k
* Top-p
* Beam Search（了解）

---

# 第十阶段：工程实践

建议项目顺序

### 项目1

手写 Linear Regression

---

### 项目2

手写 MLP

---

### 项目3

手写 Single-Head Attention

---

### 项目4

手写 Multi-Head Attention

---

### 项目5

手写 Transformer Block

---

### 项目6

训练一个 Character GPT

例如：

* 莎士比亚文本
* 小说文本

输入：

```text
To be or not
```

输出：

```text
to be
```

---

### 项目7

阅读并实现：

nanoGPT

这是从 Transformer 到 GPT 的最佳实践项目之一。

---

如果按「理解原理」优先级排序，最值得深入钻研的 10 个主题是：

1. 线性代数
2. 梯度与链式法则
3. Gradient Descent
4. 神经网络（MLP）
5. Backpropagation
6. Embedding
7. Softmax
8. Cross Entropy
9. Attention
10. Transformer

当这 10 个主题真正理解后，再看 GPT、Llama、DeepSeek 等 LLM 的论文和源码，难度会明显下降。
