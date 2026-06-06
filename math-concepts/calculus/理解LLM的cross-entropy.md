# 理解LLM的cross-entropy 


理解 LLM 的 **Cross-Entropy（交叉熵损失）**，最好的方式是从「LLM 实际在做什么」开始。

---

## 1. LLM 本质上在做什么？

假设训练数据里有一句话：

> I love eating ice cream

训练时，LLM 会不断做：

```text
输入: I
预测: love

输入: I love
预测: eating

输入: I love eating
预测: ice

输入: I love eating ice
预测: cream
```

对于每一步，模型都会输出一个 logits 向量：

```text
Vocabulary:
0: cat
1: dog
2: ice
3: cream
...
```

例如预测下一个词时：

```python
logits = [1.2, 0.5, 4.8, 2.1]
```

经过 softmax：

```python
p = [0.025, 0.012, 0.84, 0.123]
```

表示：

```text
cat    2.5%
dog    1.2%
ice   84.0%
cream 12.3%
```

---

## 2. 什么是正确答案（Ground Truth）？

假设真实下一个 token 是：

```text
ice
```

会变成 one-hot：

```text
y = [0, 0, 1, 0]
```

即：

```text
cat    0
dog    0
ice    1
cream  0
```

---

## 3. Cross-Entropy 的公式

交叉熵定义：

```math
H(y,p)=-\sum_i y_i\log p_i
```

因为 y 是 one-hot：

```text
只有正确答案位置是 1
其他全是 0
```

所以公式直接变成：

```math
\text{Loss}=-\log(p_{\text{correct}})
```


这就是 LLM 训练时最常见的损失函数。

---

## 4. 举例

真实答案：

```text
ice
```

模型预测：

```text
cat    0.025
dog    0.012
ice    0.84
cream  0.123
```

正确 token 的概率：

```text
0.84
```

损失：

```python
-loss = log(0.84)
```

约等于：

```python
0.174
```

损失很小。

---

## 5. 如果预测错了？

模型预测：

```text
cat    0.60
dog    0.20
ice    0.10
cream  0.10
```

真实答案还是：

```text
ice
```

那么：

```python
Loss = -log(0.10)
```

约等于：

```python
2.30
```

损失变大。

---

## 6. 为什么要用 log？

如果直接用：

```python
Loss = 1 - p_correct
```

那么：

```text
0.9 → 0.1
0.1 → 0.9
```

惩罚力度不够。

而 log 会放大严重错误：

| 正确概率  | Cross Entropy |
| ----- | ------------- |
| 0.99  | 0.01          |
| 0.9   | 0.10          |
| 0.5   | 0.69          |
| 0.1   | 2.30          |
| 0.01  | 4.61          |
| 0.001 | 6.91          |

可以看到：

```text
越自信地预测错
惩罚越重
```

这是训练非常需要的性质。

---

## 7. 在 LLM 中是怎样计算的？

一句话：

```text
I love eating ice cream
```

会产生多个预测：

| 输入                | 正确答案   |
| ----------------- | ------ |
| I                 | love   |
| I love            | eating |
| I love eating     | ice    |
| I love eating ice | cream  |

每个位置都有一个 Cross-Entropy：

```text
0.2
0.3
0.1
1.5
```

最终：

```python
loss = mean(all_token_losses)
```

即：

```math
\text{Loss}
=
\frac{1}{N}
\sum_{t=1}^{N}
-\log p_t(\text{correct token})
```

---

## 8. 与 Perplexity 的关系

LLM 常见指标：

```math
\text{Perplexity}
=
e^{\text{CrossEntropy}}
```

例如：

```text
CrossEntropy = 1.0
```

则：

```text
Perplexity = 2.718
```

表示：

> 模型平均相当于在约 2.7 个候选 token 中犹豫。

因此：

```text
Cross-Entropy ↓
Perplexity ↓
模型效果 ↑
```

---

## 9. 一句话理解

LLM 的训练过程其实就是：

```text
1. logits
    ↓
2. softmax
    ↓
3. 得到每个 token 的概率
    ↓
4. 取真实 token 的概率
    ↓
5. 计算 -log(p_correct)
    ↓
6. 反向传播更新参数
```

所以：

> **Cross-Entropy 本质上是在惩罚模型没有把足够高的概率分配给正确 token。**
>
> 正确 token 概率越高，Loss 越小；正确 token 概率越低，Loss 越大。
