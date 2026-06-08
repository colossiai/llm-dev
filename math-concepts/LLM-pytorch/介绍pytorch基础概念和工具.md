# 给想入门LLM读者介绍pytorch基础概念和工具


如果目标是**理解 LLM 原理并能够阅读 PyTorch 实现代码**，其实不需要先成为 PyTorch 专家。掌握大约 20% 的 PyTorch 知识，就能看懂 80% 的神经网络和 LLM 教程。

下面按照「学习 LLM 最小知识集」来介绍。

# 1. Tensor（张量）

Tensor 是 PyTorch 最核心的数据结构。

可以理解为：

| 数学概念 | PyTorch   |
| ---- | --------- |
| 标量   | 0维 Tensor |
| 向量   | 1维 Tensor |
| 矩阵   | 2维 Tensor |
| 多维数组 | Tensor    |

例如：

```python
import torch

x = torch.tensor([1, 2, 3])

print(x)
```

输出：

```python
tensor([1, 2, 3])
```

---

二维矩阵：

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])
```

形状：

```python
print(x.shape)
```

输出：

```python
torch.Size([2, 2])
```

LLM 中几乎所有东西：

* 输入 Token
* Embedding
* Attention
* 权重矩阵
* Loss

本质都是 Tensor。

---

# 2. Shape（形状）

看懂 shape 非常重要。

例如：

```python
x = torch.randn(3, 4)
```

表示：

```text
3行4列
```

shape：

```python
(3, 4)
```

---

LLM 经常出现：

```python
(batch_size, seq_len, hidden_size)
```

例如：

```python
(32, 128, 768)
```

含义：

```text
32个句子

每个句子128个token

每个token用768维向量表示
```

---

# 3. Tensor运算

加法：

```python
a + b
```

乘法：

```python
a * b
```

矩阵乘法：

```python
a @ b
```

或者：

```python
torch.matmul(a, b)
```

---

例如：

```python
x = torch.tensor([[1,2]])

w = torch.tensor([
    [3],
    [4]
])

y = x @ w
```

计算：

```text
1×3 + 2×4

= 11
```

这就是神经网络最基本的计算。

---

# 4. 自动求导（Autograd）

这是 PyTorch 最强大的功能。

例如：

```python
w = torch.tensor(2.0, requires_grad=True)

loss = (w - 5) ** 2
```

计算：

```text
loss = (2-5)^2
     = 9
```

---

求导：

```python
loss.backward()
```

查看梯度：

```python
print(w.grad)
```

输出：

```python
-6
```

因为：

\frac{d}{dw}(w-5)^2 = 2(w-5)

当：

```text
w=2
```

时：

```text
2(2-5)

= -6
```

---

这就是反向传播的核心。

LLM 训练过程本质上就是：

```text
前向传播

↓

计算Loss

↓

backward()

↓

得到梯度

↓

更新参数

↓

继续训练
```

---

# 5. nn.Module

PyTorch 用它表示神经网络。

例如：

```python
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)
```

使用：

```python
model = MyModel()
```

---

实际上：

```python
nn.Linear(3,1)
```

就是：

```text
y = Wx + b
```

线性回归。

---

# 6. Parameters（参数）

查看模型参数：

```python
for p in model.parameters():
    print(p.shape)
```

例如：

```text
(1,3)
(1,)
```

对应：

```text
W
b
```

---

LLM 的几十亿参数，本质上也是这些 Tensor。

例如：

```text
7B模型

≈ 70亿个数字
```

训练就是不断调整这些数字。

---

# 7. Optimizer（优化器）

负责更新参数。

最常见：

```python
torch.optim.AdamW
```

例如：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)
```

训练：

```python
loss.backward()

optimizer.step()

optimizer.zero_grad()
```

含义：

```text
1. 算梯度

2. 更新参数

3. 清空梯度
```

---

# 8. Dataset 和 DataLoader

训练数据通常很多。

PyTorch 提供：

```python
Dataset
```

和：

```python
DataLoader
```

例如：

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

每次返回：

```text
32条训练数据
```

---

LLM 训练时：

```text
数十亿 Token

↓

切成 Batch

↓

送入GPU训练
```

---

# 9. GPU加速

把 Tensor 放到 GPU：

```python
device = "cuda"

x = x.to(device)

model = model.to(device)
```

---

查看 GPU：

```python
torch.cuda.is_available()
```

---

LLM 训练几乎全部依赖 GPU。

例如：

* NVIDIA H100
* NVIDIA B200
* NVIDIA A100

这些设备专门加速 Tensor 运算。

---

# 10. 一个完整训练循环

这是所有深度学习的经典模板：

```python
for x, y in loader:

    pred = model(x)

    loss = criterion(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
```

流程图：

```text
输入数据

↓

模型预测

↓

计算Loss

↓

反向传播

↓

获得梯度

↓

更新参数

↓

下一轮
```

---

# 11. 学习 LLM 必须掌握的 PyTorch 模块

优先级如下：

### 第一阶段（必须掌握）

```python
torch.tensor
```

```python
shape
```

```python
reshape
```

```python
transpose
```

```python
matmul
```

```python
softmax
```

```python
autograd
```

```python
backward
```

```python
nn.Module
```

```python
nn.Linear
```

```python
optim.AdamW
```

---

### 第二阶段（Attention 必须）

```python
einsum
```

```python
bmm
```

```python
masked_fill
```

```python
tril
```

```python
view
```

```python
permute
```

这些是实现 Transformer 时最常见的操作。

---

### 第三阶段（阅读 LLM 源码）

```python
nn.Embedding
```

```python
nn.LayerNorm
```

```python
nn.MultiheadAttention
```

```python
nn.GELU
```

```python
nn.CrossEntropyLoss
```

掌握这些后，基本就能看懂：

* PyTorch 教程
* Hugging Face Transformers 模型代码
* nanoGPT 源码
* 大部分 Transformer 实现

# 给 LLM 学习者的最短路径

建议顺序：

```text
Tensor
↓
Shape
↓
矩阵乘法
↓
自动求导
↓
Linear Regression
↓
MLP
↓
PyTorch训练循环
↓
Softmax
↓
Cross Entropy
↓
Embedding
↓
Self-Attention
↓
Transformer
↓
GPT
↓
LLM
```

如果你的目标是「真正理解 GPT 是怎么工作的」，那么学到 **Tensor + Autograd + nn.Module + Softmax + CrossEntropy + Embedding + Attention** 这一层就已经进入核心区域了。剩下的大部分内容都是这些基础模块的组合与扩展。

