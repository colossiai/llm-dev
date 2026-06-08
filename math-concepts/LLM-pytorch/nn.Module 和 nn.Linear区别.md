# nn.Module 和 nn.Linear有什么区别


很多初学者都会混淆这两个概念。

简单来说：

* **`nn.Module` = 神经网络模块的基类（框架）**
* **`nn.Linear` = 一个具体的神经网络层（线性层）**

关系类似于：

```text
动物 (nn.Module)
│
├── 猫 (nn.Linear)
├── 狗 (nn.Conv2d)
├── 鸟 (nn.LSTM)
└── ...
```

---

## 1. nn.Module 是什么？

`nn.Module` 是 PyTorch 所有神经网络组件的父类。

例如：

```python
import torch.nn as nn

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
```

这里：

```python
MyModel
```

继承了：

```python
nn.Module
```

因此获得了：

* 参数管理
* 自动求导支持
* `.parameters()`
* `.to(device)`
* `.train()`
* `.eval()`
* 模型保存/加载

等功能。

---

## 2. nn.Linear 是什么？

`nn.Linear` 是 PyTorch 已经帮你写好的一个模块。

数学上：

```math
y = Wx + b
```


例如：

```python
linear = nn.Linear(
    in_features=3,
    out_features=2
)
```

表示：

```text
输入维度 3

输出维度 2
```

内部自动创建：

```python
W.shape = (2, 3)

b.shape = (2,)
```

---

使用：

```python
x = torch.tensor([[1.,2.,3.]])

y = linear(x)
```

实际上执行：

```python
y = x @ W.T + b
```

---

## 3. nn.Linear 本身也是 nn.Module

这是关键。

查看：

```python
print(isinstance(
    nn.Linear(3,2),
    nn.Module
))
```

输出：

```python
True
```

因为源码本质类似：

```python
class Linear(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        ...
```

继承关系：

```text
nn.Module
    ↑
nn.Linear
```

---

## 4. 为什么需要 nn.Module？

因为真实神经网络通常由多个层组成。

例如：

```python
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)

        x = self.fc2(x)

        return x
```

这里：

```text
MLP
```

是一个 Module。

里面包含：

```text
fc1
fc2
```

两个 Linear。

结构：

```text
MLP (Module)
│
├── fc1 (Linear)
│
└── fc2 (Linear)
```

---

## 5. parameters() 为什么能找到所有参数？

因为 `nn.Module` 会递归管理子模块。

例如：

```python
model = MLP()
```

执行：

```python
for p in model.parameters():
    print(p.shape)
```

输出类似：

```text
(4,3)
(4,)
(1,4)
(1,)
```

即：

```text
fc1.weight
fc1.bias

fc2.weight
fc2.bias
```

全部自动收集。

这也是为什么训练时只需要：

```python
optimizer = Adam(
    model.parameters()
)
```

而不需要手动把每层参数都传进去。

---

## 6. 一个形象比喻

假设你在搭乐高：

### nn.Linear

是一块积木：

```text
┌──────┐
│Linear│
└──────┘
```

---

### nn.Module

是乐高模型的设计图和容器：

```text
机器人 (Module)

├─ Linear
├─ ReLU
├─ Linear
└─ Softmax
```

你把很多小积木组合起来，形成一个更大的模块。

---

## 7. 在 LLM 里的例子

GPT 中：

```text
GPTModel (Module)
│
├─ Embedding
├─ TransformerBlock
│   ├─ Attention
│   │   ├─ Linear(Q)
│   │   ├─ Linear(K)
│   │   └─ Linear(V)
│   │
│   └─ MLP
│       ├─ Linear
│       └─ Linear
│
└─ Linear(Output)
```

这里几乎所有东西：

* `Embedding`
* `Attention`
* `MLP`
* `TransformerBlock`
* `GPTModel`

都继承自 `nn.Module`。

而其中大量计算层都使用：

```python
nn.Linear
```

实现。

---

### 一句话总结

```text
nn.Module = 神经网络组件的基类（容器）

nn.Linear = 一个具体的线性层

所有 nn.Linear 都是 nn.Module，
但并非所有 nn.Module 都是 nn.Linear。
```

在阅读 GPT/Transformer 源码时，经常会看到：

```python
class Attention(nn.Module):
```

和

```python
self.q_proj = nn.Linear(...)
```

前者是在**定义一个新模块**，后者是在**使用现成的线性层作为模块中的零件**。
