# 比较 https://github.com/karpathy/nanochat 和 https://github.com/karpathy/nanoGPT 


如果你正在学习 **LLM 从 0 到 1**，特别是结合我们之前讨论的路线（Linear Algebra → PyTorch → Transformer → GPT → ChatGPT），那么 **nanochat 已经取代了 nanoGPT**。

实际上，Andrej Karpathy 自己已经在 nanoGPT 的 README 第一行写道：

> **"nanoGPT has a new and improved cousin called nanochat... nanoGPT is now very old and deprecated."** ([GitHub][1])

下面从几个角度比较。

| 项目                 | nanoGPT            | nanochat                    |
| ------------------ | ------------------ | --------------------------- |
| 发布时间               | 2023               | 2025                        |
| 定位                 | GPT Pretraining 教学 | 完整 ChatGPT Pipeline         |
| 状态                 | 已废弃（Deprecated）    | 当前官方推荐                      |
| 难度                 | ★★☆☆☆              | ★★★★☆                       |
| 代码规模               | ~600 行核心代码         | ~8000 行                     |
| 是否适合研究 Transformer | ⭐⭐⭐⭐⭐              | ⭐⭐⭐⭐                        |
| 是否能训练聊天模型          | ❌                  | ✅                           |
| 是否包含推理(Inference)  | 简单 generate        | 完整 Engine                   |
| Web Chat UI        | ❌                  | ✅                           |
| RL                 | ❌                  | 可选 GRPO                     |
| Fine-tuning        | 很基础                | 完整流程                        |
| Tokenizer          | GPT2 tokenizer     | 自己训练 tokenizer              |
| Evaluation         | loss/perplexity    | CORE、MMLU、GSM8K、HumanEval 等 |

---

# nanoGPT

### 目标

只有一件事情：

> **训练 GPT**

它回答的问题只有：

> GPT 是怎么训练出来的？

整个仓库主要只有几个文件：

```
model.py
train.py
sample.py
config/
```

模型就是：

```
Embedding
↓

Transformer Block

↓

Transformer Block

↓

Transformer Block

↓

Linear

↓

Softmax
```

训练：

```
dataset
↓

tokenize

↓

forward

↓

cross entropy

↓

backward

↓

AdamW

↓

update
```

没有其它东西。

因此它特别适合理解：

* Embedding
* Attention
* MLP
* Residual
* LayerNorm
* Positional Embedding
* Causal Mask
* Cross Entropy
* Backpropagation

几乎每一行代码都对应 GPT 论文。([GitHub][1])

---

# nanochat

nanochat 的目标已经完全不同。

Karpathy 自己描述它是：

> "the simplest experimental harness for training LLMs"

也就是说：

**训练整个 ChatGPT。** ([GitHub][2])

它覆盖了完整生命周期：

```
Raw Text

↓

Tokenizer Training

↓

Pretraining

↓

Mid-training

↓

SFT

↓

RL(Optional)

↓

Inference Engine

↓

Web UI
```

所以它已经不是：

> 如何训练 GPT？

而是：

> 如何做一个完整 ChatGPT。

---

# 最大区别

nanoGPT：

```
文本

↓

GPT

↓

预测下一个 token
```

nanochat：

```
Raw Documents
        │
        ▼
Tokenizer Training
        │
        ▼
Pretraining
        │
        ▼
Instruction Data
        │
        ▼
Mid-training
        │
        ▼
SFT
        │
        ▼
RL (optional)
        │
        ▼
Inference Engine
        │
        ▼
Web Chat
```

所以：

nanoGPT 只包含：

> Transformer

nanochat 包含：

> Transformer + ChatGPT 全流程。

---

# 自动计算超参数

这是 nanochat 一个非常有意思的新设计。

nanoGPT：

```
n_layer = ?

n_head = ?

learning rate = ?

batch size = ?

warmup = ?

weight decay = ?
```

全部自己配。

nanochat：

```
--depth 26
```

其它全部自动推导：

```
hidden size

head 数

learning rate

batch

tokens

训练步数

weight decay

warmup

...
```

Karpathy 希望：

> 一个参数控制整个模型复杂度。([GitHub][2])

---

# Pipeline

nanoGPT：

```
train.py

↓

model.py

↓

sample.py
```

结束。

nanochat：

```
tokenizer

↓

pretrain

↓

midtrain

↓

SFT

↓

RL

↓

evaluate

↓

serve

↓

chat UI
```

---

# 推理(Inference)

nanoGPT：

```
generate()
```

就是不断：

```
forward

↓

argmax/sample

↓

append token
```

没有 KV Cache。

nanochat：

有独立推理引擎：

```
Engine

↓

KV Cache

↓

Prefill

↓

Decode

↓

Tool Use

↓

Chat UI
```

已经接近真实 LLM 服务架构。([GitHub][2])

---

# Evaluation

nanoGPT：

基本只有：

```
loss

perplexity
```

nanochat：

直接支持：

* CORE
* ARC
* MMLU
* GSM8K
* HumanEval

还能自动生成训练报告。([GitHub][2])

---

# 教学价值

这是很多人最关心的。

## nanoGPT

优点：

* 非常容易读
* Transformer 实现极其清晰
* Attention 最容易理解
* 非常适合第一次学习 GPT

缺点：

* 不是真实工业流程
* 缺少 tokenizer、SFT、推理等完整组件

---

## nanochat

优点：

* 完整 ChatGPT 生命周期
* 接近真实训练流程
* 包含 tokenizer、SFT、RL、推理、UI

缺点：

* 代码量约是 nanoGPT 的十几倍
* 不适合作为第一次阅读 Transformer 源码

---

# 对于你的学习路线

结合我们之前已经完成的内容：

* ✅ 向量、矩阵
* ✅ Tensor / Shape
* ✅ Broadcasting
* ✅ Matrix Multiplication
* ✅ Softmax
* ✅ Gradient
* ✅ Chain Rule
* ✅ Computational Graph

以及你下一步准备深入：

* PyTorch
* Attention
* Transformer
* GPT

我建议采用下面的顺序：

```
1. nanoGPT
```

理解：

* model.py
* train.py

直到你能够自己写一个 GPT。

↓

```
2. 自己实现一个 mini GPT
```

包括：

* Embedding
* Multi-head Attention
* MLP
* Training Loop

↓

```
3. 再看 nanochat
```

理解：

* tokenizer
* pretraining
* SFT
* RL
* inference
* WebUI

↓

```
4. 最后阅读 vLLM、llama.cpp 等工业级推理框架
```

## 我的建议

**不要直接从 nanochat 开始。**

虽然 nanochat 是 Karpathy 当前推荐的项目，也是未来的发展方向，但它更像一本关于 **"如何构建完整 ChatGPT 系统"** 的书；而 nanoGPT 更像一本关于 **"Transformer 和 GPT 是如何工作的"** 的书。

对于你目前的学习阶段，**先彻底读懂 nanoGPT，再升级到 nanochat**，学习曲线会更平滑，也更符合你正在构建的数学与 Transformer 基础。

[1]: https://github.com/karpathy/nanogpt?utm_source=chatgpt.com "GitHub - karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. · GitHub"
[2]: https://github.com/karpathy/nanochat/blob/master/README.md?utm_source=chatgpt.com "nanochat/README.md at master · karpathy/nanochat · GitHub"
