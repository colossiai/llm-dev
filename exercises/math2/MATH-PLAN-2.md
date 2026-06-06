# MATH-PLAN-2: 手写 LLM 还需要哪些数学和基础知识

## Context

承接 [MATH-PLAN-1](../MATH-PLAN-1.md) 已掌握的 9 个基础概念（向量 / 矩阵
乘法 / 点积 / 线性变换 / 维度 / 基底 / 投影 / 导数 / 梯度）。本文档列出
**从基础数学走到"手写一个最简 LLM"（nanoGPT 级别的 decoder-only
Transformer）** 还需要补的全部知识，并给出推荐学习顺序。

---

## 一、数学补充

| 主题 | 用在哪 | 优先级 |
|---|---|---|
| **概率分布 + Softmax** | LLM 输出 = 词表上的概率分布；softmax 把 logits → 概率 | ★★★ |
| **交叉熵 (Cross-Entropy)** | LLM 训练的 loss 函数本身 | ★★★ |
| **链式法则 (Chain Rule)** | 反向传播的核心；理解 `.backward()` 怎么工作 | ★★★ |
| **极大似然估计 (MLE)** | next-token prediction 训练目标的理论基础 | ★★ |
| **KL 散度 / 熵 / Perplexity** | 评估指标、RLHF/DPO 对齐训练 | ★★ |
| **矩阵的秩 (rank)** | 看懂 LoRA 等参数高效微调 | ★ |
| **范数 L1/L2** | 正则化、weight decay、梯度裁剪 | ★ |

---

## 二、深度学习基础（从无到有，5 件套）

1. **神经网络结构** — `nn.Linear` 全连接层、激活函数 (ReLU / GELU / SiLU)
2. **损失函数** — `nn.CrossEntropyLoss`
3. **优化器** — SGD → Adam → **AdamW**（LLM 标配）
4. **归一化** — **LayerNorm** 或 **RMSNorm**
5. **训练循环工程** — `Dataset` / `DataLoader` / mini-batch、梯度清零、
   梯度裁剪、学习率调度 (warmup + cosine)

---

## 三、Transformer 专属（重头戏）

### 模型核心 — 一个 Transformer Block 的构成

1. **缩放点积注意力**：`softmax(Q @ K^T / √d) @ V` ← MATH-PLAN-1 的点积
   直接用上
2. **Multi-Head Attention** — 多组注意力并行
3. **Causal Mask** — "只能看前面 token"，decoder-only LLM 的关键
4. **Feed-Forward Network (FFN)** — 两层 Linear + 激活
5. **残差连接 + Pre-norm**：
   - `x = x + attn(norm(x))`
   - `x = x + ffn(norm(x))`

### 数据 / 输入侧

- **Tokenization (分词)** — BPE / SentencePiece；起步用 `tiktoken` 库
- **位置编码 (Positional Encoding)** — Sinusoidal（经典）或 **RoPE**
  （LLaMA 系主流）

### 训练目标

- **自回归语言建模** — 用 `x[:-1]` 预测 `x[1:]`，teacher forcing
- **shift labels** 的标签错位技巧

### 推理 / 生成

- **采样策略**：Greedy / Temperature / Top-k / Top-p (nucleus)
- **KV cache** — 推理加速（最后再学）

---

## 四、PyTorch 必备 API

```
nn.Module                         # 自定义模型的基类
nn.Embedding                      # token → 向量 (已用过)
nn.Linear                         # 矩阵乘法 + bias
nn.LayerNorm                      # 归一化
nn.functional.softmax / cross_entropy
torch.utils.data.Dataset / DataLoader
torch.optim.AdamW
torch.no_grad() / model.train() / model.eval()
.to(device)                       # Intel Mac → device='cpu'
torch.save / torch.load           # checkpoint
```

---

## 五、推荐学习路径

```
现在: 向量 / 矩阵 / 点积 / 梯度 ✅ (MATH-PLAN-1)
   ↓
A. 概率 + Softmax + 交叉熵 + 链式法则           (1-2 天)
   ↓
B. 用 PyTorch 写一个 MNIST 分类器                (1-2 天)
   关键: 跑通 nn.Module + DataLoader + 训练循环
   ↓
C. Tokenizer + Embedding + 位置编码              (2-3 天)
   ↓
D. 单独实现 Multi-Head Causal Attention          (2-3 天)
   ← 这是最难的一步, 静下心吃透
   ↓
E. 拼 TransformerBlock → 堆 N 层 → 完整模型      (2 天)
   ↓
F. 在小数据集 (莎士比亚字符级) 上训练 + 生成     (3-5 天)
   ↓
🎉 最简 LLM 诞生
```

---

## 六、推荐资源

- **Karpathy "Let's build GPT from scratch"** YouTube 视频 — 这一条视频
  就够你从零打通到底
- **nanoGPT** (github.com/karpathy/nanoGPT) — 训练 + 推理总共 ~300 行
  PyTorch
- **3Blue1Brown** 神经网络系列 — 直觉建立神器

---

## 七、关于硬件（Intel Mac, CPU-only）

训练真模型在 CPU 上会很慢。建议：

- 学习阶段用 **极小模型** (`n_layer=4`, `n_head=4`, `d_model=128`)
  + **小数据集** (莎士比亚 ~1MB)
- 想训练正经规模时再上 Colab / Lightning Studio 借 GPU
