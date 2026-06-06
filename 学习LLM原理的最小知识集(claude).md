# 学习LLM原理的最小知识集

## 核心结论(TL;DR)

**是的,需要先学神经网络基础,但只需最小集**。要点速览:

### 最小知识集(按优先级)

**1. 数学(够用即可)**
- 线性代数:矩阵乘法、张量维度、点积
- 概率:softmax、交叉熵
- 微积分:梯度、链式法则(知道概念)

**2. 神经网络基础**
- MLP(全连接层)、激活函数(ReLU/GELU)
- 反向传播 + 梯度下降(Adam)
- Embedding、LayerNorm、残差连接

**3. 序列建模**
- Tokenization(BPE)
- 自回归 + Next Token Prediction
- 因果掩码

**4. Transformer(重点!)**
- **Attention(Q/K/V)** — LLM的灵魂
- Multi-Head Attention
- Transformer Block 组成
- 位置编码(RoPE)

**5. LLM特有**
- 预训练 / SFT / RLHF(概念)
- KV Cache、采样策略

### 可以跳过

CNN、RNN/LSTM细节、GAN、Diffusion、GNN — 与LLM主线无关。

### 最佳学习路径

1. **3Blue1Brown 神经网络系列**(直观)
2. **Karpathy "Neural Networks: Zero to Hero"**(从零手搓GPT,最有效)
3. **The Illustrated Transformer** 博客
4. 最后读 **Attention Is All You Need** 论文

全职 2-4 周,业余 1-2 月可达"理解LLM原理"水平。

---

## Context

目标:理解LLM(大语言模型)的工作原理。

LLM本质是基于Transformer架构的神经网络,通过自回归方式预测下一个token。要理解其原理,需要先掌握神经网络的核心基础,但**不需要**学完整个深度学习课程。本计划给出**最小必要集合**,聚焦在"理解LLM"这个目标上,跳过与LLM关系不大的内容(如CNN、GAN、强化学习等)。

学习顺序建议:数学基础 → 神经网络基础 → 序列建模 → Transformer → LLM训练。

---

## 一、数学基础(最小集)

只需要够用,不需要系统重学。

### 1.1 线性代数
- **向量、矩阵、张量**:LLM中所有数据都是张量
- **矩阵乘法**:理解 `Y = XW + b` 这一行公式即可
- **维度变换**:能看懂 `[batch, seq_len, d_model]` 这种shape
- **点积(dot product)**:Attention的核心运算

### 1.2 微积分
- **导数、偏导数**:概念即可
- **梯度(gradient)**:知道是"参数更新方向"
- **链式法则**:反向传播的数学基础(知道原理即可)

### 1.3 概率与统计
- **概率分布**:LLM输出是token的概率分布
- **Softmax**:把任意分数转成概率分布的函数(必须掌握)
- **交叉熵(Cross Entropy)**:LLM的损失函数
- **采样(sampling)**:temperature、top-k、top-p 都基于此

---

## 二、神经网络基础(核心)

### 2.1 前馈神经网络(MLP / FFN)
- 神经元 = 加权求和 + 激活函数
- 全连接层(Linear layer):`y = Wx + b`
- Transformer中的FFN就是2层MLP

### 2.2 激活函数
- **ReLU**(经典)、**GELU**(Transformer常用)、**SiLU/Swish**(现代LLM如LLaMA用)
- 知道"为什么需要非线性"即可

### 2.3 训练机制
- **损失函数(Loss)**:衡量预测与真实的差距
- **反向传播(Backpropagation)**:计算梯度的算法(理解原理,不必手推)
- **梯度下降(SGD / Adam / AdamW)**:用梯度更新参数
- **学习率(Learning Rate)**

### 2.4 训练中的关键技巧
- **Embedding**:把离散token映射成稠密向量(LLM的输入层)
- **Layer Normalization**:稳定训练(Transformer必备)
- **Residual Connection(残差连接)**:`x + f(x)`,让深层网络可训练
- **Dropout**:正则化(了解即可,现代LLM用得少)

---

## 三、序列建模基础

### 3.1 为什么需要序列模型
- 语言是序列,token之间有顺序依赖
- 简单了解 RNN/LSTM 的"思想"即可(知道它们处理序列、但有长程依赖问题),**不必深入**

### 3.2 Tokenization(分词)
- **BPE(Byte Pair Encoding)**:GPT系列使用
- **SentencePiece**:LLaMA使用
- 理解"为什么不是字符也不是单词"

### 3.3 自回归生成
- **Next Token Prediction**:LLM的本质任务
- **因果掩码(Causal Mask)**:训练时防止看到未来token

---

## 四、Transformer 架构(LLM的骨架,最重要!)

这是LLM理解的**核心**,需要重点学习。

### 4.1 Attention 机制
- **Q、K、V(Query、Key、Value)**:必须深入理解
- **Scaled Dot-Product Attention**:`softmax(QK^T / √d) · V`
- **Multi-Head Attention(多头注意力)**

### 4.2 Transformer Block 组成
1. Multi-Head Self-Attention
2. Add & LayerNorm(残差 + 归一化)
3. Feed-Forward Network(FFN)
4. Add & LayerNorm

### 4.3 位置编码(Positional Encoding)
- **绝对位置编码**(原始Transformer)
- **RoPE(旋转位置编码)**:现代LLM(LLaMA、Qwen)使用
- 知道"为什么需要位置编码"即可

### 4.4 Decoder-only架构
- LLM主流架构(GPT、LLaMA、Qwen 都是)
- 与Encoder-Decoder的区别(简单对比即可)

---

## 五、LLM特有内容

### 5.1 训练范式
- **预训练(Pre-training)**:海量文本上的Next Token Prediction
- **SFT(Supervised Fine-Tuning)**:指令微调
- **RLHF / DPO**:对齐人类偏好(知道概念即可)

### 5.2 推理(Inference)
- **KV Cache**:为什么生成快(必须理解)
- **解码策略**:Greedy、Beam Search、Sampling、Temperature、Top-k、Top-p

### 5.3 缩放法则(Scaling Laws)
- 模型大小、数据量、计算量之间的关系(了解即可)

---

## 六、可以**跳过**的内容(节省时间)

为了最小集,以下内容可以**先不学**,等需要时再回头补:

- **CNN(卷积神经网络)**:与LLM无关
- **RNN/LSTM 的细节实现**:思想了解即可
- **GAN / VAE / Diffusion**:生成模型,但与LLM路线不同
- **强化学习基础**:除非要深入RLHF,否则跳过
- **图神经网络(GNN)**:与LLM无关
- **复杂优化理论**:用得上Adam就够了

---

## 七、推荐学习资源(精选,不贪多)

按优先级排序,**前3个就够入门**:

1. **3Blue1Brown 神经网络系列**(YouTube/B站):可视化讲解神经网络和Transformer,最直观
2. **Andrej Karpathy 的 "Let's build GPT" 视频** + **"Neural Networks: Zero to Hero" 系列**:从零手搓GPT,理解原理的最佳路径
3. **《动手学深度学习》(d2l.ai)**:中文教材,有代码,挑选神经网络基础 + Transformer 章节即可
4. **The Illustrated Transformer**(Jay Alammar 博客):图解Transformer
5. **Attention Is All You Need 论文**:最后再读原论文

---

## 八、验证学习成果(Checkpoint)

学完后,你应该能回答这些问题:

- [ ] LLM是怎么"预测"下一个词的?(自回归 + softmax概率分布)
- [ ] 什么是Attention?为什么需要它?(Q/K/V、长程依赖)
- [ ] 一个Transformer Block里有什么?(Attention + FFN + 残差 + LayerNorm)
- [ ] LLM是怎么训练的?Loss是什么?(交叉熵损失,Next Token Prediction)
- [ ] KV Cache 为什么能加速推理?(避免重复计算K、V)
- [ ] Tokenization 在做什么?为什么不用字符或单词?(BPE平衡词表大小与语义)
- [ ] 能徒手画出 Transformer 架构图吗?

**实操检验**:能跟着 Karpathy 的视频从零写一个mini-GPT,就基本掌握了LLM原理。

---

## 学习路径总结(一句话)

> **数学够用 → MLP/激活/反向传播 → Embedding/Tokenization → Attention机制 → Transformer架构 → 自回归训练与推理**

预计投入:**全职 2-4 周**,业余 **1-2 个月**,即可达到"理解LLM原理"的水平。
