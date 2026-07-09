# LLM 原理夯实 · 开源项目实战学习计划

## Context(为什么做这件事)

你已经学完 LLM 原理,但 16 个核心概念(tensor/autograd/attention/tiny transformer 等)**不牢固**。
根本原因是这些概念只"看过"没"手写调过"——梯度怎么流、shape 怎么变、attention 几何长什么样,
这些只有亲手训练 + 亲手制造 bug 再修才能内化。

目标:通过 **2~3 个经典开源项目**,把 16 个概念全部覆盖并做到"能训、能改、能 debug"。

约束(已与你确认):
- **硬件**:纯 CPU / Mac(可用 MPS)。主线放在 CPU 就能跑透的项目,不追求大规模 pretrain。
- **材料语言**:中英结合。英文打地基(概念最清晰),中文项目快速跑通全流程拿正反馈。

---

## 概念 → 项目 覆盖矩阵

| 你的 16 个概念 | 主练项目 | 说明 |
|---|---|---|
| 偏导 / chain rule / computational graph / autograd / gradient descent | **micrograd** | 手写反向传播引擎(~150 行),看透梯度流动 |
| Tensor/shape / transpose/reshape/broadcasting / batch matmul / softmax / cross entropy / normalization / PyTorch tensor ops | **makemore** | 从计数模型升到 MLP,每个 tensor 操作亲手调 |
| attention 几何 / positional encoding / residual / 手写 tiny transformer | **nanoGPT**(char 级 Shakespeare) | CPU/MPS 可训的最小 GPT |
| 全流程(pretrain→SFT→chat) + 中文实践 | **minimind** | 中文注释,快速跑通完整链路 |

---

## 推荐主线(四阶段,循序渐进)

### 阶段 0 · 环境(用 uv)
- `uv` 建虚拟环境,装 `torch`(Mac 走 CPU/MPS 版)、`numpy`、`matplotlib`。
- 跟随 Karpathy **"Neural Networks: Zero to Hero"** 视频系列作为主线讲解(英文打地基)。

### 阶段 1 · autograd 内核 — karpathy/micrograd
仓库:`https://github.com/karpathy/micrograd`
- **目标概念**:偏导、chain rule、computational graph、autograd、gradient descent。
- **做法**:不要只读——把 `Value` 类的 `backward()` 遮住自己重写一遍,画出计算图手算梯度对拍。
- **产出**:能口述"一次 forward + backward 中每个节点的 grad 从哪来"。

### 阶段 2 · tensor 与训练循环 — karpathy/makemore
仓库:`https://github.com/karpathy/makemore`
- **目标概念**:shape/reshape/transpose/broadcasting、batch matmul、softmax、cross entropy、normalization、PyTorch ops。
- **做法**:按视频从 bigram → MLP → 引入 BatchNorm,逐步替换。每步用 `.shape` 打印验证维度直觉。
- **产出**:不看文档能写出 softmax + cross entropy 的前向,并解释为什么 softmax 要减 max。

### 阶段 3 · 手写 tiny transformer — karpathy/nanoGPT
仓库:`https://github.com/karpathy/nanoGPT`(或先读更易懂的 `minGPT`)
- **目标概念**:attention 几何、positional encoding、residual、手写 tiny transformer。
- **做法(Mac/CPU 版)**:用 char 级 Shakespeare 数据,把 `n_layer/n_head/n_embd/block_size` 调小(如 4/4/128/64),
  `device` 设 `mps` 或 `cpu`。目标是能训到 loss 下降并生成半通顺文本,不追求质量。
- **产出**:能画出单个 attention head 的 Q/K/V 维度流,解释 causal mask 与 residual 的作用。

### 阶段 4 · 跑通完整链路(中文)— jingyaogong/minimind
仓库:`https://github.com/jingyaogong/minimind`
- **目标**:体验 pretrain → SFT → (可选)DPO/chat 的完整工程链路,中文注释友好。
- **做法**:用它最小的配置在 CPU/Mac 上跑通一次全流程,重点是"链路"而非"规模"。
- **产出**:能说清一个能对话的小模型从数据到 chat 要经过哪几步。

### 系统性补充(贯穿全程)
- **rasbt/LLMs-from-scratch**(Sebastian Raschka 配套代码):章节顺序几乎等于你的清单,当"教科书"随时查漏。
- **The Annotated Transformer**(Harvard)/ **labml.ai annotated implementations**:逐行注释,attention 卡壳时对照。

---

## Debug Case 训练法(核心学法:注入 bug → 观察现象 → 修复)

在阶段 2/3 的可运行代码上,故意注入以下经典 bug,先预测现象再验证:

| Bug 注入 | 在哪注入 | 预期可观测现象 |
|---|---|---|
| 转置/广播维度错 | attention `Q @ K.transpose` 转错维 | shape mismatch,或 loss 完全不降 |
| 去掉 causal mask | 删掉下三角 mask | 训练 loss 好得反常,但生成乱码(偷看未来) |
| softmax 不减 max | 去掉数值稳定项 | 序列一长就 NaN |
| 漏 zero_grad | 注释 `optimizer.zero_grad()` | 梯度累加,loss 震荡/爆炸 |
| loss reduction 错 | `mean` 改 `sum` | 梯度尺度爆炸,配 lr 直接 NaN |
| positional encoding off-by-one | 位置索引错位 | 短序列 OK,长序列崩 |
| LayerNorm 归错 axis | 归一化维度写错 | 收敛变慢/不收敛 |
| detach 误用 | 在需要梯度处 `.detach()` | 该部分参数不更新,loss 卡住 |

**建议**:每个 bug 用 git 建一个分支,commit 记录"现象 + 定位过程 + 修复",形成自己的 debug 笔记。

---

## 验证(如何确认学到位)

1. **micrograd**:手写 `backward()` 与原实现对拍,梯度数值一致。
2. **makemore**:独立写出 softmax+CE 前向,loss 曲线随训练下降。
3. **nanoGPT**:char 级模型在 Mac 上训练 loss 明显下降,能生成半通顺文本;
   能对着代码讲清一个 attention head 的维度流与 mask 作用。
4. **debug 表**:每一行 bug 都能"先预测现象、注入后验证一致、并修复"。
5. **minimind**:CPU/Mac 上跑通一次 pretrain→SFT 最小链路,产出一个能简单对话的模型。

---

## 一句话总结

主线走 Karpathy 的 micrograd → makemore → nanoGPT(英文打地基),
minimind 做中文完整链路,Raschka 的书随时查漏;
真正内化靠"故意制造 bug 再修"的 debug 笔记。
