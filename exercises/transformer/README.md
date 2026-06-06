# Transformer 练习

从 Attention 本体到完整 Mini GPT 的端到端实现。
**前置**:看完 `neural-network/` 全部 + `pytorch-llm-basic/` 全部。

---

## 学习路径(按顺序)

| # | 脚本 | 学什么 | 关键产出 |
|---|------|------|---------|
| 01 | `01_attention_scratch.py` | 手写 scaled dot-product attention | 注意力矩阵热图 |
| 02 | `02_causal_self_attention.py` | 加因果掩码 + 封装 nn.Module | masked vs unmasked 对比 |
| 03 | `03_multihead_attention.py` | 多头并行 + reshape/transpose | 4 个头的注意力分布 |
| 04 | `04_transformer_block.py` | 完整 Block: MHA + FFN + Residual + LN | 6 层激活分布稳定性 |
| 05 | `05_mini_gpt.py` | 完整 GPT 架构(无训练) | 213K 参数模型, 形状验证 |
| 06 | `06_train_and_generate.py` | 真实训练 + 自回归生成 | **从 prompt 续写正确句子** |

---

## 运行方式

```bash
cd exercises

# 默认: 不画图 (只打印日志, 跑得快)
uv run python transformer/01_attention_scratch.py

# 加 --plot: 保存 PNG 到 transformer/plots/
uv run python transformer/01_attention_scratch.py --plot
```

依次跑全部:

```bash
for s in transformer/0*.py; do
    uv run python "$s" --plot
done
```

06 还支持自定义训练步数:

```bash
uv run python transformer/06_train_and_generate.py --plot --epochs 5000
```

---

## 关键学习产出

### 你会理解:
- ✓ **Attention 的本质**:Q/K/V 是同一份输入的 3 个视角,通过 `softmax(QK^T/√d) · V` 实现"加权汇总"
- ✓ **为什么用因果掩码**:LLM 训练时禁止"作弊看未来"
- ✓ **多头注意力的工程实现**:reshape + transpose 实现并行
- ✓ **Pre-Norm 设计**:`x + Attn(LN(x))` 的现代写法
- ✓ **完整 GPT 数据流**:Token Embedding → N Blocks → LM Head
- ✓ **训练范式**:cross-entropy + Adam + 自回归采样

### 你能做到:
- 看懂 [nanoGPT](https://github.com/karpathy/nanoGPT) 全部代码
- 看懂 [LLaMA 源码](https://github.com/meta-llama/llama) 主体结构
- 从零写一个新的 Transformer 变种(改激活、改 norm、改 attention 形式)
- 读懂"Attention Is All You Need"论文

---

## 模型对比

| 模型 | d_model | n_layers | n_heads | 参数量 | 训练数据 |
|------|---------|----------|---------|--------|---------|
| 本 mini GPT | 64 | 3-4 | 4 | ~150K | 1 段英文文本 |
| GPT-2 small | 768 | 12 | 12 | 124M | WebText |
| GPT-3 | 12288 | 96 | 96 | 175B | 大量互联网文本 |
| LLaMA 3 70B | 8192 | 80 | 64 | 70B | 公开 + 私有数据 |

**架构本质一样,只是规模不同。**

---

## 验证示例(06 训练后的实际输出)

```
=== 用训好的模型生成 ===

Prompt: 'the q'
续写: 'the quick brown fox jumps over the lazy dog. '

Prompt: 'pack '
续写: 'pack my box with five dozen liquor jugs. pack'

Prompt: 'how v'
续写: 'how vexingly quick daft zebras jump. the five'
```

→ Mini GPT 成功学会了训练文本,并能在给定 prompt 后续写正确内容。
**这就是 LLM 的工作机制 — 只是规模更大,数据更多。**

---

## 下一步学什么?

学完这 6 个脚本你已经掌握 LLM 的核心机制。如果想继续深入:

| 主题 | 推荐资源 |
|------|---------|
| 看真实 GPT 实现 | [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) |
| LLaMA 优化点 | RMSNorm(已在 `pytorch-llm-basic/15`)、RoPE(已在 `17`)、SwiGLU |
| 高效推理 | KV Cache、Flash Attention |
| 微调 | LoRA、QLoRA、PEFT |
| 对齐 | SFT、RLHF、DPO |
| 上下文扩展 | RoPE NTK scaling、YaRN |
