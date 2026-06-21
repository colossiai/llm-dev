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
| 07 | `07_inspect_and_use_checkpoint.py` | 读取 checkpoint:查看内容 + 加载续写 | 不重训直接复用已存模型 |

---

## 图解详细流程(逐脚本拆解)

每个脚本配一份 `NN_process.md` 图解文档:用 ASCII 数据流图 + shape 变化 + 表格对比 + 一句话总结,把对应脚本从输入到输出的每一步讲穿。建议**先读脚本注释,再对照图解**。

| # | 图解文档 | 核心看点 |
|---|---------|---------|
| 01 | [`01_process.md`](01_process.md) | 查表得 X → Q/K/V 投影 → 打分+softmax → 加权 V,四步看 shape |
| 02 | [`02_process.md`](02_process.md) | 因果掩码怎么造(`tril`)、怎么用(softmax 前 `masked_fill` 填 `-inf`)、为什么用 -inf 不用 0 |
| 03 | [`03_process.md`](03_process.md) | 拆头/合头的 `view+transpose` 重排;多头靠"塞进 batch 维"并行、头数不增参数 |
| 04 | [`04_process.md`](04_process.md) | FFN / 残差 / LayerNorm 三零件;Pre-Norm vs Post-Norm;GPT-3 175B 参数怎么算 |
| 05 | [`05_process.md`](05_process.md) | 完整 GPT 五段数据流;每个位置都出预测;自回归生成机制 |
| 06 | [`06_process.md`](06_process.md) | 字符级分词、错位标签(右移一位)、训练四步循环、温度采样生成 |

---

## 输入设计(每个脚本如何造输入张量)

不同脚本根据**教学重点**选择不同的输入方式 — 是否要把 `tokens` 标签和输入张量真正"关联"起来。

| 脚本 | 输入来源 | 教学重点 | 验证 |
|------|---------|---------|------|
| **01** | tokens → numpy embedding lookup | Attention 公式本身 | "The" 和 "the" 拿同向量 |
| **02** | tokens → torch embedding lookup | 因果掩码作用 | 同上 + 掩码切断未来 |
| **03** | tokens → torch embedding lookup | 多头并行 | 不同头学不同模式 |
| **04** | **纯随机张量(故意的)** | 形状不变 + 激活稳定 | 6 层后 std 仍 ~1 |
| **05** | 真实 token id + `nn.Embedding` | 完整 GPT 架构 | 213K 参数, shape 全对 |
| **06** | char-level token + 真实训练 | 端到端训练 + 生成 | 从 prompt 续写训练文本 |

**为什么 01/02/03 用 embedding lookup,04 用随机张量?**

- 01/02/03 演示**注意力模式**:相同 token 应该产生相同的注意力行为 — 必须用 lookup 保证一致
- 04 演示**形状/数值性质**:任何输入都不应改变形状, 激活分布应稳定 — 随机张量足够
- 05 是**真正的 LLM 架构**:用 `nn.Embedding(vocab_size, d_model)` 学习 token 表示
- 06 是**真训练**:char-level 分词 + cross-entropy + Adam 完整训练

→ **设计意图**:让读者看出"什么时候关心语义,什么时候只关心数值"。

---

## 运行方式

```bash
cd exercises

# 默认: 不画图 (只打印日志, 跑得快)
uv run python transformer/01_attention_scratch.py

# 显示图 (GUI 窗口)
uv run python transformer/01_attention_scratch.py --plot

# 保存到 transformer/plots/
uv run python transformer/01_attention_scratch.py --save

# 显示 + 保存
uv run python transformer/01_attention_scratch.py --plot --save
```

依次跑全部并保存:

```bash
for s in transformer/0*.py; do
    uv run python "$s" --save
done
```

06 还支持自定义训练步数,以及把训好的模型存到磁盘:

```bash
# 自定义训练步数
uv run python transformer/06_train_and_generate.py --save --epochs 5000

# 训练后保存模型到 transformer/checkpoints/06_minigpt.pt
uv run python transformer/06_train_and_generate.py --save_model
```

`--save_model` 存的是一个**自包含 checkpoint**(一个 dict):
`model_state`(权重)+ `config`(模型骨架超参)+ `vocab`(字符↔id 映射)+ `final_loss`。
单存权重不够 — 重建模型需要 config,做 encode/decode 需要 vocab。

07 读取这个 checkpoint:先打印内容(keys / config / 权重形状 / 参数量),
再三步把它变回能跑的模型(`MiniGPT(**config)` → `load_state_dict` → `eval`),
最后用 `vocab` 续写 —— **不用重新训练**:

```bash
# 先训练并保存, 再读取复用
uv run python transformer/06_train_and_generate.py --save_model
uv run python transformer/07_inspect_and_use_checkpoint.py

# 也可指定别的 checkpoint 路径
uv run python transformer/07_inspect_and_use_checkpoint.py --ckpt path/to/xxx.pt
```

> 坑提示:PyTorch 2.6+ 的 `torch.load` 默认 `weights_only=True`,只收纯张量。
> 我们的 checkpoint 含 config/vocab 这类 dict,需显式 `weights_only=False`
> (仅对自己生成的可信文件这么做)。

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

学完这 7 个脚本你已经掌握 LLM 的核心机制。如果想继续深入:

| 主题 | 推荐资源 |
|------|---------|
| 看真实 GPT 实现 | [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT) |
| LLaMA 优化点 | RMSNorm(已在 `pytorch-llm-basic/15`)、RoPE(已在 `17`)、SwiGLU |
| 高效推理 | KV Cache、Flash Attention |
| 微调 | LoRA、QLoRA、PEFT |
| 对齐 | SFT、RLHF、DPO |
| 上下文扩展 | RoPE NTK scaling、YaRN |
