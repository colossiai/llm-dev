# 手写最小 LLM 必备（09–17）

> 前置: 你已经做完 `../pytorch-tensors/`（01–08：张量基本功）。
>
> 本目录每节都直接对应手写最小 LLM 里实际用到的 PyTorch 原语。

---

## 环境准备（使用 uv）

```bash
cd /Users/ericyeung/ai-space/claude-buildllm/exercises/pytorch-llm-basic

# 初始化 uv 项目（如果还没有）
uv init --no-readme

# 安装依赖（Intel Mac 用 CPU 版即可）
uv add torch matplotlib numpy

# 如果走 Zscaler 网络遇到 TLS 问题：
# uv --native-tls add torch matplotlib numpy
```

## 运行方式

```bash
# 单独运行 (默认: 只打印控制台输出)
uv run 09_softmax_logits.py

# 加 --plot 才会保存可视化图到 plots/
uv run 14_autograd.py --plot

# 或运行所有 (含可视化)
for f in [01]*.py; do echo "=== $f ==="; uv run "$f" --plot; done
```

每个脚本都接受 `--plot` 参数：
- 不加: 只控制台输出
- `--plot`: 同时保存 PNG 到 `./plots/`

---

## 学习顺序

| 文件 | 主题 | 在 LLM 里的角色 |
|---|---|---|
| `09_softmax_logits.py` | softmax / log_softmax / 温度 | 注意力概率、输出概率 |
| `10_cross_entropy.py` | F.cross_entropy / next-token loss | 训练目标 |
| `11_embedding.py` | nn.Embedding (= 查表 = one-hot @ W) | token / 位置嵌入 |
| `12_mask_tril.py` | tril + masked_fill | 因果注意力掩码 |
| `13_nn_module_linear.py` | nn.Module / Linear / Parameter | 所有层的"积木" |
| `14_autograd.py` | requires_grad / backward / optimizer | 训练循环 |
| `15_layernorm.py` | LayerNorm 与 RMSNorm | 每个 block 的归一化 |
| `16_sampling.py` | argmax / topk / multinomial / 温度 | model.generate() |
| `17_rope.py` | 旋转位置编码 | 现代 LLM 的位置编码方案 |

另外:
- `manual-softmax.py` — 纯 Python 手写 softmax（不依赖 torch），用来吃透公式

建议**按顺序**学完。学完之后, 可以直接打开 `minimal_edu/model.py` 通读全文。

---

## 每个文件的结构

```
1. 概念说明（注释）
2. 代码演示（print 出每一步）
3. 可视化（保存 PNG 到 plots/）
4. 小练习（带 assert 检查，TODO 留给你填）
```

碰到 `# TODO`，先自己想答案再看下面的参考。
所有 `assert` 通过说明你写对了。

---

## 学完之后你应该能回答

1. `F.cross_entropy` 接受什么 shape？输入应不应该先 softmax？
2. `nn.Embedding(V, d)` 在数学上等价于什么？
3. 因果掩码为什么要在 `softmax` 之**前**填 `-inf` 而不是之后乘 0？
4. 为什么 LayerNorm 沿"最后一维"归一化、而 BatchNorm 沿"batch 维"？
5. greedy / multinomial / top-k / top-p 各自的取舍是什么？
6. RoPE 和"加法式位置嵌入"最本质的区别是什么？

跟 LLM 相关的算子笔记（argmax / 采样等）见 `pytorch-notes.md`。
