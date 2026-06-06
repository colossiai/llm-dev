# LLM 学习练习

按学习目标"理解 LLM 原理"组织的渐进式练习。所有脚本基于 PyTorch + numpy + matplotlib,可用 `uv run` 直接跑。

---

## 目录分工(从基础到 LLM)

```
exercises/
├── pytorch-tensors/         张量基础 (shape, reshape, broadcasting, einsum...)
├── gradient-loss/           梯度下降直观:loss 是山, 梯度是下山方向
├── manual/                  手写线性回归(没有框架)
├── neural-network/          ★ 神经网络基础 (Perceptron, 激活函数, MLP, Backprop)
├── pytorch-llm-basic/       LLM 组件零件库 (softmax, embedding, mask, layernorm, RoPE)
├── transformer/             ★ Attention + Transformer Block + Mini GPT
├── prerequisite/, math1/, math2/    更早的数学预备
```

---

## 推荐学习顺序

```
   pytorch-tensors  →  gradient-loss  →  manual (线性回归)
            ↓
   neural-network/                       ← 神经网络基础到 MLP
   (01 Perceptron → 02 激活 → 03 非线性 → 04 手写 backprop
    → 05 autograd → 06 完整 MLP)
            ↓
   pytorch-llm-basic/                    ← LLM 单个零件
   (09 softmax → 10 cross-entropy → 11 embedding → 12 mask
    → 13 nn.Module → 14 autograd → 15 LayerNorm → 16 sampling → 17 RoPE)
            ↓
   transformer/                          ← Attention + 完整 GPT
   (01 Attention → 02 Causal → 03 Multi-Head → 04 Block
    → 05 Mini GPT → 06 训练 + 生成)
```

---

## 各目录详细说明

### `pytorch-tensors/`(张量基础)
- 01 shape / 02 reshape / 03 transpose / 04 batch matmul
- 05 broadcasting / 06 indexing / 07 squeeze / 08 contiguous
- 目的:写代码前的"语法准备",每个操作都要熟练

### `gradient-loss/`(梯度下降直觉)
- 01 loss 像一座山
- 02 梯度 = 上坡方向
- 03 最小化 loss = 下山
- 目的:用 3D 曲面图直观看到 loss 是什么

### `manual/`(手写线性回归)
- 用纯 Python 一行 PyTorch 都不用,从零写梯度下降
- 目的:理解"框架替你做了什么"

### `neural-network/`(神经网络基础)★ 已补全
| # | 主题 | 学什么 |
|---|------|------|
| 01 | Perceptron | 单个神经元 = 一条直线分两类 |
| 02 | 激活函数 | 6 种激活(ReLU/Sigmoid/GELU/SiLU...)对比 |
| 03 | 为什么需要非线性 | XOR 实验:线性 vs ReLU |
| 04 | 手写反向传播 | 纯 numpy 实现 forward + backward |
| 05 | PyTorch Autograd | 用 autograd 验证 04 的推导 |
| 06 | 完整 MLP | nn.Module + Adam + make_moons |

### `pytorch-llm-basic/`(LLM 组件库)
| # | 主题 | 类型 |
|---|------|------|
| 09 | softmax_logits | 把分数→概率 |
| 10 | cross_entropy | LLM 的损失函数 |
| 11 | embedding | 查表 = one-hot @ W |
| 12 | mask_tril | 因果掩码(Attention 配件) |
| 13 | nn_module_linear | nn.Module / nn.Linear |
| 14 | autograd | PyTorch 自动求导 |
| 15 | layernorm | LayerNorm + RMSNorm |
| 16 | sampling | top-k / top-p 采样 |
| 17 | rope | 旋转位置编码 |

**注意**:这是"LLM 零件库",每个脚本独立讲一个组件,**不串成完整模型**。
要看组件如何组装成 LLM,看下面的 `transformer/`。

### `transformer/`(Attention + 完整 GPT)★ 新补
| # | 主题 | 学什么 |
|---|------|------|
| 01 | Attention 本体 | 手写 Q/K/V + softmax(QK^T/√d)V |
| 02 | 因果自注意力 | 加掩码, 封装成 nn.Module |
| 03 | 多头注意力 | 把 d_model 拆成 N 个头并行 |
| 04 | Transformer Block | MHA + FFN + Residual + LayerNorm |
| 05 | Mini GPT | 完整 GPT 架构(无训练) |
| 06 | 训练 + 生成 | 在小文本上训练, 生成续写 |

---

## 关于 `pytorch-llm-basic/` 的常见疑问

**Q: `pytorch-llm-basic` 包含深度学习例子吗?**

答:**是,但是"LLM 组件向"的深度学习,不是通用入门**。

- 涵盖了深度学习核心机制(autograd、loss、归一化)
- 但所有例子都是**LLM 相关组件**(embedding、mask、RoPE、采样)
- **没有** CNN/RNN、图像分类、完整训练循环、完整 Attention 实现

可以理解为"LLM 的零件铺",**真正把零件组装成 LLM 是在 `transformer/` 目录**。

---

## 运行方式

整个 `exercises/` 共享一个 uv 项目(`pyproject.toml`)。

```bash
cd exercises

# 默认: 不画图 (只跑日志, 最快)
uv run python neural-network/01_perceptron.py

# 显示图 (调出 GUI 窗口)
uv run python neural-network/01_perceptron.py --plot

# 保存图到 plots/ 子目录
uv run python neural-network/01_perceptron.py --save

# 两个都要 (显示 + 保存)
uv run python neural-network/01_perceptron.py --plot --save
```

依次跑一个目录的全部脚本(只保存,不弹窗):

```bash
for s in neural-network/0*.py; do
    uv run python "$s" --save
done
```

---

## 学习时间预估

| 目录 | 投入(全职) | 投入(业余) |
|------|----------|----------|
| 数学预备 + tensors | 1-2 天 | 4-7 天 |
| neural-network | 2-3 天 | 1-2 周 |
| pytorch-llm-basic | 2-3 天 | 1-2 周 |
| transformer | 3-5 天 | 2-3 周 |
| **总计** | **8-13 天** | **5-8 周** |

学完后应该能:
- ✅ 看懂 LLaMA / nanoGPT 源码
- ✅ 从零写一个 mini GPT 并训练
- ✅ 看懂"Attention Is All You Need"论文
