# 05_mini_gpt.py 详细流程图解

> 配套脚本:`exercises/transformer/05_mini_gpt.py`
> 核心:把前面所有零件**组装成一个完整的 GPT**:从 token id 进、到下一个词的概率出,再演示自回归生成。

## 全景:一个完整 GPT 的数据流

```
   idx (B,T)  token id 整数,如 [[42, 7, 91, ...]]
        │
        │  ① Token Embedding  (查表 vocab→d)
        ▼
   tok (B,T,64)
        │  ② + Positional Embedding (位置 0..T-1 → d)  [broadcast]
        ▼
   x = tok + pos  (B,T,64)        ← 现在有"词义 + 位置"
        │
        │  ③ N=4 个 Transformer Block (04 整个, 堆叠)
        ▼   (B,T,64) → ... → (B,T,64)  形状不变
   x  (B,T,64)
        │  ④ 最终 LayerNorm (ln_f)
        ▼
        │  ⑤ LM Head: Linear(64 → vocab=100)
        ▼
   logits (B,T,100)               ← 每个位置: 下一个 token 的打分
        │  softmax
        ▼
   每个位置一个"下一个词"的概率分布
```

| 部件 | 形状变化 | 干什么 |
|------|---------|--------|
| ① Token Emb | `(B,T) → (B,T,d)` | id → 词义向量 |
| ② Pos Emb | `+ (T,d)` | 注入位置信息 |
| ③ Block × N | `(B,T,d) → (B,T,d)` | 主体加工(大部分参数) |
| ④ LayerNorm | `(B,T,d)` 不变 | 输出前最后校准 |
| ⑤ LM Head | `(B,T,d) → (B,T,vocab)` | 投影到词表打分 |

---

## ① + ②:两个 Embedding 相加 (133–169 行)

GPT 输入不是向量而是 **token id 整数**。两张查找表分别提供"词义"和"位置":

```
   idx = [[42, 7, 91, ...]]        token id (B,T)
          │
   tok_emb(idx)        位置 0,1,2,...,T-1
          │            pos_emb(pos_ids)
          ▼                  ▼
   tok (B,T,64)        pos (T,64)
          └────────┬─────────┘
                   │  x = tok + pos
                   ▼   (pos 沿 batch 维 broadcast)
              x (B,T,64)
```

**为什么要加位置?** 注意力本身对顺序"无感"(打乱 token 算出的注意力一样)。位置 embedding 给每个位置一个独有向量,模型才能区分"谁在前谁在后"。

| | Token Emb | Positional Emb |
|---|-----------|----------------|
| 表大小 | `vocab × d` | `max_seq_len × d` |
| 索引用 | token id | 位置 0..T-1 |
| 现代替代 | (不变) | LLaMA 改用 RoPE,省掉这张表 |

---

## ③:N 个 Transformer Block 堆叠 (171–173 行)

就是 04 那个 block,用 `for` 循环串 N 层。因为输入输出同形,串多少层都行:

```
  x ─[Block 0]─►[Block 1]─►[Block 2]─►[Block 3]─► x
     每层: x = x + attn(LN(x)); x = x + ffn(LN(x))
     (B,T,64) 一路不变, 但内容被层层加工
```

**大部分参数都在这里。** 本 mini 模型 d=64、4 层,真实 GPT-3 是 d=12288、96 层。

---

## ④ + ⑤:LayerNorm + LM Head (175–177 行)

```
  x (B,T,64)
    │ ln_f: 最后一次 LayerNorm
    ▼
    │ lm_head: Linear(64 → 100)   无 bias
    ▼
  logits (B,T,100)
```

`logits[b, t, :]` 是 100 个分数,表示"位置 t 之后,下一个 token 是词表里每个词的可能性"。过 softmax 就成概率(238–241 行验证和为 1)。

> **关键:每个位置都输出一个预测**,不只是最后一个。训练时这让一条长度 T 的序列同时产生 T 个"预测下一个词"的监督信号(配合 02 的因果掩码,位置 t 只看了 0..t,预测 t+1 不作弊)。

---

## 参数分布 (203–224 行)

```
   Mini GPT (~50K 参数) 构成
   ┌──────────────────────────────────────────┐
   │ Token+Pos Embedding   (vocab+seq)×d        │ 一部分
   │ 4 × Transformer Block (≈12d²/层)   主体    │ ████ 大头
   │ LayerNorm + LM Head   d×vocab              │ 一部分
   └──────────────────────────────────────────┘
```

> 小模型里 embedding 和 LM head 占比可观(因为 d 小、12d² 不大);模型一放大,Block 那项 `12d²·N` 平方+线性增长,迅速主导(GPT-3 里 Block 占 99.6%)。本模型约 50K,是 GPT-3 的约 350 万分之一。

---

## 自回归生成:GPT 怎么"写字" (243–260 行)

训练后的 GPT 靠**一次预测一个词、把它接回去再预测**滚动生成:

```
  generated = [42]
    │ 喂进模型 → 取最后位置 logits → argmax 选下一个词
    ▼
  [42, 7]
    │ 整段重新喂 → 取最后位置 → argmax
    ▼
  [42, 7, 91]
    │ ... 循环 20 次
    ▼
  [42, 7, 91, ...]   (本脚本随机权重, 输出无意义)
```

```
  step 0:  [42]                    ──► 预测 → 7
  step 1:  [42, 7]                 ──► 预测 → 91
  step 2:  [42, 7, 91]             ──► 预测 → ...
           └ 每步只取最后位置的预测, 拼回输入, 再来一遍 ┘
```

这里用 `argmax`(贪心,总选最高分);真实生成会用温度采样/top-k/top-p 增加多样性。本脚本没训练,纯演示**生成接口长什么样**——06 才会真正训练。

---

## 各代 GPT 规模对照 (注释 55–59 行)

| 模型 | d_model | n_layers | n_heads | 参数 |
|------|---------|----------|---------|------|
| 本 mini | 64 | 4 | 4 | ~50K |
| GPT-2 small | 768 | 12 | 12 | 124M |
| GPT-2 large | 1280 | 36 | 20 | 774M |
| GPT-3 | 12288 | 96 | 96 | 175B |

**架构完全一样,只是把 d/层数/头数往上调。** 这就是"scaling"的字面意思。

---

## 一句话总结

Mini GPT = **token嵌入 + 位置嵌入 → N 层 Transformer Block → LayerNorm + LM Head**:输入 token id `(B,T)`,经两张查找表变成带词义+位置的向量,过 N 个 block 层层加工(形状始终 `(B,T,d)`),最后投影到词表得 `(B,T,vocab)` 的下一词打分;生成时一次预测一个词、接回去自回归滚动。所有现代 LLM(GPT/LLaMA/Qwen)都是这个结构,区别只在规模。
