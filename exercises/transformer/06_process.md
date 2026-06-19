# 06_train_and_generate.py 详细流程图解

> 配套脚本:`exercises/transformer/06_train_and_generate.py`
> 系列集大成:给 05 的模型**喂真实文本、跑训练循环、再让它续写**。重点是三件新东西:**字符级分词、错位标签、训练循环(forward→loss→backward→step)**。

## 全景:训练 + 生成两大阶段

```
  ┌─────────────────── 训练阶段 ───────────────────┐
  text 字符串                                        │
    │ ① 字符级分词 (char→id)                          │
    ▼                                                │
  data (token id 一维张量)                            │
    │ ② get_batch: 随机切窗 + 错位标签                 │
    ▼                                                │
  x (B,T)  ──模型 forward──► logits (B,T,V)           │
  y (B,T)  ───────────────────┐                      │
    │ ③ cross_entropy(logits, y)                      │
    ▼                                                │
  loss ──backward──► 梯度 ──optimizer.step──► 更新权重 │
    └────────────── 循环 3000 步 ◄───────────────────┘
  ┌─────────────────── 生成阶段 ───────────────────┐
  prompt "the q" ──encode──► ids                     │
    │ ④ generate: 自回归采样 40 个字符                 │
    ▼                                                │
  续写文本 ──decode──► "the quick brown fox..."        │
```

| 阶段 | 关键函数 | 干什么 |
|------|---------|--------|
| 分词 | `encode/decode` | 字符 ↔ id 互转 |
| 取数据 | `get_batch` | 随机切窗 + 错位 |
| 训练 | `cross_entropy` + `backward` + `step` | 调权重 |
| 生成 | `generate` | 自回归采样续写 |

---

## ① 字符级分词 (148–177 行)

最简单的分词:**词表 = 文本里出现过的所有不重复字符**(约 30 个,含字母、空格、句号)。

```
  text = "the quick brown fox..."
    │ chars = sorted(set(text))     → [' ', '.', 'a', 'b', ..., 'z']
    │ char_to_id = {' ':0, '.':1, 'a':2, ...}
    ▼
  encode("the") = [19, 7, 4]        ← 字符 → id
  decode([19,7,4]) = "the"          ← id → 字符
    │
    ▼
  data = encode(整段文本)  → 一维 token id 张量
```

> 真实 LLM 用 BPE/子词分词(词表 5 万+),这里用字符级纯粹为了**简单到能看穿**。代价是序列更长、语义粒度粗,但对"背一段话"的过拟合实验足够。

---

## ② get_batch:随机切窗 + 错位标签 (195–206 行)

这是 LLM 训练范式的核心——**target 就是 input 右移一位**:

```
  从 data 里随机选起点 i, 切长度 T 的窗口:

  原始:  data[i]  data[i+1] data[i+2] data[i+3] data[i+4]
           t  h  e ' '  q  ...
  ┌──────────────────────────────────────────┐
  │ x = data[i   : i+T  ]   [t, h, e, ' ']     │  输入
  │ y = data[i+1 : i+T+1]   [h, e, ' ', q]     │  目标(错位1)
  └──────────────────────────────────────────┘
        位置0看到 t      → 该预测 h
        位置1看到 t,h    → 该预测 e
        位置2看到 t,h,e  → 该预测 ' '
        ...每个位置都在"预测下一个字符"
```

```
  一次取 batch_size=16 段, 堆成:
    x (16, 32)   y (16, 32)
```

**为什么错位 1 个就是"预测下一个词"?** 因为对齐后,`x` 的每个位置 t 对应的正确答案,正好是 `y` 同位置(= 原文 t+1)。配合因果掩码(位置 t 只能看 0..t),这就是不作弊的"预测下一个词"。

---

## ③ 训练循环:四步一拍 (208–230 行)

每个 step 都是固定的四步:

```
  ┌─► ① x,y = get_batch()           取一批数据
  │   ② logits = model(x)           前向: (B,T) → (B,T,V)
  │   ③ loss = cross_entropy(...)   算"预测得多差"
  │   ④ zero_grad → backward → step  反向传播 + 更新权重
  └──────────── 循环 3000 次 ───────────┘
```

**③ cross_entropy 在算什么?**

```
  logits.view(-1, V)  →  (B*T, V)    把所有位置摊平
  y.view(-1)          →  (B*T,)      对应的正确答案

  loss = 平均 over 所有位置 [ -log( 模型给"正确下一个字符"的概率 ) ]
         └ 模型越确信正确答案, loss 越小 ┘
```

**loss 的参照系**(232–233 行):

| loss 值 | 含义 |
|---------|------|
| `log(vocab_size)` ≈ 3.4 | 完全瞎猜(均匀分布)的基线 |
| 显著 < 3.4 | 模型学到了模式 |
| << 1 | 几乎背下来了(本实验目标) |

**④ 三件套的固定顺序:**

```
  optimizer.zero_grad()  清掉上一步的旧梯度(否则会累加)
  loss.backward()        自动微分, 算出每个参数的梯度
  optimizer.step()       AdamW 按梯度更新所有权重
```

---

## ④ 生成:自回归采样续写 (121–139, 235–245 行)

训练完用 `generate` 续写。比 05 的 `argmax` 多了**滑窗 + 温度采样**:

```
  ids = encode("the q")
    │ 循环 40 次:
    │   idx_cond = ids[:, -max_seq_len:]   ← 滑窗: 太长只留最后 32 个
    │   logits = model(idx_cond)[:, -1, :] / temperature
    │   probs = softmax(logits)
    │   next = multinomial(probs)          ← 按概率随机采样(非贪心)
    │   ids = cat([ids, next])
    ▼
  decode(ids) → "the quick brown fox jumps over..."
```

| 机制 | 作用 |
|------|------|
| 滑窗 `[-max_seq_len:]` | 序列超长时只保留最近 32 个 token(模型上限) |
| `/ temperature` | 调"随机度":<1 更确定/保守,>1 更发散 |
| `multinomial` 采样 | 按概率掷骰子,而非永远选最高(避免重复死板) |

> 05 用 `argmax`(贪心)只为演示接口;06 训练后用**温度采样**,生成更自然。三个 prompt(`"the q"` / `"pack "` / `"how v"`)分别测模型能否续出三句不同的原文。

---

## ⑥ 可视化:loss 曲线 (247–269 行,需 `--draw`)

```
  loss
   3.4 ┤╲                    ← 随机基线 log(vocab) 虚线
       │ ╲___
       │     ╲___
   1.0 ┤         ╲_____
       │               ╲________  ← 滑动平均(红), 稳步下降
   0.0 ┼──────────────────────────► step
       0                      3000
```

loss 从 ~3.4(瞎猜)稳步降到 <1,就证明 **Transformer + Cross-Entropy + AdamW 真的能学到语言模式**。

---

## 一句话总结

06 = **把 05 的模型放进真实训练闭环**:字符级分词把文本变 token id → `get_batch` 随机切窗并把标签右移一位(=预测下一个词)→ 训练循环重复「forward 出 logits、cross_entropy 算损失、backward 求梯度、AdamW 更新权重」3000 步 → loss 从 `log(vocab)≈3.4` 降到 <1 → 最后用带滑窗和温度采样的 `generate` 自回归续写,验证小 GPT 真的把训练文本学会了。
