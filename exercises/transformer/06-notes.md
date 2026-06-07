# 05 vs 06 的 MiniGPT 类对比

**结论:不一样**。功能上 `__init__` 和 `forward` 等价,但 06 多了一个方法,05 多了一句断言和注释。

## 核心差异对比

| 差异点 | 05_mini_gpt.py | 06_train_and_generate.py |
|---|---|---|
| 类 docstring | ✅ 有(说明这是完整 GPT) | ❌ 无 |
| `forward` 里的 `assert T <= max_seq_len` | ✅ 有(L159) | ❌ 无 |
| `pos_emb` 写法 | 分两步:先 `pos_ids = arange(...)` 再查表 | 一行内联:`self.pos_emb(torch.arange(...))` |
| `forward` 返回 | `logits = self.lm_head(x); return logits` | `return self.lm_head(x)` |
| **`generate` 方法** | ❌ **没有** | ✅ **有**(L121-139, `@torch.no_grad()`, 带 temperature + multinomial 采样 + 滑窗) |
| 行内注释 | 多(讲解性) | 几乎没有 |

## 一句话总结

**05 是"架构展示版"**(裸 forward + 主函数里手写贪心生成),**06 是"实用训练版"**(把生成逻辑封装成 `generate` 方法,支持 temperature 采样和上下文滑窗,训练循环直接调用)。

剥掉 `generate` 方法、docstring、assert 和注释后,两者 `__init__` + `forward` 的计算逻辑完全一致。

## `generate` 方法关键点(只在 06 里)

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0):
    self.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -self.max_seq_len:]        # 滑窗:超长就截断
        logits = self(idx_cond)
        logits = logits[:, -1, :] / temperature       # 只看最后位置 + 温度
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # 概率采样(非贪心)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
```

对比 05 主函数里的"裸生成":
- 05 用 `argmax`(贪心,确定性)
- 06 用 `multinomial`(按概率采样,有多样性)+ temperature 控制随机程度
- 06 有滑窗 `idx[:, -max_seq_len:]`,05 没有(会撞 `assert`)
