# 15 笔记

## RMSNorm 里的 RMS 指什么?

### RMS = **Root Mean Square**(均方根)

这是一个**经典的数学/物理概念**(电气工程里常用,比如交流电"有效电压"就是 RMS 电压),不是 ML 专有的。

---

### 字面拆解:Root - Mean - Square

读名字时是 R → M → S, 但**计算时反着来**:

```
RMS(x) = √( mean( x² ) )
         ↑       ↑    ↑
       Root   Mean Square
       (开方)(平均)(平方)
```

→ 先**平方**, 再**取平均**, 最后**开方**。

代码里就是这一行:

```python
rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
#     └──────┘└────────────────────────┘                  └──┘
#      Square          Mean                                Root
```

---

### 一个数值例子

```python
x = [3, 4]

x²        = [9, 16]
mean(x²)  = 12.5
sqrt(...) = 3.535...   ← 这就是 RMS
```

注意:

- 算术平均 = (3+4)/2 = 3.5
- RMS = 3.535

**RMS 几乎等于"绝对值的某种平均", 但对大值的惩罚更重**(因为先平方了)。

---

### 关键对比:RMS vs 标准差(std)

这两个长得像但有本质区别:

| | 公式 | 直觉 |
|---|------|------|
| **RMS** | `√( mean(x²) )` | 数值的"整体大小" |
| **std** | `√( mean((x - μ)²) )` | 数值与**平均值**的偏差 |

**唯一区别**:std 要先减均值, RMS 不减。

举例:`x = [10, 11, 12]`

- mean = 11
- std ≈ 0.82(数值之间的波动很小)
- RMS ≈ 11.0(数值整体很大)

→ std 关心"离散程度", RMS 关心"绝对大小"。

---

### 所以 LayerNorm vs RMSNorm

```
LayerNorm:  y = (x - μ) / σ            ← 减均值, 再除以标准差
                ↑──┘    ↑
              中心化   缩放

RMSNorm:    y = x / RMS(x)             ← 不中心化, 只缩放
                  └──┬──┘
                    缩放
```

#### LLaMA / Mistral / Qwen 为什么选 RMSNorm?

| 维度 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 计算量 | 2 次遍历(算 μ, 再算 σ) | 1 次遍历(直接算 RMS) |
| 参数量 | γ 和 β 都要 | 只要 γ, **省一半** |
| 数学操作 | 减、平方、平均、开方、除 | 平方、平均、开方、除 |
| 实测效果 | 标准 | **几乎一样好** |

→ **少做一步"减均值", 但效果不掉, 所以现代 LLM 都改用 RMSNorm**。
作者(Zhang & Sennrich, 2019)在论文里发现:LayerNorm 真正起作用的是"缩放"那一步, "减均值"几乎可以省掉。

---

### 在脚本里验证一下

可以加几行测试代码:

```python
x = torch.tensor([3.0, 4.0])
# RMS 计算
rms_val = (x.pow(2).mean()).sqrt()
print(f"RMS = {rms_val.item():.4f}")   # 3.5355

# std 对比 (会减均值)
std_val = x.std(unbiased=False)
print(f"std = {std_val.item():.4f}")   # 0.5

# 看出区别: RMS 受"绝对大小"影响, std 只看波动
```

---

### 一句话总结

> **RMS = Root Mean Square(均方根) = √( mean( x² ) )**。
> 计算时是"平方 → 平均 → 开方", 读名字时反着读 R-M-S。
> 它和标准差很像, 但**不减均值**, 只衡量"数值整体的大小"。
> LLaMA 用 RMSNorm 替代 LayerNorm, 是因为发现"减均值"那一步可以省掉, 速度更快、效果不掉。
