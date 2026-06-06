# PyTorch 学习笔记（LLM 篇）

本文件收录跟训练 / 推理直接相关的张量算子；通用的 shape / broadcasting / indexing 笔记见 `../pytorch-tensors/pytorch-notes.md`。

---

## `argmax()` 是什么

```python
logits = torch.randn(4, 1000)
top1 = logits.argmax(dim=-1)  # (4,)
```

`argmax` = **arg**ument of the **max**imum，返回 **"最大值所在的位置（索引）"**，而不是最大值本身。

### `max` vs `argmax`

```python
x = torch.tensor([0.3, 1.7, -0.5, 2.1, 0.8])

x.max()       # → tensor(2.1)   ← 最大值是多少
x.argmax()    # → tensor(3)     ← 最大值在第几个位置（索引 3）
```

记住：`argmax` 返回的是 **整数索引**（位置），不是浮点数（数值）。

### 在分类例子里

```python
logits = torch.randn(4, 1000)   # 4 个样本，每个有 1000 个类别的分数
top1 = logits.argmax(dim=-1)    # (4,)  每个样本的"最高分类别号"
```

意思是：

```
logits[0]  ──argmax──>  某个 0..999 的整数（样本 0 预测的类别）
logits[1]  ──argmax──>  某个 0..999 的整数
logits[2]  ──argmax──>  某个 0..999 的整数
logits[3]  ──argmax──>  某个 0..999 的整数
```

`top1.shape = (4,)`：4 个样本各自的 **top-1 预测类别号**。

> 这是分类模型 inference 的最后一步：模型输出 1000 个类别的 logits（原始分数），用 `argmax` 取分数最高的那个类别作为最终预测。ImageNet 就是 1000 类。

### `dim` 参数：沿哪个维度取最大

`dim` 指定 **"在哪个轴上找最大值"**。可以理解为 **"消掉哪个轴"**：

```python
logits.shape          # (4, 1000)

logits.argmax()                 # → scalar  整个张量里最大值的"展平索引"
logits.argmax(dim=0).shape      # → (1000,) 沿样本维找最大 → 每个类别"哪个样本得分最高"
logits.argmax(dim=1).shape      # → (4,)    沿类别维找最大 → 每个样本"得分最高的类别"
logits.argmax(dim=-1).shape     # → (4,)    -1 即最后一维，跟 dim=1 等价
```

**`dim=-1` 是惯例**：不管张量多少维，"最后一维"通常是 "特征/类别" 维 → 跨任意维度通用。

### 形状演变

```
logits.shape  = (4, 1000)
                 ^   ^
                 |   |
                 |   └── dim=-1，沿这个维度找最大 → 被消掉
                 └────── dim=0，保留

argmax(dim=-1).shape = (4,)    ← 1000 这一维被"消掉"，只剩 4
```

通用规则：**`argmax(dim=k)` 把第 k 维去掉**（同 `max`、`sum`、`mean` 等聚合算子）。

### 具体例子

```python
logits = torch.tensor([
    [0.1, 0.5, 0.2, 0.8],   # 样本 0，4 个类别
    [0.9, 0.1, 0.3, 0.4],   # 样本 1
    [0.2, 0.7, 0.6, 0.3],   # 样本 2
])
# shape = (3, 4)

logits.argmax(dim=-1)
# → tensor([3, 0, 1])
#
#   样本 0 → 类别 3 (0.8 最大)
#   样本 1 → 类别 0 (0.9 最大)
#   样本 2 → 类别 1 (0.7 最大)
```

### 配合 gather / 直接索引

`argmax` 拿到的索引常用来 **取回对应的值** 或 **查表**：

```python
# 取每个样本的最大值（等价于 .max(dim=-1).values）
top1_idx = logits.argmax(dim=-1)        # (4,)
top1_val = logits.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)

# 查类别名（ImageNet 1000 类）
class_names = ['cat', 'dog', ...]       # len=1000
predictions = [class_names[i] for i in top1_idx.tolist()]
```

### `argmax` 的一些陷阱

#### 1. 不可导（不能用于训练）

`argmax` 输出整数，**没有梯度**。所以训练时算 loss 用的是 softmax + cross-entropy（连续的），inference 时才用 argmax 拿最终预测。

#### 2. 平局只返回第一个

```python
torch.tensor([0.5, 0.5, 0.5]).argmax()  # → tensor(0)  ← 取最小索引
```

#### 3. 想同时拿索引和值，用 `max`

`max(dim=...)` 返回一个命名元组，同时含 values 和 indices：

```python
result = logits.max(dim=-1)
result.values   # (4,)  每行最大值
result.indices  # (4,)  每行最大值的位置 = argmax 结果
```

### 兄弟函数

| 函数 | 返回 | 用途 |
|------|------|------|
| `argmax` | 最大值的索引 | 分类预测、top-1 |
| `argmin` | 最小值的索引 | 找最近邻、最小损失 |
| `argsort` | 按值排序后的索引 | top-k、ranking |
| `topk(k)` | 前 k 大的值 + 索引 | beam search、top-5 准确率 |

### 一句话总结

> `argmax(dim=-1)` = "**沿最后一维找最大值，返回它的位置**"。在分类里，最后一维通常是 **类别维度**，所以 `logits.argmax(dim=-1)` 就是 **每个样本预测的类别号**。
