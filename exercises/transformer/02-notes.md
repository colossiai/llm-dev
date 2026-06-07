# 02 笔记

## 理解 `nn.Linear`:就是一个矩阵 + 偏置

### 疑问

> `nn.Linear` 实际上就是定义一个矩阵, 作为后面线性变换?
> 看注释 `Applies a linear transformation to the incoming data: y = xA^T + b`
> 所以它接受的是行向量参数

完全对!这两个理解都对 ✓

---

## 1. `nn.Linear` = 一个矩阵 + 一个偏置

源码本质就这么简单:

```python
class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # 就两个可学习参数:
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  # 矩阵 A
        self.bias = nn.Parameter(torch.zeros(out_features))                  # 偏置 b

    def forward(self, x):
        # 线性变换 + 偏置
        return x @ self.weight.T + self.bias    # ← 就这一行
```

→ "线性变换" = `y = xA^T + b`, 没有任何花哨的东西。

---

## 2. 接受行向量(`(batch, in_features)`)

```python
linear = nn.Linear(in_features=8, out_features=4)

x = torch.randn(3, 8)    # ← 3 个样本, 每个 8 维, 行向量
y = linear(x)
print(y.shape)            # (3, 4) ← 每行变换成 4 维
```

输入第 0 维一般是 batch 维(或 batch + seq_len 等), 最后一维是 features。

---

## 关键细节:为什么 weight 存成 `(out, in)` 而不是 `(in, out)`?

这其实是个**让用户读代码舒服的小心机**:

```python
nn.Linear(in_features=8, out_features=4)
        ↓
self.weight.shape = (out_features, in_features) = (4, 8)
        ↑
        和数学课本一致: "out × in" 矩阵, 直觉是"输出是输入的 4 维投影"
```

你写 `Linear(8, 4)`, 直觉是"8 维 → 4 维", weight 也是 `(4, 8)`, **像数学课本那样左乘行得通**。

但**实际计算**时, 因为输入是行向量(batch 在前):

```python
# 数学课本视角 (列向量, 一次一个样本)
y_col = W @ x_col           # (4, 8) @ (8, 1) = (4, 1)

# PyTorch 实际计算 (行向量, 批处理)
y_row = x_row @ W.T          # (3, 8) @ (8, 4) = (3, 4)
                  ^^^^^
              转置一下, 让行向量乘法行得通
```

→ **存储用数学约定, 计算用 ML 约定**, 通过一次 `.T` 桥接。这就是 docstring 里 `y = xA^T + b` 的 `^T` 的来源。

---

## 一图看清

```
   用户写:               nn.Linear(8, 4)
   →
   存储:        self.weight = (4, 8) 矩阵   ← 数学约定 (out × in)
                self.bias   = (4,)
   →
   输入:        x.shape = (batch, 8)        ← ML 行向量约定
   →
   计算:        y = x @ self.weight.T + b
                  (batch, 8) @ (8, 4) = (batch, 4)
   →
   输出:        y.shape = (batch, 4)
```

---

## 在 02 脚本里的使用

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        # 4 个 Linear 层 — 4 个矩阵 (无 bias, 现代 LLM 习惯)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (B, T, d_model)
        Q = self.W_q(x)   # 等价于 x @ self.W_q.weight.T, shape (B, T, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        ...
```

→ **`self.W_q(x)` 一行调用, 内部就是 `x @ weight.T`**。
→ 在 01 脚本(numpy)里我们写 `Q = X @ W_Q`, 在 02 脚本(PyTorch)里写 `Q = self.W_q(x)` — 干的是同一件事, 只是 PyTorch 替你管了参数和转置。

---

## 一句话总结

> **`nn.Linear(in, out)` = 存一个 `(out, in)` 的矩阵 W 和 `(out,)` 的偏置 b,
> 接受 `(..., in)` 的行向量, 通过 `y = x @ W^T + b` 输出 `(..., out)`。**
>
> 矩阵是数学约定(`out × in`), 计算是 ML 行向量约定 — 中间靠一次转置桥接。

---

## `self.register_buffer("mask", mask)` 是什么?

### 一句话直觉

> **`register_buffer` = 把一个张量绑到 model 上, 让它跟着 model 走(`.to(device)` 也会带过去), 但不参与训练(没梯度)。**

适用于"我需要这个张量, 但它是常量, 不该被学习"的场景 — 比如**因果掩码**、归一化的统计量(running mean/var)、位置编码表等。

---

### nn.Module 里的"三种存储方式"

| 存储方式 | 学习/有梯度? | 在 state_dict? | `.to(device)` 跟随? | 例子 |
|---------|------------|--------------|-------------------|------|
| `nn.Parameter` | ✓ 学 | ✓ | ✓ | W, b(权重、偏置) |
| `register_buffer` | ✗ 不学 | ✓ | ✓ | 掩码、BN 的 running mean |
| 普通 `self.x = tensor` | ✗ 不学 | ✗ | ✗ | 临时变量(踩坑高发) |

→ **`register_buffer` 是"会跟着模型走的常量"**。

---

### 为什么因果掩码用 `register_buffer` 而不用普通赋值?

来对比看。假设我们写错成普通赋值:

```python
# ❌ 错误写法
self.mask = torch.tril(torch.ones(64, 64))   # 普通 tensor, 没注册
```

会出三个问题:

#### 1. `.to('cuda')` 不会把 mask 也搬到 GPU

```python
model = CausalSelfAttention(d_model=8).to('cuda')
# 现在 model 的 weight 在 cuda, 但 model.mask 还在 cpu!
# 前向计算时会报错: "tensor on cpu and cuda"
```

#### 2. `state_dict()` 不会保存 mask

```python
torch.save(model.state_dict(), "model.pt")
# 加载时 mask 字段缺失, 要重新生成
```

#### 3. 没法用 `model.mask` 这种属性访问的"模块化"语义

---

### 如果误用 `nn.Parameter` 呢?

```python
# ❌ 也错: 把 mask 当参数
self.mask = nn.Parameter(torch.tril(torch.ones(64, 64)))
```

问题:

- mask 会参与梯度计算和优化 — 训练过程中会被更新成"不是下三角"了!
- 因果约束彻底失效, 模型作弊看未来。
- 浪费内存(每个参数都要存 grad)。

→ **常量结构必须用 buffer, 不能用 Parameter**。

---

### 在 02 脚本里的实际效果

```python
mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
self.register_buffer("mask", mask)
```

之后可以:

```python
# ✓ 像普通属性访问
self.mask[:T, :T]

# ✓ 训练时不算它的梯度
# ✓ model.to('cuda') 自动搬到 GPU
# ✓ torch.save(model.state_dict()) 会保存它
# ✓ model.eval() / train() 都不影响它
```

---

### 类比一下

把 `nn.Module` 想象成一个"工具箱":

| 角色 | 类比 |
|------|------|
| `nn.Parameter` | **可调节的旋钮** — 训练时一直在转(更新) |
| `register_buffer` | **固定的零件** — 不调, 但跟着工具箱搬家 |
| 普通 `self.x` | **临时放进去的便利贴** — 工具箱一搬就丢了 |

---

### 其他常见 buffer 使用场景

| 场景 | 为什么用 buffer |
|------|---------------|
| 因果掩码(本例) | 下三角矩阵是结构常量 |
| BatchNorm/RMSNorm 的 running mean/var | 推理时要用, 但不靠梯度更新 |
| 位置编码表(原始 Transformer 的 sin/cos) | 固定值, 跟着模型搬 |
| RoPE 的 cos/sin 预计算表 | 同上, LLaMA 也用 buffer 存 |

---

### 一句话总结

> **`register_buffer("mask", tensor)` = "存一个常量 tensor 当 model 的一部分:**
>
> - 不学(无梯度)
> - 跟着 model 搬(`.to(device)`)
> - 跟着 model 存(`state_dict`)
> - 通过 `self.mask` 访问"
>
> 因果掩码的下三角结构是**约束, 不是学的东西**, 所以用 buffer 最合适。
