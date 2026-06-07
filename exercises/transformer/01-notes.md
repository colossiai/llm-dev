# 01 笔记

## 观察:`tokens` 好像没有和 `X` 关联起来?

你观察得很准!这是脚本的一个缺陷 — 代码里 `tokens` 只是给打印/画图用的**标签**, 跟 `X` 没有任何关系。

### 原代码的问题

```python
tokens = ["The", "cat", "sat", "on", "mat"]
X = np.random.randn(T, d_model)   # ← 完全独立的随机数
```

**X 是 5 行随机向量, 跟 token 是 "The" 还是 "cat" 完全无关**。即使把 tokens 改成 `["A", "B", "C", "D", "E"]`, X 也一模一样。

### 在真实 LLM 里应该是怎样

每个 token 通过 **embedding 层"查表"**得到向量:

```python
# 简化版的 embedding lookup
embedding_table = {
    "The": [0.21, -0.13, 0.45, ...],   # 固定向量
    "cat": [0.88, 0.32, -0.17, ...],
    ...
}
X[i] = embedding_table[tokens[i]]
```

→ 同一个 token 在不同位置应该拿到**相同**的向量(语义一致)。
→ 而原脚本里, 把 tokens 里的 "the" 改成 "cat", X 不会变。

---

## 修复:让 tokens 真正驱动 X

### 改造后的代码

```python
# ===== 模拟 embedding lookup =====
# 真实 LLM 里 nn.Embedding(vocab_size, d_model) 干的事:
#   每个唯一的 token 在 "embedding 表" 里对应一个固定向量
#   同一个 token (无论出现在哪个位置) 拿到相同的向量
tokens = ["The", "cat", "sat", "on", "the", "mat"]   # 包含重复的 "the"/"The"
unique_vocab = sorted(set(t.lower() for t in tokens))
embedding_table = {t: np.random.randn(d_model) for t in unique_vocab}
# 查表: tokens[i] 对应的向量 → X[i]
X = np.array([embedding_table[t.lower()] for t in tokens])
T = len(tokens)
```

### 验证效果

修改后注意力矩阵立刻能看到证据:

```
                The     cat     sat      on     the     mat
  The   |   0.067   0.036   0.817   0.005   0.067   0.007    ← The 和 the 行
  cat   |   0.498   0.000   0.000   0.000   0.498   0.003       完全相同!
  sat   |   0.165   0.227   0.077   0.238   0.165   0.128
  on    |   0.000   0.221   0.014   0.748   0.000   0.017
  the   |   0.067   0.036   0.817   0.005   0.067   0.007    ← 和 The 一行一样
  mat   |   0.243   0.117   0.145   0.095   0.243   0.156
```

→ **"The" 和 "the"(都映射到同一个 embedding)行向量完全一样**, 因为同样的 X[i] → 同样的 Q[i] → 同样的 attention 权重。

---

## 修改对比

| | 之前 | 现在 |
|---|------|------|
| `tokens` 的角色 | 只是 print/画图的标签 | 真正驱动 X 的内容 |
| X 的来源 | `np.random.randn(T, d_model)` | `X[i] = embedding_table[tokens[i]]` |
| 改 tokens | 不影响 X | 直接改 X |
| 重复 token | 拿不同向量(违反语义) | 拿相同向量 ✓ |

新增验证打印:

```
验证: X[0](The) == X[4](the) ?  → True ✓ (同一 token 同一向量)
```

让读者一眼看到 embedding lookup 在工作。

---

## 这就是 LLM 里 `nn.Embedding` 的真实行为

PyTorch 里 `nn.Embedding(vocab_size, d_model)` 本质上是个**查表函数**:

```python
# 概念上等价于
class Embedding:
    def __init__(self, vocab_size, d_model):
        self.table = np.random.randn(vocab_size, d_model)  # 可学习的表

    def __call__(self, token_ids):
        return self.table[token_ids]   # 查表
```

- **vocab_size 行**(每行一个 token 的向量)
- **d_model 列**(向量维度)
- `token_id → vector`: 同一个 token id 总是返回同一个向量
- 表的内容是**可训练的参数**, 反向传播会更新它们

→ 这是为什么 LLM 能"理解" token 语义 — 训练过程中, **相关的 token 的 embedding 向量会被拉近**, 不相关的会被推远。

---

## 疑问:`Q = X @ W_Q` 还是 `Q = W_Q @ X`?

之前学线性变换的时候, 矩阵 M 乘以向量 V 是把 V 投射到 M 坐标。
那这里 `Q = X @ W_Q` 是否应该写成 `Q = W_Q @ X`, 把 X 投射到 W_Q 坐标空间?

非常好的问题!这是 ML 代码和数学课本之间最容易绕晕的地方。两种写法**都是对的, 只是用了不同的"数据排列约定"**。

### 数学课本约定 vs ML 代码约定

#### 数学课本(列向量约定)

向量是**列向量**, 变换写成 `y = M x`:

```
x = ⎡ x₁ ⎤  (8 × 1 列向量, d_model 维)
    ⎢ x₂ ⎥
    ⎢ .. ⎥
    ⎣ x₈ ⎦

W_Q shape = (d_k, d_model) = (8, 8)   ← "out × in"

Q = W_Q @ x:  (8, 8) @ (8, 1) = (8, 1)
              "M 把 x 投射到新坐标空间"
```

→ "向量从右边进, 矩阵在左边作用", 这是线性代数标准写法。

#### ML 代码(行向量约定 — 我们用的)

每个 token 是**行向量**, 多个 token 堆成矩阵 X(每行一个 token):

```
X = ⎡ ─── token 0 (8 维) ─── ⎤    (T × d_model) = (5, 8)
    ⎢ ─── token 1 (8 维) ─── ⎥
    ⎢ ........................⎥
    ⎣ ─── token 4 (8 维) ─── ⎦

W_Q shape = (d_model, d_k) = (8, 8)   ← "in × out"

Q = X @ W_Q:  (5, 8) @ (8, 8) = (5, 8)
              每行: token 的 query 向量
```

### 两种写法的关系:互为转置

它们**本质上是同一个变换**, 只是数据摆放不同。

| | 列向量约定 | 行向量约定(我们的) |
|---|---|---|
| 单个 token | `x` shape `(d, 1)` | `x^T` shape `(1, d)` |
| 矩阵 W_Q | `(d_k, d_model)` | `(d_model, d_k)` 转置过 |
| 公式 | `q = W_Q x` | `q = x W_Q` |
| 结果 q | `(d_k, 1)` 列 | `(1, d_k)` 行 |

数学等价性:

$$
(W_Q \cdot x)^T = x^T \cdot W_Q^T
$$

所以 `Q = X @ W_Q` **等价于** "对每个 token $x_i$ 做 $q_i = W_Q^T x_i$"。

### 为什么 ML 选行向量?

1. **批处理自然**:`(batch, features)` 是表格数据的标准摆法
2. **内存布局**:row-major 顺序遍历样本最快(C/numpy/PyTorch 都是 row-major)
3. **可读性**:`X[i]` 直接拿到第 i 个样本, 而列向量约定要写 `X[:, i]`
4. **所有 ML 框架统一了这个约定**:numpy/PyTorch/TF/JAX 都是行向量

### 一个有趣的细节:PyTorch 的 `nn.Linear`

PyTorch 内部其实**用了数学约定存储 W**, 但计算时转置:

```python
linear = nn.Linear(in_features=8, out_features=8)
linear.weight.shape   # (out, in) = (8, 8)  ← 数学约定!

# 但 forward 时:
# output = input @ self.weight.T   (内部转置)
```

→ 这是为了让用户**写代码时**用 `Linear(in, out)` 直觉(对应数学 W 的 `(out, in)`),
但**计算时**走行向量约定的 `X @ W^T`。

### 在我们的 numpy 代码里要选哪种?

如果你**真的想用数学约定**, 可以写成:

```python
# 数学约定: tokens 作为列, W 作为 "out × in"
X_col = X.T              # (d_model, T) = (8, 5)
W_Q_math = W_Q.T         # (d_k, d_model) = (8, 8)
Q_col = W_Q_math @ X_col # (d_k, T) = (8, 5), 列是 token query
```

但这样后面所有代码都要跟着调, 不如**全程坚持行向量约定**:

```python
Q = X @ W_Q   # (T, d_model) @ (d_model, d_k) = (T, d_k)
```

### 一句话总结

> **`Q = X @ W_Q` 是 ML 行向量约定下的"X 中每个 token 通过 W_Q 投影"** —
> 完全等价于数学课本的 `q = W_Q^T x`(对每个 token)。
>
> **不要混用**两种约定, 代码会一团乱。坚持"X 是 (batch, features), 每行一个样本",
> 后面所有矩阵乘法都跟着这个走 — 这是 numpy/PyTorch 整个生态系统的统一约定。

`列向量: W.x` 课本

`行向量: x.W` ML代码
