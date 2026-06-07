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

`列向量: W.x` 课本: (3,4) dot (4,5) = (3,5)

`行向量: x.W` ML代码 (4,5)^T dot (3,4)^T =  (5,4) dot (4,3) = (5,3) = (3,5)^T

