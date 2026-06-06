# MATH-SUMMARY-1: LLM 基础数学 · PyTorch 实践小结

本轮覆盖了 LLM 数学的 9 个最基础概念，通过 `math1/` 下 4 个可运行的 PyTorch
脚本动手实践。

## 概念 → 代码 索引

| 概念 | 关键直觉 | PyTorch 关键 API | 文件 |
|---|---|---|---|
| 向量 (Vector) | 一组带方向的数 | `torch.tensor([...])` | `math1/01_vectors_dot.py` |
| 维度 (Dimension) | 张量的形状 | `.shape` / `.dim()` / `.ndim` | `math1/01_vectors_dot.py` |
| 点积 (Dot Product) | 两向量"方向有多接近" | `torch.dot(a, b)`、`a @ b` | `math1/01_vectors_dot.py` |
| 矩阵乘法 | 对向量做"空间变换" | `A @ B`，`(m,k)@(k,n)→(m,n)` | `math1/02_matrix_transform.py` |
| 线性变换 | 矩阵把图形旋转/缩放/剪切 | `(M @ points.T).T` | `math1/02_matrix_transform.py` |
| 基底 (Basis) | 矩阵的列 = 变换后的新基底 | `M[:, 0]`、`M[:, 1]` | `math1/02_matrix_transform.py` |
| 投影 (Projection) | a 沿 b 方向的"影子" | `(a·b / b·b) * b` | `math1/03_projection.py` |
| 导数 (Derivative) | f(x) 的瞬时变化率 | `requires_grad=True` + `.backward()` | `math1/04_derivative_gradient.py` |
| 梯度 (Gradient) | 多元函数上升最快的方向 | `x.grad`、`x.grad.zero_()` | `math1/04_derivative_gradient.py` |

## 三句话总结

1. **向量 + 矩阵 = LLM 的基本货币**：一个 token 在模型里就是一个向量；矩阵
   就是把向量"搬到新空间"的操作。注意力机制里频繁出现的 `Q @ K^T`、
   `nn.Linear` 内部，本质都是矩阵乘法。

2. **点积 / 投影 / cos 相似度是一家**：embedding 之间"语义有多接近"用 cos
   相似度衡量，它就是归一化后的点积。RAG 检索、聚类、最近邻搜索都靠它。

3. **梯度下降 = LLM 训练**：整个训练过程就是循环 "前向算 loss → backward
   算梯度 → 沿 -∇loss 走一步"。`math1/04_derivative_gradient.py` 第 65 行
   附近那 5 行循环就是缩微版——真实 LLM 只是参数从 2 个变成几十亿个。

## 运行方式

```
cd math1
uv run 01_vectors_dot.py
uv run 02_matrix_transform.py     # 生成 matrix_transform.png
uv run 03_projection.py           # 生成 projection.png
uv run 04_derivative_gradient.py  # 生成 gradient_descent.png
```

## 下一步建议

- 把 `04` 的梯度下降扩展成 **线性回归**：给一组 `(x, y)` 点，用梯度下降拟合
  `y = w·x + b`，自己写训练循环。
- 把 `plot_embeddings.py` 里的随机 `nn.Embedding` 换成真实模型的 embedding
  （比如 `sentence-transformers`），看 cos 相似度是否真的反映语义。
- 学 `torch.nn.Linear` 和 `nn.Module`：把矩阵乘法 + 梯度下降组装成一个标准
  的 "可训练模块"，这就是搭建 Transformer 的最小积木。
