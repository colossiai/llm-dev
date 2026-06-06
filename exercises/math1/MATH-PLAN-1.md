# PyTorch 实践 Demo：巩固 LLM 数学基础

## Context

用户刚学完 LLM 相关的基础线性代数与微积分概念（见
`requirements/study-plan-requirement-1.md`），希望通过 PyTorch 动手代码加深
理解。项目目前已有 `plot_embeddings.py` 这一可视化示例，依赖已含 `torch`、
`matplotlib`、`scikit-learn`、`numpy`，无需再装包。

目标：把 9 个数学概念（向量、矩阵乘法、点积、线性变换、维度、基底、投影、
导数、梯度）映射到 4 个独立可运行的 PyTorch 脚本，每个脚本聚焦一个主题，
带 **中文注释**，关键处配 matplotlib 可视化，便于逐个吃透。

## 文件结构（4 个新文件）

每个文件可独立 `uv run <file>.py` 运行；注释与 `print` 输出统一使用中文。
风格参照已有的 `plot_embeddings.py`（模块 docstring + 行内解释 + 控制台
打印中间结果）。

### 1. `01_vectors_dot.py` — 向量 · 点积 · 维度

涵盖：**向量**、**点积**、**维度**

- 用 `torch.tensor([...])` 创建向量
- 向量加法、标量乘法、长度（`torch.norm`）、归一化
- 维度查看：`.shape`、`.dim()`、`.ndim`；演示 1D / 2D 张量区别
- 点积：`torch.dot(a, b)`、`a @ b`；手算验证 `sum(a * b)`
- 点积的几何意义：`cos θ = (a·b) / (|a| |b|)`
  - 用两组向量演示：平行 → cos≈1，垂直 → cos≈0，反向 → cos≈-1
- 控制台打印为主，无需画图

### 2. `02_matrix_transform.py` — 矩阵乘法 · 线性变换 · 基底

涵盖：**矩阵乘法**、**线性变换**、**基底**、**维度**

- 矩阵 × 向量：`M @ v` 把一个向量变成另一个向量
- 矩阵 × 矩阵：`A @ B`，并说明 `(m,k) @ (k,n) → (m,n)` 维度规则
- 在 2D 平面上对一组点（比如单位正方形的 4 个角）应用：
  - 旋转矩阵 `[[cosθ,-sinθ],[sinθ,cosθ]]`
  - 缩放矩阵 `diag(sx, sy)`
  - 剪切矩阵 `[[1,k],[0,1]]`
- 用 matplotlib 双子图对比 "变换前 vs 变换后" 的形状
- **基底**的直观演示：把标准基 `e1=[1,0]`、`e2=[0,1]` 经矩阵 M 变换后，
  结果正好是 M 的两列 → 说明 "矩阵的列就是新基底"

### 3. `03_projection.py` — 投影

涵盖：**投影**（顺便复用点积）

- 投影公式：`proj_b(a) = (a·b / b·b) * b`，用 PyTorch 算
- 分解 `a = proj_b(a) + a_perp`，验证 `a_perp · b ≈ 0`
- matplotlib 画：向量 a、向量 b、投影 proj_b(a)、垂直分量 a_perp
  - 用 `ax.quiver` 画箭头，标注长度
- 说明应用：点积本质上就是 "a 在 b 方向上的分量长度 × |b|"

### 4. `04_derivative_gradient.py` — 导数 · 梯度（autograd 入门）

涵盖：**导数**、**梯度** + 引出 "LLM 怎么训练"

- 单变量导数：`f(x) = x²`，用 `requires_grad=True` + `.backward()` 求
  `f'(2) = 4`，对比手算
- 多变量梯度：`f(x, y) = x² + y²`，求 `∇f(1, 2) = [2, 4]`
- matplotlib 画等高线 + 箭头（`ax.contour` + `ax.quiver`）展示梯度场
- 最后一个小高潮：手写最简 **梯度下降** 求 `f(x,y) = (x-3)² + (y+1)²`
  最小值
  - 循环 50 步，每步 `x -= lr * x.grad`、`y -= lr * y.grad`，记得
    `.grad.zero_()`
  - 在等高线上画出收敛轨迹，落到 `(3, -1)`
  - 注释里点明："LLM 训练就是这个套路，只是参数多了亿万倍"

## 需要修改的关键文件

仅新增，不修改现有文件：

- `pytorch-basic/01_vectors_dot.py` *(新)*
- `pytorch-basic/02_matrix_transform.py` *(新)*
- `pytorch-basic/03_projection.py` *(新)*
- `pytorch-basic/04_derivative_gradient.py` *(新)*

可复用已有：

- `pytorch-basic/plot_embeddings.py` — 注释风格（中文友好、概念先于代码）、
  matplotlib 双子图布局、`plt.savefig(...)` + `plt.show()` 模式

## 验证方式

1. 逐个运行：
   ```
   uv run 01_vectors_dot.py
   uv run 02_matrix_transform.py
   uv run 03_projection.py
   uv run 04_derivative_gradient.py
   ```
2. 每个脚本应：
   - 无报错
   - 控制台输出关键张量的值（便于对照手算）
   - 2/3/4 号脚本弹出 matplotlib 窗口并保存 PNG
3. 重点验收点：
   - `01`：cos 相似度三个用例（平行/垂直/反向）数值符合直觉
   - `02`：变换后图形可视上确实"转/缩/斜"了
   - `03`：`a_perp · b` 打印结果接近 0（浮点误差范围内）
   - `04`：梯度下降轨迹最终收敛到 `(3, -1)`，控制台打印最终 `(x, y)`
