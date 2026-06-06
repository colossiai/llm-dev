# PyTorch 张量基本功（01–08）

> 你已经掌握了：向量、矩阵、矩阵乘法、点积、线性变换、维度、基底、投影、导数
>
> 本目录帮你把这些数学概念**对应到 PyTorch 的实际张量操作**。

学完本目录后再去 `../pytorch-llm-basic/`（09–17：手写最小 LLM 必备）。

---

## 环境准备（使用 uv）

```bash
cd /Users/ericyeung/ai-space/claude-buildllm/exercises/pytorch-tensors

# 初始化 uv 项目（如果还没有）
uv init --no-readme

# 安装依赖（Intel Mac 用 CPU 版即可）
uv add torch matplotlib numpy

# 如果走 Zscaler 网络遇到 TLS 问题：
# uv --native-tls add torch matplotlib numpy
```

## 运行方式

```bash
# 单独运行 (默认: 只打印控制台输出)
uv run 01_shape.py

# 加 --plot 才会保存可视化图到 plots/
uv run 01_shape.py --plot

# 或运行所有 (含可视化)
for f in [01]*.py; do echo "=== $f ==="; uv run "$f" --plot; done
```

每个脚本都接受 `--plot` 参数：
- 不加: 只控制台输出
- `--plot`: 同时保存 PNG 到 `./plots/`

---

## 学习顺序

| 文件 | 主题 | 对应数学/概念 |
|---|---|---|
| `01_shape.py` | 张量形状 | 向量(1D) / 矩阵(2D) / 高维张量 |
| `02_reshape_view.py` | 重塑维度 | "同一组数据的不同排列方式" |
| `03_transpose_permute.py` | 转置 / 维度交换 | 矩阵转置 $A^T$ |
| `04_batch_matmul.py` | 批量矩阵乘法 | 批量线性变换 |
| `05_broadcasting.py` | 广播机制 | "自动补齐维度"以做加法 |
| `06_indexing_slicing.py` | 索引与切片 | 选行 / 选列 / 选元素 |
| `07_unsqueeze_squeeze.py` | 增 / 减维度 | 在做广播或 matmul 前调形状 |
| `08_contiguous.py` | 内存连续性 | 为什么 view 有时会报错 |

建议**按顺序**学完。后面（`pytorch-llm-basic` 09–17）的概念依赖前面的。

---

## 每个文件的结构

```
1. 概念说明（注释）
2. 代码演示（print 出每一步）
3. 可视化（保存 PNG 到 plots/）
4. 小练习（带 assert 检查，TODO 留给你填）
```

碰到 `# TODO`，先自己想答案再看下面的参考。
所有 `assert` 通过说明你写对了。

---

## 学完之后你应该能回答

1. `x.shape == (4,)` 和 `x.shape == (4,1)` 有什么区别？
2. `view` 和 `reshape` 各自什么时候用？
3. `(B, M, K) @ (B, K, N)` 输出什么形状？
4. 形状 `(3,1)` 和 `(1,4)` 相加后是什么形状？
5. 为什么 `x.transpose(0,1).view(-1)` 会报错？

更系统的概念笔记见 `pytorch-notes.md`。
