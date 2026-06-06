# 神经网络学习脚本

参考 `math-concepts/neural-network/学习神经网络需要哪些前置知识.md` 末尾推荐路线编写。
**前置已学完**:矩阵乘法、向量点积、导数、链式法则、Softmax、Cross Entropy。

## 学习路径(按顺序运行)

| 序号 | 脚本 | 学什么 |
|------|------|--------|
| 01 | `01_perceptron.py` | 单个神经元 = 一条直线分两类 |
| 02 | `02_activation_functions.py` | 6 种激活函数(ReLU/Sigmoid/Tanh/GELU/LeakyReLU/SiLU)及导数 |
| 03 | `03_why_nonlinearity.py` | XOR 实验:为什么没有非线性激活就不行 |
| 04 | `04_backprop_numpy.py` | 纯 numpy 手写反向传播 |
| 05 | `05_autograd_pytorch.py` | PyTorch autograd,与 04 数值对比验证 |
| 06 | `06_mlp_complete.py` | nn.Module + Adam + make_moons 完整训练 |

## 运行方式

```bash
cd exercises

# 默认: 只打印日志, 不画图 (快, 适合反复跑)
uv run python neural-network/01_perceptron.py

# 加 --plot: 生成 PNG 到 plots/ 子目录
uv run python neural-network/01_perceptron.py --plot
```

依次跑全部 6 个脚本 (都加 --plot):

```bash
for s in neural-network/0*.py; do
    uv run python "$s" --plot
done
```

每个脚本会:
- 在控制台打印训练日志和关键数值
- 加 `--plot` 才生成对应 PNG 到 `plots/0X_xxx.png`

## 推荐学习节奏

1. **01-02**:理解最小单元 + 激活函数,30 分钟
2. **03**:亲眼看到"线性激活的网络学不会 XOR",10 分钟
3. **04**:这一步最关键。读懂 forward + backward 的每一行,推导链式法则
4. **05**:运行后看到 `max |diff| < 1e-9`,确认 04 推导正确
5. **06**:看现代 PyTorch 工程代码长什么样,体会 nn.Module 的便利

## 关键产出

- `04` 和 `05` 共用同一组初始参数 → 验证手写 backprop 与 autograd 一致
- `06` 训练完决策边界会从直线弯成月牙形,直观看到 MLP 的非线性表达力

## 下一步

学完这 6 个脚本后,自然进入 Attention / Transformer:
- 手写 Single-Head Attention
- Multi-Head Attention
- Transformer Block (Attention + FFN + Residual + LayerNorm)
