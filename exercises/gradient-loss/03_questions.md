# Gradient & Loss 学习对话记录 (续 2)

---

## Q1: `exercises/gradient-loss/03_minimize_loss_is_downhill.py`, `requires_grad=True` 是什么意思?

### 回答

`requires_grad=True` 是创建张量时的一个**标记**, 它告诉 PyTorch:

> **"这个张量是我要学习的参数, 请帮我追踪它参与的所有运算, 以后好对它求梯度。"**

### 在 03 脚本里它在哪?

```python
w = torch.tensor(-1.5, requires_grad=True)     # ← w 是要学的, 加标记
b = torch.tensor(-2.5, requires_grad=True)     # ← b 是要学的, 加标记
```

而 `x_data, y_data` 没有这个标记——因为它们是**固定的训练数据**, 不是要学的东西。

### 这个标记的三重作用

#### 1. **建计算图** (track 操作)
没标记的张量, PyTorch 算完就算完了:
```python
x = torch.tensor(3.0)              # 默认 requires_grad=False
y = x ** 2                          # y 就是个数, 9.0, 没人记得它怎么来的
y.backward()                        # ❌ 报错: 没图可反向
```

加了标记后, 每一步运算都被**偷偷记下来**:
```python
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2                          # PyTorch 记: "y = x², 反向时要用 2x"
y.backward()                        # ✅ 算出 x.grad = 6.0
```

#### 2. **接收 `.grad` 字段**
只有带 `requires_grad=True` 的"叶子张量" (leaf tensor, 你直接创建的) 才会被分配 `.grad` 属性:

```python
w = torch.tensor(-1.5, requires_grad=True)
print(w.grad)         # None (还没 backward)
loss.backward()
print(w.grad)         # tensor(...)  ← 这里出现梯度
```

#### 3. **会传染** (autograd 推断)
任何张量, 只要**参与运算的某一方** `requires_grad=True`, 结果也会自动 `requires_grad=True`:

```python
w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)                  # 没标记
y = w * x                              # y.requires_grad → True (因为 w 有标记)
```

这就是为什么 `loss = ((w*x_data + b - y_data)**2).mean()` 这个 loss 自动可以 `.backward()`——因为它"沾上了"w 和 b 这两个有标记的张量。

### 在 03 的完整闭环里看它的角色

```python
w = torch.tensor(-1.5, requires_grad=True)   # ① 标记: 这是参数
b = torch.tensor(-2.5, requires_grad=True)   # ① 标记: 这是参数

for step in range(80):
    loss = loss_fn(w, b)                      # ② loss 沾上了 w, b → 进图
    loss.backward()                           # ③ 反向: 沿图回溯, 算出 w.grad, b.grad
    with torch.no_grad():
        w -= lr * w.grad                      # ④ 用 grad 更新参数 (这步要 no_grad)
        b -= lr * b.grad
    w.grad.zero_()                            # ⑤ 清零, 不然下一轮会累加
    b.grad.zero_()
```

**没有 `requires_grad=True`, 整个循环就废了** —— ② 不会建图, ③ 没法 backward, ④ 没有 `.grad` 可用。

### 谁该加, 谁不该加?

| 张量 | 加 `requires_grad=True`? | 原因 |
|---|---|---|
| 模型参数 (w, b, weights, bias) | ✅ 加 | 这些是要学的 |
| 输入数据 (x_data, images) | ❌ 不加 | 数据是固定的, 不是学的 |
| 标签 (y_data, labels) | ❌ 不加 | 同上 |
| 中间结果 (y_pred, loss) | 不用手动加 | autograd 自动传染上 |
| 推理时的输入 | ❌ 不加 | 推理不需要梯度 |

### 跟之前学过的概念串起来

| 关键词 | 干啥的 | 类比 |
|---|---|---|
| `requires_grad=True` | **打开**追踪 | "开始录像" |
| 默认状态 (`False`) | 不追踪 | "摄像机关着" |
| `torch.no_grad()` | **临时**关闭追踪 | "暂停录像" (前面 Q1 讲过) |
| `.backward()` | 沿录像反向算梯度 | "倒带" |
| `.grad` | 算出来的梯度存放处 | "导演笔记" |
| `.grad.zero_()` | 清空梯度 | "撕掉旧笔记" |

### 一句话总结

> **`requires_grad=True` = 给张量贴一个"我是参数, 请追踪我"的标签**, 让 autograd 在后续运算中把它纳入计算图, 这样 `.backward()` 才能为它算出梯度, 存进 `.grad`, 给梯度下降用。
>
> **不贴这个标签的张量 = 死数据**, 不参与学习, 也算不出梯度。

`nn.Linear`、`nn.Parameter` 这些 PyTorch 内建的模型组件, 内部的权重张量**默认就帮你贴好了** `requires_grad=True`——所以平时写大模型时你看不到这个标记, 但它就在那里, 在每个可学习参数上。
