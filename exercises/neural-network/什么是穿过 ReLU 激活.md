# 穿过 ReLU 激活


“**穿过 ReLU 激活（through ReLU）**”通常出现在神经网络前向传播和反向传播的讲解中，意思是：

> 数据（或梯度）经过 ReLU 这一层进行变换。

ReLU（Rectified Linear Unit）定义为：

f(x)=\max(0,x)

即：

* x > 0 → 输出 x
* x ≤ 0 → 输出 0

例如：

| 输入 x | ReLU(x) |
| ---- | ------- |
| -3   | 0       |
| -1   | 0       |
| 0    | 0       |
| 2    | 2       |
| 5    | 5       |

---

## 前向传播：数据穿过 ReLU

假设某个神经元计算得到：

```math
z = w x + b
```

结果：

```math
z = -2
```

穿过 ReLU：

```math
a = \mathrm{ReLU}(z)=0
```

流程：

```text
输入 x
   ↓
线性层 wx+b
   ↓
z = -2
   ↓
ReLU
   ↓
a = 0
```

这就叫：

> activation passes through ReLU

或者

> z goes through ReLU

---

## 反向传播：梯度穿过 ReLU

这时“穿过 ReLU”更重要。

ReLU 的导数：

\frac{d,\mathrm{ReLU}(x)}{dx}=\begin{cases}1,&x>0\0,&x<0\end{cases}

假设上一层传来的梯度是：

```math
\frac{\partial L}{\partial a}=5
```

### 情况1：z = 3

因为 z > 0，

```math
\frac{\partial a}{\partial z}=1
```

所以：

```math
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
=
5 \times 1
=
5
```

梯度成功穿过 ReLU。

```text
梯度 5
 ↓
ReLU导数=1
 ↓
梯度仍然是5
```

---

### 情况2：z = -3

因为 z < 0，

```math
\frac{\partial a}{\partial z}=0
```

所以：

```math
\frac{\partial L}{\partial z}
=
5 \times 0
=
0
```

梯度被截断了。

```text
梯度 5
 ↓
ReLU导数=0
 ↓
梯度变成0
```

这时经常会说：

> 梯度无法穿过 ReLU

（gradient cannot flow through ReLU）

---

## 为什么 ReLU 这么重要？

它相当于一个“门”：

```text
        z>0
梯度 ───────► 通过

        z<0
梯度 ───────► 阻断
```

因此：

* 正区间：神经元参与学习
* 负区间：神经元暂时不学习

这也是为什么会有“Dead ReLU（死亡 ReLU）”问题：

如果一个神经元长期输出负数：

```text
z < 0
↓
ReLU = 0
↓
梯度 = 0
↓
参数不更新
```

它就像“死掉”了一样。

---

在 LLM 的训练里，你经常会看到这样的描述：

```text
Linear
  ↓
ReLU / GELU
  ↓
Linear
```

前向传播时是：

```text
数据穿过激活函数
```

反向传播时是：

```text
梯度穿过激活函数
```

其中“穿过 ReLU”本质上就是：

> **先应用 ReLU 的函数值（前向），再应用 ReLU 的导数（反向）。**
