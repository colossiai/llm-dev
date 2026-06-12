# sigmoid求导


${\Huge S(x)=\frac{1}{1+e^{-x}} }$

${\Huge \text{Let } y=1+e^{-x} \text{, so } S(x) = y^{-1}}$

```math

{\Huge

\begin{aligned}

\frac{d}{dx}S(x)
&=\frac{d}{dy}S(x) \cdot \frac{dy}{dx} \quad (\text {Chain Rule})\\
&=-1(y)^{-2} \cdot (0 + \frac{d}{dx}e^{-x})\\
&=-1(y)^{-2} \cdot \frac{d}{dx}e^{-x}\\
&=-1(y)^{-2} \cdot (\frac{d}{du}e^u \cdot \frac{d}{dx}u) \quad (\text {Let } u = -x) \\
&=-1(y)^{-2} \cdot (e^u \cdot -1)\\
&=-1(y)^{-2} \cdot (e^{-x} \cdot -1)\\
&=\frac{e^{-x}}{y^2}\\
&=\frac{y-1}{y^2}\\
&=\frac{1}{y} \cdot (1 - \frac{1}{y}) \\
&=S(x) \cdot (1 - S(x))\\

\end{aligned}

}

```


## Sigmoid 函数

Sigmoid 函数是神经网络历史上最经典的激活函数之一：

```math
\sigma(x)=\frac{1}{1+e^{-x}}
```

它的图像是一条 S 型曲线：

* 当 (x \to +\infty) 时，输出接近 1
* 当 (x \to -\infty) 时，输出接近 0
* 当 (x = 0) 时，输出为 0.5

---

## 化简成神经网络最常用形式

观察：

```math
\sigma(x)
=
\frac{1}{1+e^{-x}}
```

以及

```math
1-\sigma(x)
=
1-\frac{1}{1+e^{-x}}
=
\frac{e^{-x}}{1+e^{-x}}
```

两者相乘：

```math
\sigma(x)(1-\sigma(x))
=
\frac{1}{1+e^{-x}}
\cdot
\frac{e^{-x}}{1+e^{-x}}
=
\frac{e^{-x}}{(1+e^{-x})^2}
```

恰好等于刚才求出的导数。

因此：

```math
\frac{d\sigma(x)}{dx}=\sigma(x)(1-\sigma(x))
```

这是机器学习和深度学习里经常直接使用的公式。

---

## 为什么这个结果这么漂亮？

设

```math
y=\sigma(x)
```

那么：

```math
y' = y(1-y)
```

导数只需要知道当前输出 (y)，不需要再次计算指数函数。

例如：

### 当 (x=0)

```math
y=\sigma(0)=0.5
```

导数：

```math
y' = 0.5(1-0.5)
```

```math
=0.25
```

这是 Sigmoid 的最大斜率。

---

### 当 (x=5)

```math
y\approx0.993
```

导数：

```math
y'(5)
=
0.993(1-0.993)
```

```math
\approx0.007
```

已经非常小。

---

### 当 (x=-5)

```math
y\approx0.007
```

导数：

```math
0.007(1-0.007)
```

```math
\approx0.007
```

同样非常小。

---

## 为什么会导致梯度消失？

因为：

```math
\sigma'(x)=\sigma(x)(1-\sigma(x))
```

而

```math
0 < \sigma(x) < 1
```

所以：

```math
0 < \sigma'(x) \le 0.25
```

导数最大也只有 0.25。

多层网络反向传播时，梯度会不断相乘：

```math
0.25 \times 0.25 \times 0.25 \times \cdots
```

例如：

```math
0.25^{10}
=
9.5\times10^{-7}
```

几乎变成 0。

这就是深层网络中著名的 **梯度消失（Vanishing Gradient）** 问题，因此现代 LLM 和大多数深度网络更常使用：

* ReLU
* GELU
* SiLU

而较少使用 Sigmoid 作为隐藏层激活函数。

---

记忆一个最重要的结论即可：

```math
\boxed{
\frac{d}{dx}\sigma(x)
=
\sigma(x)\bigl(1-\sigma(x)\bigr)
}
```

这是反向传播推导中出现频率最高的公式之一。
