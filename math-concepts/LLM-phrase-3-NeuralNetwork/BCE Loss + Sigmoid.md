# 为什么 d(L)/d(z) = $\hat y$ - y

其中:
```math
L = BCE
```

```math
\hat y=\sigma(z)
```


---


这是神经网络里最经典、最重要的一个推导。

很多人第一次看到都会觉得：

> 怎么 BCE Loss + Sigmoid 这么复杂，
>
> 最后导数居然变成
>
```math
\frac{\partial L}{\partial z}=\hat y-y
```
>
> 这么简单？

其实是因为发生了一个非常漂亮的抵消。

---

# 第一步：写出 Loss

二分类交叉熵（BCE）：

```math
L
=
-\Big[
y\log(\hat y)
+
(1-y)\log(1-\hat y)
\Big]
```

其中

```math
\hat y=\sigma(z)
```

---

# 第二步：链式法则

我们要算：

```math
\frac{\partial L}{\partial z}
```

先求：

```math
\frac{\partial L}{\partial \hat y}
```

再乘：

```math
\frac{\partial \hat y}{\partial z}
```

即：

```math
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
```

---

# 第三步：求 dL/dŷ

对 BCE 求导：

```math
L=
-y\log(\hat y)
-(1-y)\log(1-\hat y)
```

所以：

```math
\frac{\partial L}{\partial \hat y}
=
-\frac{y}{\hat y}
+
\frac{1-y}{1-\hat y}
```

整理一下：

```math
\frac{-y(1-\hat y)+\hat y(1-y)}
{\hat y(1-\hat y)}
```

展开：

```math
\frac{-y+y\hat y+\hat y-y\hat y}
{\hat y(1-\hat y)}
```

中间两项抵消：


```math
\frac{\hat y-y}
{\hat y(1-\hat y)}
```

因此：

```math
\frac{\partial L}{\partial \hat y}
=
\frac{\hat y-y}{\hat y(1-\hat y)}
```

---

# 第四步：求 dŷ/dz

Sigmoid：

```math
\hat y=\sigma(z)\frac1{1+e^{-z}}
```

它有一个著名导数：

```math
\frac{\partial \hat y}{\partial z}=\hat y(1-\hat y)
```

这是神经网络必背公式之一。

如果画图：

```math
y=\frac{1}{1+e^{-x}}
```

你会看到中间斜率最大，两边越来越平。

---

# 第五步：乘起来

链式法则：

```math
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial \hat y}
\frac{\partial \hat y}{\partial z}
```

代入：

```math
\frac{\hat y-y}
{\hat y(1-\hat y)}
\times
\hat y(1-\hat y)
```

分子分母直接约掉：

```math
\hat y-y
```

得到：

```math
\boxed{
\frac{\partial L}{\partial z}
=
\hat y-y
}
```

---

# 为什么这么重要？

假设：

```math
y=1
```

模型预测：

```math
\hat y=0.9
```

那么：

```math
dL/dz=0.9-1=-0.1
```

负数。

说明：

> 增大 z 会降低 Loss。

因为 sigmoid(z) 会变得更接近 1。

---

再看另一种情况：

```math
y=1
```

但模型预测：

```math
\hat y=0.2
```

则：

```math
dL/dz=0.2-1=-0.8
```

梯度更大。

说明：

> 你错得很严重，
>
> 赶紧把参数往正确方向调整。

---

# 为什么大家喜欢 BCE + Sigmoid？

如果不用 BCE，而用 MSE：

```math
L=(\hat y-y)^2
```

则：

```math
\frac{\partial L}{\partial z}
=
2(\hat y-y)
\hat y(1-\hat y)
```

这里多出了：

```math
\hat y(1-\hat y)
```

当 sigmoid 饱和时：

```math
\hat y \approx 0
```

或

```math
\hat y \approx 1
```

会导致梯度接近 0。

训练变慢。

---

而 BCE + Sigmoid 恰好把这一项消掉：

```text
BCE 的导数:
(ŷ-y)/(ŷ(1-ŷ))

Sigmoid 的导数:
ŷ(1-ŷ)

相乘：

(ŷ-y)
```

因此梯度更干净、更稳定。

这也是为什么现代二分类神经网络几乎都会使用：

```text
Linear
→ Sigmoid
→ BCE Loss
```

或者在 PyTorch 中直接使用：

```python
nn.BCEWithLogitsLoss()
```

它把 **Sigmoid + BCE** 合并在一起，数学上等价于上面的推导，但数值稳定性更好。




