# self.fc1(x) 是会自动调用 forward() 吗？


对，**会自动调用 `forward()`**。

这是 `nn.Module` 最重要的设计之一。

例如：

```python
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 4)

    def forward(self, x):
        return self.fc1(x)
```

当执行：

```python
y = self.fc1(x)
```

实际上不是直接调用 `Linear.forward()`。

而是：

```python
y = self.fc1.__call__(x)
```

然后 `__call__()` 内部再调用：

```python
y = self.fc1.forward(x)
```

即：

```text
self.fc1(x)

↓

self.fc1.__call__(x)

↓

self.fc1.forward(x)

↓

返回结果
```

---

## 为什么不直接写 forward()？

理论上可以：

```python
y = self.fc1.forward(x)
```

但 PyTorch 不推荐。

因为：

```python
self.fc1(x)
```

会经过 `nn.Module.__call__()`，里面做了很多额外工作：

* forward hook
* backward hook
* autograd 图构建
* mixed precision
* profiling
* distributed training

简化后源码类似：

```python
class Module:

    def __call__(self, *args):

        # 一些预处理

        result = self.forward(*args)

        # 一些后处理

        return result
```

所以官方建议：

```python
module(x)
```

不要：

```python
module.forward(x)
```

---

## 举个简单例子

自己实现一个模块：

```python
class Double(nn.Module):

    def forward(self, x):
        print("forward called")
        return x * 2
```

使用：

```python
m = Double()

m(torch.tensor(3))
```

输出：

```text
forward called

tensor(6)
```

这里：

```python
m(...)
```

自动触发：

```python
m.forward(...)
```

---

## 那么 model(x) 呢？

也是一样。

例如：

```python
model = MLP()

y = model(x)
```

实际上：

```python
model.__call__(x)
```

↓

```python
model.forward(x)
```

↓

```python
self.fc1(x)
```

↓

```python
fc1.__call__(x)
```

↓

```python
fc1.forward(x)
```

整个调用链：

```text
model(x)
│
▼
MLP.forward(x)
│
▼
fc1(x)
│
▼
Linear.forward(x)
│
▼
xW+b
```

---

因此在 PyTorch 中：

```python
model(x)
```

和

```python
layer(x)
```

背后的统一机制都是：

```text
对象(...)
      ↓
__call__()
      ↓
forward()
```

这也是为什么你在 Transformer、GPT 源码里几乎只看到：

```python
x = self.attn(x)

x = self.mlp(x)

x = self.norm(x)
```

而很少看到：

```python
self.attn.forward(x)
```

因为 `()` 本身就是 PyTorch 推荐的 `forward()` 调用方式。
