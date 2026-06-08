**MoE = Mixture of Experts（专家混合模型）**

它是当前 GPT-4 之后许多大型模型采用的一种架构思想：

> **不是让所有参数都工作，而是每次只调用少数几个“专家”来回答问题。**

---

## 先看传统 Transformer（Dense Model）

例如 GPT-3：

```text
Question
   │
   ▼

┌─────────────────┐
│   175B Params   │
└─────────────────┘

   │
   ▼

Answer
```

无论你问：

* 数学
* 编程
* 中文
* 英文
* 法律

全部 175B 参数都参与计算。

优点：

* 简单

缺点：

* 非常贵
* 推理速度慢
* 参数利用率低

---

## MoE 的想法

现实世界不会让所有专家同时工作：

```text
医院

病人感冒
   │
   ▼

内科医生

而不是：

内科 + 外科 + 眼科 + 牙科
一起看
```

MoE 模型模仿这种机制。

---

## MoE 架构

```text
                 Input
                   │
                   ▼

               Router
            (路由器)

        ┌──────┼──────┐
        ▼      ▼      ▼

    Expert1 Expert2 Expert3
      数学     代码     中文

        ▼      ▼      ▼

        └──────┼──────┘
               ▼

             Output
```

---

## Router 做什么？

Router 是一个小神经网络。

它会决定：

```text
"2+2=?"
```

应该找谁：

```text
数学专家
```

---

```text
"写一个Python排序"
```

应该找谁：

```text
代码专家
```

---

```text
"翻译成中文"
```

应该找谁：

```text
语言专家
```

---

## Top-K Routing

实际上不会只选一个专家。

通常：

```text
64 个专家

选择最相关的 2 个
```

例如：

```text
Question:
解释 PyTorch 反向传播
```

Router：

```text
代码专家      0.92
数学专家      0.88
法律专家      0.01
医学专家      0.00
```

选：

```text
代码专家
+
数学专家
```

共同处理。

---

## 为什么省钱？

假设：

```text
64 个专家

每个专家 25B 参数
```

总参数：

```text
64 × 25B

=
1.6T 参数
```

但每次只激活：

```text
2 个专家

=
50B 参数
```

所以：

```text
总参数：
1600B

实际计算：
50B
```

这就是 MoE 的魔法。

---

## 一个简单数字例子

Dense 模型：

```text
总参数 = 500B

每次计算 = 500B
```

MoE：

```text
20 个专家

每个 50B

总参数：

20 × 50B
=
1000B

每次激活：

2 × 50B
=
100B
```

结果：

```text
参数翻倍

成本只增加一点点
```

---

## 为什么效果更好？

因为不同专家会逐渐学会不同领域。

例如：

```text
Expert 1
数学

Expert 2
代码

Expert 3
推理

Expert 4
中文

Expert 5
英文
```

虽然训练时没有人手动规定这些角色，但训练过程中会自然形成分工。

这称为：

> Emergent Specialization（涌现式专业化）

---

## DeepSeek 的例子

公开资料中：

```text
DeepSeek V4-Pro

总参数:
1.6T

激活参数:
49B
```

意味着：

```text
1,600B 参数存在

每个 token

只用约 49B 参数
```

所以推理成本远低于一个真正的 1.6T Dense 模型。

---

## 用代码理解 MoE

普通 MLP：

```python
y = mlp(x)
```

MoE：

```python
expert1 = mlp1(x)
expert2 = mlp2(x)
expert3 = mlp3(x)

scores = router(x)

y = (
    scores[0] * expert1 +
    scores[1] * expert2 +
    scores[2] * expert3
)
```

实际实现会使用 Top-K：

```python
只计算得分最高的2个专家
```

其余专家完全跳过。

---

## MoE 的核心思想（一句话）

```text
Dense:
所有专家一起干活

MoE:
先找最合适的专家，
然后只让少数专家干活
```

可以把它理解成：

```text
GPT-3
=
一个什么都会的大专家

MoE
=
一个专家团队
+
一个调度员（Router）
```

这也是为什么现代模型的讨论中，经常会看到：

```text
Total Parameters（总参数）
≠
Active Parameters（激活参数）
```

对于 MoE 模型，更重要的往往是 **Active Parameters**，因为它更接近实际推理成本。
