# 什么是RLHF


**RLHF = Reinforcement Learning from Human Feedback（基于人类反馈的强化学习）**

它解决的问题是：

> **预训练（Pretraining）让模型学会语言，但不会让模型变得“有用、听话、符合人类偏好”。**

---

## 为什么需要 RLHF？

假设模型只做预训练。

训练目标是：

```text
预测下一个词
```

例如：

```text
用户：
如何煮鸡蛋？

模型：
在历史上，鸡蛋是一种常见食品...
```

这可能是语法正确的，但不是用户真正想要的回答。

人类更希望：

```text
1. 烧开水
2. 放入鸡蛋
3. 煮7分钟
4. 捞出冷却
```

所以需要让模型学会：

```text
什么回答更受人类喜欢
```

---

## RLHF 三步走

### 第一步：预训练（Pretraining）

就是 GPT 最初的训练。

```text
互联网文本
      ↓
Transformer
      ↓
预测下一个 Token
```

得到 Base Model。

例如：

```text
GPT-3 Base
```

此时模型知识很多。

但不一定听指令。

---

### 第二步：监督微调（SFT）

SFT = Supervised Fine-Tuning

人工写很多高质量问答。

例如：

```text
Q:
法国首都？

A:
巴黎。
```

```text
Q:
写一个Python排序程序

A:
def sort(...)
```

训练模型模仿这些答案。

```text
人工示范
      ↓
模型学习
```

得到：

```text
Instruct Model
```

---

### 第三步：RLHF

这是最有趣的一步。

---

## 人类给答案打分

同一个问题：

```text
解释神经网络
```

模型生成两个回答：

### 回答A

```text
神经网络是一种机器学习模型...
```

### 回答B

```text
把神经网络想象成很多个小计算器连接在一起...
```

标注员选择：

```text
B 更好
```

于是得到：

```text
B > A
```

---

再来：

```text
问题2
```

得到：

```text
C > D
```

---

收集大量数据：

```text
A > B
C > D
E > F
...
```

---

## 训练 Reward Model

奖励模型（Reward Model）学习：

```text
人类喜欢什么答案
```

输入：

```text
问题 + 回答
```

输出：

```text
分数
```

例如：

```text
解释神经网络

回答A

Reward = 0.3
```

```text
解释神经网络

回答B

Reward = 0.9
```

奖励模型学会了模拟人类偏好。

---

## 强化学习优化

现在：

```text
LLM
   ↓
生成回答
   ↓
Reward Model打分
   ↓
强化学习更新参数
```

目标：

```text
获得更高奖励
```

类似训练狗狗：

```text
动作
 ↓
奖励
 ↓
下次更可能重复
```

模型也是：

```text
回答
 ↓
高分
 ↓
以后更倾向这样回答
```

---

## 图解整个流程

```text
           海量文本
               │
               ▼

         Pretraining
               │

               ▼

          Base Model
               │

               ▼

     人工示范问答(SFT)
               │

               ▼

       Instruction Model
               │

               ▼

    人类比较答案好坏
               │

        A > B
        C > D
        E > F

               ▼

        Reward Model
               │

               ▼

     Reinforcement Learning
               │

               ▼

         ChatGPT
```

---

## 为什么 ChatGPT 突然变好了？

很多人以为突破来自参数量。

实际上 2022 年 ChatGPT 爆发的重要原因之一是：

```text
GPT-3
+
Instruction Tuning
+
RLHF
```

相比 GPT-3 Base：

```text
知识量
≈ 差不多

有用程度
↑↑↑↑↑
```

用户第一次感觉：

```text
它真的在和我交流
```

---

## 一个直觉例子

假设你训练一个学生。

### 传统预训练

给他看：

```text
100万本书
```

他知识很多。

---

### RLHF

每次回答后老师评价：

```text
这个回答很好
+1
```

```text
这个回答太啰嗦
-1
```

```text
这个回答不安全
-5
```

久而久之：

```text
学生开始知道

什么答案
最符合老师期待
```

RLHF 本质上就是这个过程。

---

## 2026 年的发展

现代模型已经不只是 RLHF。

通常是：

```text
Pretraining
    ↓
SFT
    ↓
RLHF
    ↓
Reasoning Training
    ↓
Tool Training
    ↓
Agent Training
```

例如：

* OpenAI GPT-5.x
* Anthropic Claude 4.x
* DeepSeek DeepSeek-R1
* Alibaba Qwen3

都已经在 RLHF 基础上叠加了更复杂的推理训练（Reasoning Training）。

所以可以把 RLHF 理解为：

> **让模型从“会说话”进化到“更符合人类偏好地说话”的关键一步。**
