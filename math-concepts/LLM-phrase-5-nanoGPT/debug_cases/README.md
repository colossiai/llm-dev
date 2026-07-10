# 经典训练 bug 调试案例集(预测 → 验证)

## 这是干嘛的

读懂了 `exercises/transformer/05_mini_gpt.py`(纯架构)和 `06_train_and_generate.py`(可训练 + 生成)之后,真正建立**训练调试直觉**的最快方式,不是再看一遍正确代码,而是:

> **故意注入一个经典 bug → 先在纸上预测现象 → 跑代码验证你猜得对不对。**

猜错的地方,就是你对训练机制理解的漏洞 —— 一次填一个。这套东西专门给你练这个。

## 用法(重要:先猜后看)

每个案例是一个独立文件,结构统一:

1. **文件顶部 docstring** 有【bug 是什么】+【❓ 先别往下看 —— 先预测】。
   → 先只读这部分,把三个预测问题的答案**写在纸上**。
2. **跑脚本**,它会用同种子、同数据跑「✅ 正确 vs 🐞 buggy」两遍,并排打印 loss 进度和生成样本。
3. **对照你的预测**。
4. 最后再看**文件底部的【✅ 现象解释】**,核对原理。

## 怎么运行

这些脚本要用 `torch`,复用 `exercises` 那个已经装好 torch 的虚拟环境。在**本目录下**运行:

```bash
uv run --project ../../../exercises python 01_no_causal_mask.py --epochs 2000
```

加 `--plot` 可用 matplotlib 画出 correct vs buggy 两条 loss 曲线同图(可选;缺 matplotlib 会自动降级为纯文本表)。`--epochs` 越大现象越明显,想快看用 `--epochs 1000` 也够。

## 案例索引

| 文件 | 注入的 bug | 招牌现象(先别看,先猜) |
|---|---|---|
| `01_no_causal_mask.py` | 去掉因果 mask(偷看未来) | train loss 假性极低,但生成崩坏 —— data leakage |
| `02_targets_not_shifted.py` | x/y 不错位(target = input) | loss 秒到 ~0,但生成只会复读(学成"抄写") |
| `03_forgot_zero_grad.py` | 忘记 `optimizer.zero_grad()` | 梯度跨步累积,loss 抖动 / 收敛更差 |
| `04_lr_too_high.py` | 学习率过大(lr=10.0) | loss 不降反升,剧烈震荡并爆炸到成百上千 |
| `05_forgot_optimizer_step.py` | 忘记 `optimizer.step()` | 不报错,但 loss 全程平线(参数从没更新) |

## 文件说明

- `shared.py` —— 唯一的"正确基线":一份没有 bug、能正常训练的 Mini GPT(搬自 06),外加训练循环、数据、生成、对比打印。每个 bug 文件都**只改动其中一处**,让 bug 一眼可见,不被样板淹没。

## 想继续加案例?

`shared.train()` 已经把几个"正确行为"做成开关(`zero_grad` / `shift_targets` / `do_step` / `lr`)。想加新 bug(如 LR 太小、softmax 维度错、忘记 √d 缩放、权重初始化过大),照 01~05 的模板新建 `06_xxx.py` 即可 —— 大多只需再多一个开关或一个子类。
