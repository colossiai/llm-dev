# 计划:经典训练 bug 调试案例集(预测→验证)

## Context(为什么做这个)

用户已经读懂 `exercises/transformer/05_mini_gpt.py`(纯架构)和 `06_train_and_generate.py`(可训练 + 生成),下一步想通过「**故意注入经典训练 bug → 先预测现象 → 跑验证**」来建立真正的训练调试直觉(主动学习法)。

关键修正(已与用户对齐方向):
- **不注入 05**。05 没有训练循环,大多数训练 bug 的现象是 **loss 曲线抽风 / 生成崩坏**,必须有可训练模型才能"看见"。载体应是 06 那样的可训练 MiniGPT。
- **不污染 05/06 参考实现**。它们是用户以后反复回看的标准版。
- 用户指定:**每个 bug 一个文件**,放在 `math-concepts/LLM-phrase-5-nanoGPT/debug_cases/`。

预期产出:一个自成体系的调试练习目录,每个案例都遵循「先预测、后验证、并排对比正确 vs buggy」的节奏。

## 目录结构

```
math-concepts/LLM-phrase-5-nanoGPT/debug_cases/
  README.md                 # 方法说明 + 案例索引 + 运行方式
  shared.py                 # 唯一"正确基线":MiniGPT + 正确训练循环 + generate + 字符级数据
  01_no_causal_mask.py      # 去掉因果 mask
  02_targets_not_shifted.py # x/y 不错位(target = input)
  03_forgot_zero_grad.py    # 忘记 optimizer.zero_grad()
  04_lr_too_high.py         # 学习率过大(NaN 爆炸)
  05_no_grad_disables_training.py  # no_grad/detach 包住 loss,参数不动
```

## `shared.py`(复用 06,是所有案例的"正确参照系")

从 `exercises/transformer/06_train_and_generate.py` 移植(结构完全一致,保持用户已熟悉的代码):
- `MiniGPT`(含 `MultiHeadCausalSelfAttention` / `FeedForward` / `TransformerBlock` / `generate`)——与 06 逐行一致。
- `make_data()`:返回 `data, vocab_size, encode, decode`,用 06 同一段绕口令文本 + 字符级分词。
- `train(model, data, *, epochs, lr, batch_size, max_seq_len, zero_grad=True, shift_targets=True, wrap_no_grad=False)`:**正确**训练循环,返回 `losses`。把几个"可被 bug 关掉"的正确行为做成参数开关,这样每个 bug 文件只需翻转一个开关或传入变体,**diff 一目了然**。
- `set_seed(seed)`:固定随机种子,保证 correct 与 buggy 跑在同一批数据上,曲线可比。

> 注:仓库惯例偏好"脚本自包含"。这里因为**目的就是隔离单个 bug**,若每个文件内联整份模型,bug 会淹没在样板里,故改用共享 `shared.py`——每个 bug 文件因此只聚焦那一处改动。

## 每个 bug 文件的统一结构(以 `01_no_causal_mask.py` 为例)

```
"""
01 - 去掉因果 mask 会怎样?

【bug 是什么】
  注意力里删掉 masked_fill(...causal...),让每个位置能看到未来 token。

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. train loss 会更高还是更低?为什么?
  2. 训练完拿去生成续写,会正常吗?
  3. 一句话:这个 bug 为什么"骗人"?

【怎么跑】
  uv run python 01_no_causal_mask.py --epochs 2000 [--plot]
  会同种子同数据跑「正确 vs buggy」两遍,并排打印 loss + 各自生成样本。

（现象解释在文件最底部,验证完你的预测再看。）
"""
```
- 导入 `shared` 的正确 `MiniGPT` / `make_data` / `train` / `set_seed`。
- 定义 buggy 变体:**只改那一处**,旁边注释 `# 正确应为: ...`。
  - 01:子类化注意力,`forward` 里去掉 `masked_fill` 那一行。
  - 02:调 `train(..., shift_targets=False)`。
  - 03:调 `train(..., zero_grad=False)`。
  - 04:调 `train(..., lr=1.0)`(对照 3e-3)。
  - 05:调 `train(..., wrap_no_grad=True)`(forward+loss 包在 `torch.no_grad()` 里)。
- `main()`:`set_seed` → 跑 correct → `set_seed`(同种子)→ 跑 buggy → **并排打印**两条 loss 进度(几个采样步)+ 两个模型对同一 prompt 的生成 → 若 `--plot` 用 matplotlib 画两条曲线同图(可选,缺 matplotlib 时降级为纯文本表)。
- 文件底部 `# ===== ✅ 现象解释(答案) =====` 注释块:讲清 buggy 为什么这样(如 01:训练时偷看未来→train loss 假性极低,但自回归生成时没有未来可看→崩坏/复读)。

## 5 个案例 + 预期现象

| 文件 | 注入的 bug | 预测/真实现象 |
|---|---|---|
| 01 | 去掉因果 mask | train loss 假性极低(偷看未来),但生成崩坏——招牌"作弊"bug |
| 02 | x/y 不错位(target=input) | loss 极快→~0,但生成只会复读当前字符(学成"抄输入") |
| 03 | 忘记 zero_grad | 梯度跨步累积,loss 抖动/收敛变差或发散 |
| 04 | LR=1.0 过大 | loss 震荡并爆成 NaN/inf |
| 05 | no_grad 包住 loss | 参数不更新,loss 全程近似平线(不下降) |

## `README.md`

- 讲「预测→验证」学习法为何有效(对应用户已建立的直觉优先风格)。
- 案例索引表(同上)。
- 强调:先只读每个文件顶部 docstring 的"❓先预测",跑完再看底部"✅答案"。
- 运行:`cd debug_cases && uv run python 0X_....py --epochs 2000`。

## 验证方式(实现后如何自证)

逐个运行,确认现象与上表一致:
```
cd math-concepts/LLM-phrase-5-nanoGPT/debug_cases
uv run python 01_no_causal_mask.py --epochs 1500      # buggy train loss 明显更低但生成崩
uv run python 02_targets_not_shifted.py --epochs 1500 # buggy loss 秒到 ~0,生成复读
uv run python 03_forgot_zero_grad.py --epochs 1500    # buggy 曲线抖/差
uv run python 04_lr_too_high.py --epochs 1500         # buggy 出 NaN
uv run python 05_no_grad_disables_training.py --epochs 1500  # buggy loss 平线
```
每个脚本自身已并排打印 correct vs buggy,肉眼即可核对预期现象。

## 备注 / 待用户确认点

- Bug 清单默认取"招牌 5 个"(现象最可视、最适合预测)。用户此前未在清单粒度上拍板;如需增删(如加 LR 太小 / softmax 维度错 / 忘记 √d 缩放)可在批准时说明。
- 采用共享 `shared.py` 而非每文件内联整份模型(理由见上)。如坚持完全自包含,可改为每文件复制 06 模型。
