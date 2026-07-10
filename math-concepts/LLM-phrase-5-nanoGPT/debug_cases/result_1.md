# 实测验证结果(result_1)

运行环境:`exercises/.venv`(torch 2.2.2),每个案例 `--epochs 800`,同种子同数据跑「正确 vs buggy」两遍。

运行方式:

```bash
cd math-concepts/LLM-phrase-5-nanoGPT/debug_cases
uv run --project ../../../exercises python 0X_....py --epochs 800
```

## 结果总览(都符合预测)

| 案例 | buggy 现象(实跑) |
|---|---|
| 01 去 mask | loss 假性更低 `0.0003` vs `0.069`,但生成崩成 `'the qumps..equthe'` —— 偷看未来 |
| 02 不错位 | loss 秒到 `0.0003`,生成复读 `'qqqqqq'` / `'vvvvv'` —— 学成抄写 |
| 03 忘 zero_grad | loss 发散涨到 `26.9`,生成乱码 —— 梯度累积 |
| 04 lr 过大 | loss 爆炸到 `751→1727`,生成全乱 —— 发散 |
| 05 忘 step | loss 死平在 `3.46`(初始噪声),不报错 —— 参数从没更新 |

## 各案例 loss 对照(节选)

### 01 去掉因果 mask
```
   step |    ✅ 正确 loss |  🐞 buggy loss
      0 |       3.4735 |        3.4727
    159 |       0.1038 |        0.0245
    399 |       0.0889 |        0.0038
    799 |       0.0689 |        0.0003   ← buggy 假性更低
```
生成对比:
```
prompt 'the q'
  ✅ 正确 : 'the quick brown fox jumps over the lazy dog. '
  🐞 buggy: 'the qumps..equthe. qumps.etherds. pazy. pack '
```

### 02 target 不错位
```
   step |    ✅ 正确 loss |  🐞 buggy loss
    239 |       0.0789 |        0.0017
    799 |       0.0689 |        0.0003   ← 秒到 ~0
```
生成对比:
```
prompt 'the q' → 🐞 buggy: 'the qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq'
prompt 'how v' → 🐞 buggy: 'how vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv'
```

### 03 忘记 zero_grad
```
   step |    ✅ 正确 loss |  🐞 buggy loss
    239 |       0.0789 |        5.6246
    559 |       0.1044 |       14.8757
    799 |       0.0689 |       26.8873   ← 发散
```
生成:🐞 buggy 全是 `'hhhhhhhh...'` 乱码。

### 04 学习率过大(lr=10.0)
```
   step |    ✅ 正确 loss |  🐞 buggy loss
      0 |       3.4735 |        3.4735
     79 |       0.2033 |      751.6068
    159 |       0.1038 |     1128.0392
    239 |       0.0789 |     1727.8389   ← 爆炸到成百上千
```
生成:🐞 buggy 彻底乱码。
注:AdamW 归一化梯度,`lr=1.0` 甚至 `50` 都不会真的 nan,只会 loss 冲到极大;换 SGD 或更极端 lr 才会 inf/nan。

### 05 忘记 optimizer.step()
```
   step |    ✅ 正确 loss |  🐞 buggy loss
    239 |       0.0789 |        3.4342
    559 |       0.1044 |        3.4728
    799 |       0.0689 |        3.4656   ← 死平在初始噪声附近, 不报错
```
生成:🐞 buggy 和随机权重一样乱(参数从没更新)。

## 与原计划的两处调整

1. **案例 04:`lr` 1.0 → 10.0**。实测 AdamW 归一化梯度,低 lr 不会 NaN;改用诚实的「loss 爆炸/发散」现象,并在答案里补了「Adam 为何通常不直接 nan」的小知识。
2. **案例 05 用「忘记 `optimizer.step()`」而非「no_grad 包住 loss」**。后者会让 `backward()` 直接 RuntimeError(崩溃),给不出「loss 平线」;忘记 step 才能干净演示「不报错但参数冻结」。no_grad/detach 作为同族变体在 docstring 里点到。
