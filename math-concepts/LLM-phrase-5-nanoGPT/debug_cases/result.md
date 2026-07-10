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

---

# 补充:案例 06–10 实测结果

补齐到 ~10 个案例。以下为 `--epochs 600` 实测(06 为 `--epochs 800`)。**重要发现:07/08/10 的现象和直觉相反 —— 玩具任务的过拟合会掩盖架构 bug。**

## 结果总览

| 案例 | 类型 | buggy 现象(实跑) |
|---|---|---|
| 06 lr 太小(1e-6) | 显性 | loss 从 3.47 缓慢降到 3.36,有明确下降趋势(对比 05 的纯噪声平线) |
| 07 softmax 维度错(dim=-2) | 隐性 | train loss 甚至更低 `0.049` vs `0.088`(陷阱),但生成错乱 |
| 08 忘记 √d 缩放 | 隐性 | head_dim=16 时几乎无差别 `0.088` vs `0.088`;head_dim=64 才变差 `0.168` vs `0.089` |
| 09 权重初始化过大(std=1.0) | 显性 | 初始 loss `15.1`,收敛到 `2.04` vs `0.088`,生成乱码 |
| 10 忘记位置编码 | 隐性 | loss `0.089` vs `0.088`,生成照样正常 —— 玩具任务掩盖了 bug |

## 关键 loss / 生成节选

### 06 lr 太小(--epochs 800)
```
   step |    ✅ 正确 loss |  🐞 buggy loss
      0 |       3.4735 |        3.4735
    399 |       0.0889 |        3.4262
    799 |       0.0689 |        3.3582   ← 缓慢下降(3.47→3.36), 与 05 死平不同
```

### 07 softmax 维度错(dim=-2)
```
  final |       0.0879 |        0.0493   ← buggy train loss 反而更低(陷阱)
生成 'the q' → ✅ 'the quick brown fox...'  /  🐞 'the quorsps the bowxrs jth five...'
```

### 08 忘记 √d 缩放
```
  final |       0.0879 |        0.0876   ← head_dim=16 几乎无差别
探针: n_heads=1(head_dim=64) → correct=0.089  noscale=0.168  ← 大 head_dim 才暴露
生成: correct 与 buggy 几乎一样好
```

### 09 权重初始化过大(std=1.0)
```
   step |    ✅ 正确 loss |  🐞 buggy loss
      0 |       3.4735 |       15.1398   ← 初始就爆高
    599 |       0.0879 |        2.0409   ← 收敛差一大截
生成 'the q' → 🐞 'the q d dioiigiia bixiauicjueec...'(乱码)
```

### 10 忘记位置编码
```
  final |       0.0879 |        0.0890   ← 几乎无差别
生成 'the q' → ✅ 'the quick brown fox...'  /  🐞 'the quick brown fox...'(照样正常)
```

## 补充案例的两处诚实修正(实测推翻了初版答案)

1. **07/10 现象与初版答案相反**:初版写"loss 高、生成乱";实测 buggy 的 **train loss 照样低甚至更低**。已改写答案:07 的问题只暴露在**生成**;10 在这个玩具任务上**根本看不出**。
2. **08 在 head_dim=16 时无现象**:已实测确认 head_dim 增大(=64)才明显变差,答案里给了复现方法(改 n_heads=1)。
3. 三者归纳出一条贯穿的元教训并写进 README:**小数据过拟合会掩盖架构 bug,train loss 好看 ≠ 架构对**。
