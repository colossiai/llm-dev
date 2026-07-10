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

分两类:**显性 bug**(现象一眼可见)和**隐性 bug**(在这个"背一小段"的玩具任务上现象很弱甚至看不出 —— 这本身是关键一课,见下)。

### 显性 bug(现象明显)

| 文件 | 注入的 bug | 招牌现象(先别看,先猜) |
|---|---|---|
| `01_no_causal_mask.py` | 去掉因果 mask(偷看未来) | train loss 假性极低,但生成崩坏 —— data leakage |
| `02_targets_not_shifted.py` | x/y 不错位(target = input) | loss 秒到 ~0,但生成只会复读(学成"抄写") |
| `03_forgot_zero_grad.py` | 忘记 `optimizer.zero_grad()` | 梯度跨步累积,loss 抖动 / 收敛更差 / 发散 |
| `04_lr_too_high.py` | 学习率过大(lr=10.0) | loss 不降反升,剧烈震荡并爆炸到成百上千 |
| `05_forgot_optimizer_step.py` | 忘记 `optimizer.step()` | 不报错,但 loss 全程平线(参数从没更新) |
| `06_lr_too_low.py` | 学习率过小(lr=1e-6) | loss 缓慢但确实在降(和 05 的"纯噪声平线"对比看趋势) |
| `09_bad_weight_init.py` | 权重初始化过大(std=1.0) | 初始 loss 巨大(~15),训练不稳,收敛差、生成乱码 |

### 隐性 bug(玩具任务上现象很弱 —— 反而是最值得体会的一课)

| 文件 | 注入的 bug | 实测现象(和直觉相反) |
|---|---|---|
| `07_softmax_wrong_dim.py` | softmax 沿错误维度(dim=-2) | train loss 照样降甚至更低(陷阱!),但**生成**明显错乱 |
| `08_forgot_sqrt_scale.py` | 忘记 1/√d 缩放 | head_dim=16 时几乎无差别;head_dim 越大才越致命 |
| `10_no_positional_embedding.py` | 不加位置编码 | loss/生成几乎不受影响;顺序更关键的真实数据上才会崩 |

> **隐性 bug 的共同教训**:小数据 + 过拟合会**掩盖架构 bug** —— train loss 好看 ≠ 架构对。这类 bug 要在更大 head_dim / 更长更难的数据 / 验证集上才暴露。别用"背绕口令"的实验去否定 √d 缩放、位置编码的必要性。

## 文件说明

- `shared.py` —— 唯一的"正确基线":一份没有 bug、能正常训练的 Mini GPT(搬自 06),外加训练循环、数据、生成、对比打印。每个 bug 文件都**只改动其中一处**,让 bug 一眼可见,不被样板淹没。

## 想继续加案例?

`shared.py` 已把常见"正确行为"做成开关或可替换点:`train()` 的 `zero_grad` / `shift_targets` / `do_step` / `lr`,`build_model()` 的 `attn_cls`(替换注意力实现)/ `use_pos_emb`。照现有模板新建 `11_xxx.py` 即可 —— 大多只需再多一个开关、一个注意力子类,或在 main 里对模型做一处改动(参考 09 的 `blow_up_init`)。
