# nanoGPT vs 本地 Transformer 教学脚本 (05 / 06)

对比对象:
- `exercises/transformer/05_mini_gpt.py`
- `exercises/transformer/06_train_and_generate.py`
- [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy)

这三个东西**内核是同一个模型**,区别在于「完整度」和「工程化程度」。先给一句话直觉:

> **05 是骨架图,06 是骨架 + 会动(能训练能生成),nanoGPT 是同一副骨架穿上了工业级装备。**

## 三者关系

| | `05_mini_gpt.py` | `06_train_and_generate.py` | nanoGPT |
|---|---|---|---|
| **定位** | 只搭架构,随机权重跑通形状 | 架构 + 真训练 + 生成 | 能真正复现 GPT-2 的生产级教学库 |
| **模型代码** | `MiniGPT` | 一模一样的 `MiniGPT` + `generate()` | `GPT` 类,结构相同但更全 |
| **训练** | ❌ 没有 | ✅ 有(过拟合一段绕口令) | ✅ 有(OpenWebText 等真数据集) |
| **生成** | 贪心 argmax,展示接口 | multinomial 采样 + temperature | top-k + temperature |
| **代码量** | ~270 行(含大段注释) | ~320 行 | 分 `model.py`/`train.py`/`sample.py` 数百行 |

## 05 vs 06 差在哪

- **05**:纯架构展示。`main()` 里造随机 token → forward → 验证 `logits` 形状对、参数量合理 → 用**随机权重** argmax 生成一段乱码。目的是「看清楚 GPT = 4 个部件」。
- **06**:把 05 的模型**原封不动搬过来**,加了三样东西:
  1. 字符级分词 + `get_batch()` 随机切窗口
  2. 训练循环(cross-entropy + AdamW)
  3. `generate()` 方法(滑窗 + 采样 + temperature)+ checkpoint 保存

两者的 `MiniGPT` 类**字节级几乎一致**(06 只多了 `generate` 方法,05 是 4 层 06 是 3 层)。

## 本地实现 vs nanoGPT 的关键差异

这是最有价值的对比 —— 本地代码是「教学最简版」,nanoGPT 是「能上生产的最简版」:

| 维度 | 本地 MiniGPT | nanoGPT | 为什么 nanoGPT 要多做 |
|---|---|---|---|
| **注意力实现** | 手写 `softmax(QKᵀ/√d)` | `F.scaled_dot_product_attention`(Flash Attention) | 快数倍、省显存 |
| **Dropout** | ❌ 无 | ✅ attention/residual 都有 | 防过拟合(真数据集需要) |
| **权重共享** | `tok_emb` 和 `lm_head` 独立 | **weight tying**(输入输出 embedding 共享) | 省参数、效果更好,GPT-2 原版做法 |
| **权重初始化** | PyTorch 默认 | 特制 init(残差层按 `1/√(2·n_layer)` 缩放) | 深层训练稳定性 |
| **LayerNorm** | `nn.LayerNorm`(带 bias) | 自定义可关 bias 的 LayerNorm | 对齐 GPT-2、略快 |
| **优化器** | 朴素 `AdamW` | 手动分组 weight decay + fused AdamW + 梯度裁剪 + LR warmup/cosine 衰减 | 大规模训练必备 |
| **训练规模** | 单段文本过拟合、CPU 几分钟 | 多 GPU(DDP)、混合精度、`torch.compile`、断点续训 | 真的要炼 GPT-2 |
| **采样** | temperature | temperature + **top-k** | 生成质量更好 |

## 一句话总结

**本地 05/06 = 把 nanoGPT 的 `model.py` 扒掉所有"性能与稳定性优化"后剩下的纯数学骨架;05 只搭不练,06 搭完练一遍。** 把这两个文件读透,再去看 nanoGPT,会发现它多出来的每一行(Flash Attention、weight tying、dropout、LR 调度)都是**为了"从能跑"走到"能炼真模型"**,而不是改变了 GPT 的本质。

## 下一步(从「看懂」到「吃透」)

挑 1-2 个 nanoGPT 的优化动手加到本地 06 里,比如:
- **weight tying**:`self.lm_head.weight = self.tok_emb.weight`
- **top-k 采样**:生成时只在概率最高的 k 个 token 里采样
