"""
09 - 权重初始化过大会怎样?

【bug 是什么】
  正确: PyTorch 默认初始化让每层权重方差很小(Linear 大约 std≈1/√fan_in)。
  这里故意把所有 Linear 权重重新初始化成一个很大的 std:
      nn.init.normal_(linear.weight, std=1.0)   # 🐞 比默认大一个量级以上

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. 初始 loss(第 0 步)会比正常大很多吗?
  2. 激活值/logits 一开始就很大, 会连带影响什么?(想想 softmax、想想梯度)
  3. 训练还稳吗?会不会 nan?

【怎么跑】
  uv run --project ../../../exercises python 09_bad_weight_init.py --epochs 2000 [--plot]
"""

import torch.nn as nn

import shared


# ---- 只改这一处: 训练前把所有 Linear 权重放大重初始化 ----
def blow_up_init(model, std=1.0):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)   # 🐞 默认约 1/√fan_in, 这里 std=1.0
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)                    # 正确: 默认初始化
    ok = shared.train(m_ok, data, epochs=args.epochs)

    shared.set_seed()
    m_bad = blow_up_init(shared.build_model(vocab_size))     # 🐞 放大初始化
    bad = shared.train(m_bad, data, epochs=args.epochs)

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="09 权重初始化过大 (std=1.0)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy 的初始 loss 明显偏高, 训练不稳(抖动大 / 收敛慢 / 更差)。
#   - 生成质量差。
#
# 为什么:
#   权重大 → 每层输出(激活值)被放大 → 层层累积, logits 幅度巨大。
#   两个连锁反应:
#     (1) 巨大 logits 进 softmax → 分布极尖 / 饱和 → 梯度消失(学不动);
#     (2) 前向数值大 → 反向梯度也可能爆炸 → 训练震荡。
#   一句话: 初始化决定了训练的"起跑姿势", 太大太小都会让优化从一开始就走进病态区。
#   → 教训: 这就是为什么各家模型都有精心设计的 init(Xavier/Kaiming, 以及 nanoGPT 里
#     "残差层按 1/√(2·n_layer) 缩放"的特殊 init) —— 不是玄学, 是为了让每层激活方差稳定。
