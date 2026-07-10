"""
10 - 忘记位置编码会怎样?

【bug 是什么】
  正确: token embedding 之外还要加位置编码, 模型才知道"谁在前谁在后":
      x = tok + pos          # 正确
  这里故意不加:
      x = tok                # 🐞 只有"是什么 token", 没有"在第几位"

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. 没有位置信息, 自注意力还能区分 "abc" 和 "cba" 吗?
  2. 对这段有强顺序结构的文本(the quick brown ...), loss 会受多大影响?
  3. 生成出来的东西会有什么特点?

【怎么跑】
  uv run --project ../../../exercises python 10_no_positional_embedding.py --epochs 2000 [--plot]
"""

import shared


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)                        # 正确: tok + pos
    ok = shared.train(m_ok, data, epochs=args.epochs)

    shared.set_seed()
    m_bad = shared.build_model(vocab_size, use_pos_emb=False)    # 🐞 只有 tok
    bad = shared.train(m_bad, data, epochs=args.epochs)

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="10 忘记位置编码", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象(实测, 可能和你预测的相反):
#   - 在本玩具任务上, buggy 和正确版几乎**没差别**: loss ~0.089 vs 0.088, 生成也照样
#     吐出 'the quick brown fox jumps over the lazy dog.'。你可能预测"崩", 结果没崩。
#
# 为什么没崩(反直觉但重要):
#   理论上自注意力是"置换等变"的——只看 token 两两关系, 不看谁在第几位, 眼里 "abc" 和 "cba"
#   是同一袋 token。所以位置编码确实是必需的"顺序信息"来源。
#   但**本任务太简单**: 语料就那几句、字符级、局部几乎唯一决定下一个字符(prefix "the q" 后
#   基本只可能是 "u")。causal 注意力按左到右处理 + token 本身, 已足够"背"下来, 不太需要位置。
#   → 于是这个 bug 在这里被过拟合掩盖了, 看不出损失。
#
#   → 教训①: Transformer = 注意力(管"关系") + 位置编码(管"顺序")。GPT-2 用可学习 pos_emb,
#     LLaMA 用 RoPE, 都在补"顺序信息"。真实文本(长、需区分'第几次出现同一词')上, 去掉它必崩。
#   → 教训②(和 07/08 同一课): 玩具任务上"没现象"不代表架构对。位置编码的价值要在更长、
#     顺序更关键的数据上才凸显 —— 别用一个背绕口令的实验去否定它的必要性。
