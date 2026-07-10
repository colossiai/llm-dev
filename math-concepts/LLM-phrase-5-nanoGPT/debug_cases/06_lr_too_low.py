"""
06 - 学习率过小会怎样?

【bug 是什么】
  正确 lr = 3e-3。这里把它调到 1e-6 (小了 3000 倍)。
      optimizer = AdamW(params, lr=1e-6)   # 🐞 步子迈太小

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. loss 会下降吗?下降得快还是慢?
  2. 同样的 step 数下, 最终 loss 会停在哪个量级?
  3. 这个 bug 和"忘记 step()"(05号)现象像不像?怎么区分?

【怎么跑】
  uv run --project ../../../exercises python 06_lr_too_low.py --epochs 2000 [--plot]
"""

import shared


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)
    ok = shared.train(m_ok, data, epochs=args.epochs)               # 正确: lr=3e-3

    shared.set_seed()
    m_bad = shared.build_model(vocab_size)
    bad = shared.train(m_bad, data, epochs=args.epochs, lr=1e-6)    # 🐞 lr 过小

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="06 学习率过小 (lr=1e-6)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy loss 确实在降, 但极其缓慢, 同样 step 数下还停在很高的位置(远没学完)。
#   - 生成基本还是乱码(没训够)。
#
# 为什么:
#   步长 ∝ 学习率。步子太小 → 每步只挪一点点 → 需要多得多的 step 才能走到谷底。
#   它不是"学不了", 而是"学得起不来", 给足够多 step 最终也能收敛。
#
#   与 05号(忘记 step)的区别 —— 这是最值得练的一点:
#     05号: loss **完全不降**, 在初始值附近纯噪声抖动(参数根本没动)。
#     06号: loss **缓慢但确实在降**(有明确下降趋势, 只是慢)。
#   → 看"趋势"而非"绝对值": 有下降趋势=在学只是慢(调大 lr); 毫无趋势=根本没更新(查 bug)。
