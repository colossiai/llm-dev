"""
02 - target 不错位会怎样?(x/y 没有偏移 1)

【bug 是什么】
  LLM 训练范式: 每个位置预测"下一个"token, 所以 target 要相对 input 右移 1 位:
      input:  [t0, t1, t2, t3]
      target: [t1, t2, t3, t4]   ← 正确, 错位 1
  这里故意让 target = input (不错位):
      target: [t0, t1, t2, t3]   ← 🐞 模型只需"抄写当前 token"

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. train loss 会怎样?(收敛快慢? 最终值?)
  2. 生成续写会长什么样?
  3. 一句话: 模型到底学会了什么"本事"?

【怎么跑】
  uv run --project ../../../exercises python 02_targets_not_shifted.py --epochs 2000 [--plot]
"""

import shared


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)
    ok = shared.train(m_ok, data, epochs=args.epochs)                        # 正确: 错位

    shared.set_seed()
    m_bad = shared.build_model(vocab_size)
    bad = shared.train(m_bad, data, epochs=args.epochs, shift_targets=False) # 🐞 不错位

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="02 target 不错位 (学成抄写)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy 的 train loss 极快掉到接近 0 (比正确版还快还低)。
#   - 但生成续写基本是"复读": 反复吐同一个字符 / 原样重复输入的最后一个 token。
#
# 为什么:
#   因果注意力里, 位置 t 是能看到"自己"这个 token 的(mask 含对角线)。
#   当 target=input 时, 位置 t 的任务变成"预测 t 位置的 token"——而它明明就看得见,
#   于是模型只要学会"把当前 token 原样输出"就满分, loss 秒到 0。
#   它压根没学"语言的下一步", 只学了"复制"。生成时自然只会复读。
#   → 教训: loss 掉得又快又低不一定是好事, 要警惕"任务被简化成了平凡解"。
