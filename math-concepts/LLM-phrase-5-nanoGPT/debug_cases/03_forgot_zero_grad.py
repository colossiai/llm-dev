"""
03 - 忘记 optimizer.zero_grad() 会怎样?

【bug 是什么】
  PyTorch 的梯度是**累加**的: 每次 loss.backward() 都把新梯度**加到**旧梯度上。
  所以每个 step 开头必须清零:
      optimizer.zero_grad()   # 正确: 每步先清零
      loss.backward()
      optimizer.step()
  这里故意不清零 → 梯度跨 step 不断累积。

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. 梯度一直累加, 相当于 step 的"步长"变大还是变小?
  2. loss 曲线会更平滑还是更抖?会收敛吗?
  3. 和"学习率调大"有什么异同?

【怎么跑】
  uv run --project ../../../exercises python 03_forgot_zero_grad.py --epochs 2000 [--plot]
"""

import shared


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)
    ok = shared.train(m_ok, data, epochs=args.epochs)                    # 正确: 每步清零

    shared.set_seed()
    m_bad = shared.build_model(vocab_size)
    bad = shared.train(m_bad, data, epochs=args.epochs, zero_grad=False) # 🐞 从不清零

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="03 忘记 zero_grad (梯度累积)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy 曲线明显更抖 / 收敛更差, 甚至发散、比正确版差一大截。
#   - 生成质量也随之变差。
#
# 为什么:
#   梯度累积 = 每步用的"更新方向"其实是历史所有步梯度之和, 越滚越大且方向陈旧,
#   等效于一个不断变化、偏大的有效学习率 → 更新过冲、震荡。
#   (Adam 会部分归一化, 所以不一定直接 NaN, 但训练明显不稳、更差。)
#   与"调大学习率"相似(都让更新变大), 但更糟: 累积的是**过时**的梯度方向。
#   → 教训: 这是新手最常见 bug 之一。记住 zero_grad → backward → step 三件套顺序。
