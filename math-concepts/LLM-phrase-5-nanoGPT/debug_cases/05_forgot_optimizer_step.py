"""
05 - 忘记 optimizer.step() 会怎样?(参数永不更新)

【bug 是什么】
  三件套 zero_grad → backward → step 里, step() 才是"真正修改权重"的一步。
  这里故意漏掉它:
      optimizer.zero_grad()
      loss.backward()         # 算出了梯度
      # optimizer.step()      # 🐞 漏掉 → 梯度算了却没拿去更新参数
  (同族经典变体: 用 torch.no_grad() 把 forward+loss 包起来 → backward 直接报错;
   或对 loss 做 .detach() → 断了梯度。三者本质都是"参数动不了"。)

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. loss 曲线会长什么样?
  2. 会报错吗?还是安静地"假装在训练"?
  3. 你怎么在 5 秒内一眼看出"训练根本没生效"?

【怎么跑】
  uv run --project ../../../exercises python 05_forgot_optimizer_step.py --epochs 2000 [--plot]
"""

import shared


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)
    ok = shared.train(m_ok, data, epochs=args.epochs)                  # 正确: 有 step()

    shared.set_seed()
    m_bad = shared.build_model(vocab_size)
    bad = shared.train(m_bad, data, epochs=args.epochs, do_step=False) # 🐞 漏掉 step()

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="05 忘记 optimizer.step() (参数不动)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy loss 全程近似一条水平线, 在初始值附近上下随机抖动, 从不下降。
#   - 不报错(最阴险的一点): 程序照常跑完, 看起来"在训练"。
#   - 生成永远是随机权重的乱码(和没训练一样)。
#
# 为什么:
#   backward() 只负责把梯度算出来存进 p.grad, 真正"沿梯度改权重"的是 step()。
#   漏掉 step(), 权重从头到尾没变过 → 每步只是换了批随机数据算一遍 loss,
#   于是 loss 在初始水平上随机波动, 均值不降。
#   一眼识别法: loss 不降且"抖动幅度像噪声" → 先怀疑参数根本没更新
#   (漏 step / no_grad / detach / requires_grad=False / 优化器没拿到这些参数)。
#   → 教训: "不报错" ≠ "在学习"。永远盯着 loss 趋势, 而不是"程序有没有跑完"。
