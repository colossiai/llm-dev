"""
04 - 学习率过大会怎样?

【bug 是什么】
  正确 lr = 3e-3。这里把它调到 10.0 (大约 3000 倍)。
      optimizer = AdamW(params, lr=10.0)   # 🐞 步子迈太大

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. loss 会一路下降、原地不动、还是爆炸?
  2. 如果爆炸, 会停在某个高位, 还是一路涨到天上?
  3. 生成会是什么样?

【怎么跑】
  uv run --project ../../../exercises python 04_lr_too_high.py --epochs 2000 [--plot]
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
    bad = shared.train(m_bad, data, epochs=args.epochs, lr=10.0)    # 🐞 lr 过大

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="04 学习率过大 (lr=10.0)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy loss 不降反升, 剧烈震荡并一路爆炸到成百上千 (远高于随机基线 log(vocab)≈3.4)。
#   - 生成是彻底的乱码。
#
# 为什么:
#   梯度下降是"沿反方向迈一步", 步长 ∝ 学习率。步子太大 → 越过最低点冲到对面更高处,
#   下一步反弹更远, 正反馈式发散 → loss 越滚越大。
#
#   小知识(为什么这里没直接变 nan):
#   本例用的是 AdamW, 它会把梯度按二阶矩归一化, 每步位移大致 ∝ lr 而与梯度大小无关,
#   所以即使 lr=10, 也多表现为"loss 爆炸到极大值"而非严格 nan。
#   若换成朴素 SGD、或 lr 再大几个量级, 就会真的冲到 inf/nan。
#   (可自己试: 把 04 改成 lr=50 看 loss 冲到几万; 或想象 SGD 下会溢出成 nan。)
#   → 教训: loss 不降反升 / 变 nan, 第一反应就是"学习率是不是太大 / 要不要加梯度裁剪 warmup"。
#     nanoGPT 里的 grad clipping + LR warmup/cosine 衰减, 正是为了防这个。
