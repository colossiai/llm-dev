"""
08 - 忘记 1/√d 缩放会怎样?

【bug 是什么】
  注意力打分要除以 √head_dim 再进 softmax:
      scores = Q @ K.transpose(-2, -1) / (hd ** 0.5)   # 正确: 缩放
  这里故意去掉缩放:
      scores = Q @ K.transpose(-2, -1)                 # 🐞 不缩放

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. 不缩放时, scores 的数值会偏大还是偏小?(想想 d 个数相乘再相加)
  2. scores 偏大, 进 softmax 后权重分布会变"尖"还是变"平"?
  3. 分布变尖对梯度、对收敛速度有什么影响?

【怎么跑】
  uv run --project ../../../exercises python 08_forgot_sqrt_scale.py --epochs 2000 [--plot]
"""

import torch.nn.functional as F

import shared
from shared import MultiHeadCausalSelfAttention


# ---- 只改这一处: 去掉 1/√d 缩放 ----
class NoScaleAttention(MultiHeadCausalSelfAttention):
    def forward(self, x):
        B, T, d = x.shape
        nh, hd = self.n_heads, self.head_dim
        Q, K, V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, nh, hd).transpose(1, 2)
        K = K.view(B, T, nh, hd).transpose(1, 2)
        V = V.view(B, T, nh, hd).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1)   # 🐞 正确应为 ... / (hd ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(out)


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)                              # 正确: 有缩放
    ok = shared.train(m_ok, data, epochs=args.epochs)

    shared.set_seed()
    m_bad = shared.build_model(vocab_size, attn_cls=NoScaleAttention)  # 🐞 无缩放
    bad = shared.train(m_bad, data, epochs=args.epochs)

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="08 忘记 1/√d 缩放", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象(实测, 可能和你预测的相反):
#   - 本例配置下(head_dim=16), buggy 和正确版几乎**没有差别**(final loss ~0.088 vs 0.088,
#     生成也几乎一样好)。你可能预测"明显变差", 结果却看不出来。
#   - 这不是说缩放不重要, 而是——head_dim 太小 + 过拟合小任务, 把这个 bug 掩盖了。
#     实测把 head_dim 从 16 加到 64(n_heads 4→1), 不缩放的 loss 就明显更差(~0.17 vs 0.09)。
#
# 为什么理论上必须缩放:
#   Q·K 是 head_dim 个乘积之和; head_dim 越大, 点积方差越大(∝ head_dim)。
#   不缩放 → scores 偏大 → softmax 被推向"接近 one-hot 的尖锐分布" → 进入饱和区, 梯度趋近 0
#   → 学得慢、易卡。除以 √head_dim 把点积方差拉回 ~1, 让 softmax 处于"梯度健康"的区间。
#   这就是 Transformer 论文 "Scaled Dot-Product Attention" 里 "Scaled" 的由来。
#
#   → 教训: head_dim 越大, 忘记缩放越致命; head_dim 小(本例 16)时几乎无感。
#     更重要的一课: **玩具任务上看不出的 bug ≠ 没 bug**。要在真实规模(大 head_dim、大数据)下才暴露。
#     想亲眼验证: 把本文件里两个 build_model 的模型改成 n_heads=1 再跑一遍。
