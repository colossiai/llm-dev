"""
07 - softmax 归一化维度错会怎样?

【bug 是什么】
  注意力 scores 形状是 (B, n_heads, T_query, T_key)。
  softmax 必须沿 **key 维 (dim=-1)** 归一化 —— 让"每个 query 对所有 key 的权重和 = 1":
      attn = F.softmax(scores, dim=-1)    # 正确
  这里故意写成 dim=-2 (沿 query 维归一化):
      attn = F.softmax(scores, dim=-2)    # 🐞 归一化错了方向

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. "每个 query 的权重和 = 1"这个性质还成立吗?
  2. 因果 mask 下, 沿 query 维做 softmax 会有什么怪事?(想想第一个 key 那一列)
  3. loss 能正常下降吗?

【怎么跑】
  uv run --project ../../../exercises python 07_softmax_wrong_dim.py --epochs 2000 [--plot]
"""

import torch.nn.functional as F

import shared
from shared import MultiHeadCausalSelfAttention


# ---- 只改这一处: softmax 沿错误维度 ----
class WrongSoftmaxDimAttention(MultiHeadCausalSelfAttention):
    def forward(self, x):
        B, T, d = x.shape
        nh, hd = self.n_heads, self.head_dim
        Q, K, V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, nh, hd).transpose(1, 2)
        K = K.view(B, T, nh, hd).transpose(1, 2)
        V = V.view(B, T, nh, hd).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (hd ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        attn = F.softmax(scores, dim=-2)   # 🐞 正确应为 dim=-1 (沿 key 维)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(out)


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)                                     # 正确: dim=-1
    ok = shared.train(m_ok, data, epochs=args.epochs)

    shared.set_seed()
    m_bad = shared.build_model(vocab_size, attn_cls=WrongSoftmaxDimAttention) # 🐞 dim=-2
    bad = shared.train(m_bad, data, epochs=args.epochs)

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="07 softmax 维度错 (dim=-2)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象(实测, 可能和你预测的相反):
#   - buggy 的 train loss 照样能降, 甚至比正确版**还低**(如 0.049 vs 0.088)。
#   - 但生成明显退化: 词都拼错位, 如 'the quorsps the bowxrs' (正确是 'the quick brown fox')。
#   - 没有 nan(见下)。
#
# 为什么 train loss 反而更低 —— 这是个陷阱:
#   本任务是"背下一小段文本"的过拟合任务, 模型容量足够。即使注意力语义错了,
#   它仍能靠死记硬背把**训练窗口**拟合得很好, 于是 train loss 照样低。
#   → 教训①: train loss 低 ≠ 模型对。真正的毛病要在"泛化行为"(这里是自回归生成)上才暴露。
#
# 为什么语义是错的:
#   softmax 要"把一组数变成和为 1 的分布"。注意力要的是"每个 query 对所有 key 的权重和=1"
#   → 必须沿 key 维(dim=-1)。写成 dim=-2 变成"沿 query 维归一化", attn @ V 不再是
#   "对 value 的加权平均", 每个 query 拿到一堆没意义的系数 → 生成时错误逐步累积 → 崩。
#   → 教训②: softmax 的 dim 必须对准"你想让谁的和为 1"。注意力永远沿 key 维(最后一维)。
#
# (关于 nan: 有人担心因果 mask 下沿 query 维 softmax 会遇到"整列 -inf"→nan。实际不会——
#  下三角 mask 保证每一列至少有对角线那个元素是有限值, 所以不会 nan, 只是语义错。)
