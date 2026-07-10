"""
01 - 去掉因果 mask 会怎样?(招牌"作弊"bug)

【bug 是什么】
  正确的注意力有一行 causal mask, 挡住"看未来":
      scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
  这里把它删掉, 让每个位置都能看到整句(包括后面的 token)。

【❓ 先别往下看 —— 先预测(写在纸上)】
  1. train loss 会比正确版更高还是更低?为什么?
  2. 训练完拿去生成续写, 会正常吗?
  3. 一句话: 这个 bug 为什么"骗人"?

【怎么跑】
  # 在本目录下运行(用 exercises 的 venv, 那里装了 torch):
  uv run --project ../../../exercises python 01_no_causal_mask.py --epochs 2000 [--plot]

  会同种子、同数据跑「正确 vs buggy」两遍, 并排打印 loss + 各自生成样本。
  验证完你的预测, 再看文件最底部的【✅ 现象解释】。
"""

import torch.nn.functional as F

import shared
from shared import MultiHeadCausalSelfAttention


# ---- 只改这一处: 去掉 mask 的注意力 ----
class NoMaskSelfAttention(MultiHeadCausalSelfAttention):
    def forward(self, x):
        B, T, d = x.shape
        nh, hd = self.n_heads, self.head_dim
        Q, K, V = self.W_qkv(x).chunk(3, dim=-1)
        Q = Q.view(B, T, nh, hd).transpose(1, 2)
        K = K.view(B, T, nh, hd).transpose(1, 2)
        V = V.view(B, T, nh, hd).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (hd ** 0.5)
        # 🐞 正确应为: scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        #    这里故意删掉 → 每个位置都能看到未来 token
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, d)
        return self.W_o(out)


def main():
    args = shared.parse_args()
    data, vocab_size, encode, decode = shared.make_data()

    shared.set_seed()
    m_ok = shared.build_model(vocab_size)                                   # 正确: 有 mask
    ok = shared.train(m_ok, data, epochs=args.epochs)

    shared.set_seed()
    m_bad = shared.build_model(vocab_size, attn_cls=NoMaskSelfAttention)    # 🐞 无 mask
    bad = shared.train(m_bad, data, epochs=args.epochs)

    shared.report(ok, bad, m_ok, m_bad, encode, decode,
                  title="01 去掉因果 mask (作弊看未来)", plot=args.plot)


if __name__ == "__main__":
    main()


# ============================================================
# ✅ 现象解释(答案) —— 验证完预测再看
# ============================================================
# 现象:
#   - buggy 的 train loss 掉得比正确版更快、更低 (假性优秀)。
#   - 但拿去生成续写时, buggy 模型明显更差 / 崩坏。
#
# 为什么"骗人":
#   训练时每个位置能看到"后面的 token", 而它的任务恰恰是预测下一个 token
#   —— 相当于开卷考试偷看答案, 于是 train loss 轻松变低。
#   但真正生成(自回归)时, 未来 token 还不存在, 没得偷看, 于是原形毕露。
#   这就是经典的"信息泄漏 / data leakage": 训练指标好看, 上线就废。
#   → 教训: 训练集/验证集的分布, 必须和"真实推理时能拿到的信息"一致。
