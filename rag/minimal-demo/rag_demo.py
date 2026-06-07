"""
最小化 RAG (Retrieval-Augmented Generation) Demo

================ 给零基础读者的 5 分钟讲解 ================

【为什么需要 RAG?】
  LLM (GPT/Qwen/DeepSeek) 训练数据有截止日期, 不知道:
    - 公司内部研报、合同、合规文档
    - 最新政策、新闻、财报
  且 LLM 容易"幻觉"(瞎编看似合理但错误的答案)。

  → RAG 给 LLM 一本"参考书", 让它基于参考书回答, 不再瞎编。

【RAG 的核心思路 (4 步)】
  1. Chunking    把文档切成小块
  2. Embedding   把每块转成向量
  3. Retrieval   用户问问题时, 找出最相关的几块
  4. Generation  把这几块作为"上下文"喂给 LLM, 让它基于上下文回答

【两种 embedding 模式】
  默认 (TF-IDF):    用 sklearn, 零下载, 立即跑, 适合 demo
  --neural:         用 sentence-transformers, 效果更好, 需下载 ~480MB 模型

【运行】
  uv sync
  uv run python rag_demo.py                  # TF-IDF, 不调 LLM
  uv run python rag_demo.py --neural         # 神经 embedding, 不调 LLM
  uv run python rag_demo.py --llm            # TF-IDF + LLM (需 .env)
  uv run python rag_demo.py --neural --llm   # 神经 embedding + LLM
"""

import argparse
import os
from pathlib import Path

import numpy as np


# =============================================================
# 1. 模拟"公司内部文档" (LLM 不知道这些 — 都是 2026 年的)
# =============================================================
DOCS = [
    "2026 Q1 蚂蚁集团 AI 战略报告: 公司计划投入 50 亿元用于金融大模型研发, "
    "重点投向反欺诈和智能投顾两个方向, 预计 2027 年完成内部推广。",

    "中国人民银行 2026 年 3 月发布《金融业人工智能应用监管指引(2026)》, "
    "要求大模型在金融场景应用必须通过模型备案, 数据不得出境, 推理结果必须可追溯。",

    "DeepSeek-V3.5 模型在金融问答 benchmark CFinBench 2026 上达到 89.2% 准确率, "
    "首次超越 GPT-4 和 Claude Opus 4, 成为金融领域开源模型 SOTA。",

    "招商银行私募基金部 2026 年部署内部 RAG 系统, 将研报检索时间从平均 12 分钟降至 30 秒, "
    "覆盖 200 多名投顾, 每月节省约 4000 工时。",

    "FinGPT 是哥伦比亚大学维护的开源金融大模型项目, 提供基于 LLaMA 和 Qwen 的 LoRA 微调脚本, "
    "支持中英文金融语料训练, 2026 年新版本加入 RAG 集成。",

    "向量数据库 Milvus 2.6 版本针对金融场景优化了混合检索 (向量 + BM25 关键词), "
    "在 CFinRetrieve benchmark 上召回率提升 15%, 延迟降低 30%。",

    "中国金融行业 2026 年大模型私有部署率达到 92%, 主要受三个因素驱动: "
    "数据合规要求、监管模型备案制度、客户隐私保护。公有云方案基本被排除在核心业务外。",
]


# =============================================================
# 2. Embedding 后端 — 两种选择
# =============================================================

class TfidfEmbedder:
    """TF-IDF embedding: 经典方案, 零下载, 适合 demo。

    工作原理:
      - TF (Term Frequency): 词在文档里出现的频率
      - IDF (Inverse Document Frequency): 词的稀有度 (常见词降权)
      - 文档向量 = TF * IDF (维度 = 词表大小)
    缺点: 不懂语义(同义词识别不了)。
    优点: 立即可用, 对短文档+关键词查询效果不错。
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # 中文按字切 (analyzer="char_wb"), 因为没有中文分词器
        # ngram_range=(2,3) 抓 2-3 字组合, 兼顾"词"和"短语"
        self.vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 3),
        )

    def encode_docs(self, docs):
        # fit_transform: 学习词表 + 编码
        mat = self.vec.fit_transform(docs).toarray().astype(np.float32)
        # 归一化让 cosine similarity = 点积
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return mat / norms

    def encode_query(self, query):
        # transform: 不重新学词表, 用已有词表编码
        vec = self.vec.transform([query]).toarray()[0].astype(np.float32)
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec

    def name(self):
        return "TF-IDF (sklearn)"


class NeuralEmbedder:
    """sentence-transformers embedding: 现代方案, 懂语义。"""

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"[加载 embedding 模型: {model_name}]")
        print("  第一次跑会下载 ~480MB, 后续从缓存读")
        print("  如果下载失败 (SSL/网络): 用默认 TF-IDF 模式 (不加 --neural)")
        self.encoder = SentenceTransformer(model_name)
        self.model_name = model_name

    def encode_docs(self, docs):
        return self.encoder.encode(docs, show_progress_bar=False,
                                    normalize_embeddings=True)

    def encode_query(self, query):
        return self.encoder.encode([query], normalize_embeddings=True)[0]

    def name(self):
        return f"sentence-transformers ({self.model_name})"


# =============================================================
# 3. RAG 核心
# =============================================================

class MinimalRAG:
    def __init__(self, embedder):
        self.embedder = embedder
        self.docs: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def add_docs(self, docs):
        self.docs = docs
        self.doc_vectors = self.embedder.encode_docs(docs)
        print(f"[已编码 {len(docs)} 个文档, 向量维度 = {self.doc_vectors.shape[1]}]")

    def retrieve(self, query, top_k=3):
        """三行核心检索逻辑:
          1. query → 向量
          2. 点积算相似度 (cosine, 因已归一化)
          3. 取 top-k
        """
        q_vec = self.embedder.encode_query(query)
        scores = self.doc_vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.docs[i]) for i in top_idx]

    def build_prompt(self, query, retrieved):
        """把检索到的上下文塞进 prompt — RAG 的"装配"步骤。"""
        context_block = "\n\n".join(f"[文档 {i+1}] {doc}"
                                     for i, (_, doc) in enumerate(retrieved))
        return f"""请基于以下文档回答问题。如果文档里没有相关信息, 请回答"未提及"。
不要编造文档中没有的内容。

参考文档:
{context_block}

问题: {query}

回答:"""


# =============================================================
# 4. 可选 LLM 调用 (OpenAI 兼容 API)
# =============================================================

def call_llm(prompt):
    try:
        from openai import OpenAI
    except ImportError:
        return "[错误] 未安装 openai, 跑 uv sync 装一下"

    api_key = os.getenv("LLM_API_KEY")
    if not api_key or api_key.startswith("sk-xxxx"):
        return ("[未配 LLM_API_KEY]\n"
                "  cp .env.example .env, 然后填上 API key (DeepSeek/Moonshot/...)")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL"),
    )
    model = os.getenv("LLM_MODEL", "deepseek-chat")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM 调用失败] {e}"


# =============================================================
# 5. 主流程
# =============================================================

def load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", action="store_true",
                        help="使用 sentence-transformers (默认 TF-IDF, 零下载)")
    parser.add_argument("--llm", action="store_true",
                        help="调用 LLM 生成最终回答 (需要 .env)")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    load_env()

    # === 选择 embedding 后端 ===
    embedder = NeuralEmbedder() if args.neural else TfidfEmbedder()
    print(f"[使用 embedding: {embedder.name()}]")

    rag = MinimalRAG(embedder)
    rag.add_docs(DOCS)

    queries = [
        "蚂蚁集团 2026 年 AI 投入金额是多少, 重点投向哪里?",
        "为什么中国金融大模型几乎都是私有部署?",
        "DeepSeek 在金融评测上的表现如何?",
    ]

    for q in queries:
        print("=" * 70)
        print(f"问题: {q}")
        print("=" * 70)

        retrieved = rag.retrieve(q, top_k=args.top_k)
        print(f"\n[检索结果] top-{args.top_k} (按相似度排序):")
        for i, (score, doc) in enumerate(retrieved):
            print(f"  #{i+1}  score={score:.4f}")
            print(f"        {doc[:80]}...")

        if args.llm:
            prompt = rag.build_prompt(q, retrieved)
            print(f"\n[LLM 回答]")
            print(call_llm(prompt))
        else:
            print(f"\n[未启用 LLM] 加 --llm 让 LLM 基于上下文回答")
        print()


if __name__ == "__main__":
    main()
