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

【数据流】
                                         ┌──────────┐
   文档库 ──切块──> chunks ──编码──> vectors          │
                                                    │
                                              vector DB
                                                    │
   用户问题 ──编码──> query vector ──相似度检索─────┘
                                          │
                                          ▼
                                    top-k 相关 chunks
                                          │
                                          ▼
                                   填入 prompt 模板
                                          │
                                          ▼
                                        LLM
                                          │
                                          ▼
                                   "基于上下文的回答"

【本 demo 的极简实现】
  - 文档:7 段 2026 年的金融 AI 假数据 (LLM 训练数据里没有)
  - Embedding:sentence-transformers 多语言模型 (~100MB)
  - 向量库:numpy 数组 + cosine similarity (不用 Milvus 等专业 DB)
  - LLM:OpenAI 兼容 API (DeepSeek/Moonshot/Ollama 都行); 没配则跳过

【运行】
  uv sync
  uv run python rag_demo.py        # 不调用 LLM, 只看检索结果
  uv run python rag_demo.py --llm  # 调用 LLM (需要先在 .env 配 API)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


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
# 2. RAG 的核心组件 — 用 numpy + sentence-transformers 实现
# =============================================================

class MinimalRAG:
    """50 行代码实现 RAG 核心逻辑。"""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        # 加载 embedding 模型 (第一次运行会下载 ~480MB)
        # 备选 (中文更好但 1.5GB):  BAAI/bge-large-zh-v1.5
        # 备选 (英文最小 80MB):    all-MiniLM-L6-v2
        print(f"[加载 embedding 模型: {model_name}]")
        self.encoder = SentenceTransformer(model_name)
        self.docs: list[str] = []
        self.doc_vectors: np.ndarray | None = None

    def add_docs(self, docs: list[str]):
        """把文档库编码成向量, 存进内存。

        生产环境会换成 Milvus / Qdrant / pgvector, 但原理一样:
        - 编码: 文档 → 768/1024 维向量
        - 存储: 向量索引以便快速检索
        """
        self.docs = docs
        # encode 返回 shape (N, dim) 的 numpy 数组
        # normalize_embeddings=True 让所有向量长度为 1, 后续点积 = cosine similarity
        self.doc_vectors = self.encoder.encode(
            docs,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        print(f"[已编码 {len(docs)} 个文档, 向量维度 = {self.doc_vectors.shape[1]}]")

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[float, str]]:
        """检索最相关的 top_k 个文档。

        步骤:
          1. 把 query 也编码成向量
          2. 算 query 和每个文档的 cosine similarity
          3. 取相似度最高的 top_k 个
        """
        # query → 向量 (1, dim)
        q_vec = self.encoder.encode([query], normalize_embeddings=True)[0]

        # cosine similarity = 点积 (因为都归一化了)
        # (N, dim) @ (dim,) = (N,) — 每个文档对 query 的相似度
        scores = self.doc_vectors @ q_vec

        # 取 top_k (argsort 默认升序, 取后 k 个倒过来)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self.docs[i]) for i in top_idx]

    def build_prompt(self, query: str, retrieved: list[tuple[float, str]]) -> str:
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
# 3. 可选: 调用 LLM (兼容 OpenAI API 格式)
# =============================================================

def call_llm(prompt: str) -> str:
    """调用 OpenAI 兼容的 LLM API (DeepSeek / Moonshot / OpenAI / Ollama 都可)。

    通过环境变量配置 (见 .env.example):
      LLM_API_KEY    API key
      LLM_BASE_URL   端点 (例: https://api.deepseek.com/v1)
      LLM_MODEL      模型名 (例: deepseek-chat)
    """
    try:
        from openai import OpenAI
    except ImportError:
        return "[错误] 未安装 openai, 跑 uv sync 装一下"

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    if not api_key:
        return ("[未配 LLM_API_KEY, 跳过 LLM 调用]\n"
                "  设置方法: cp .env.example .env, 然后填上 API key\n"
                "  例如使用 DeepSeek: 在 https://platform.deepseek.com 注册拿 key")

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,   # 低温度让 LLM 严格基于上下文
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM 调用失败] {e}"


# =============================================================
# 4. Demo 主流程
# =============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help="调用 LLM 生成最终回答 (需要先配 .env)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="检索时取前 K 个相关文档 (默认 3)")
    args = parser.parse_args()

    # 加载 .env (如果有)
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # === 初始化 RAG ===
    rag = MinimalRAG()
    rag.add_docs(DOCS)

    # === 3 个测试 query (这些答案在 LLM 训练数据里都没有) ===
    queries = [
        "蚂蚁集团 2026 年 AI 投入金额是多少, 重点投向哪里?",
        "为什么中国金融大模型几乎都是私有部署?",
        "DeepSeek 在金融评测上的表现如何?",
    ]

    for q in queries:
        print("=" * 70)
        print(f"问题: {q}")
        print("=" * 70)

        # --- 步骤 1: 检索 ---
        retrieved = rag.retrieve(q, top_k=args.top_k)
        print(f"\n[检索结果] top-{args.top_k} (按相似度排序):")
        for i, (score, doc) in enumerate(retrieved):
            print(f"  #{i+1}  score={score:.4f}")
            print(f"        {doc[:80]}...")

        # --- 步骤 2: 装配 prompt ---
        prompt = rag.build_prompt(q, retrieved)

        # --- 步骤 3: LLM 生成 (可选) ---
        if args.llm:
            print(f"\n[LLM 回答]")
            answer = call_llm(prompt)
            print(answer)
        else:
            print(f"\n[未启用 LLM 调用] 加 --llm 参数让 LLM 基于上下文生成回答")

        print()


if __name__ == "__main__":
    main()
