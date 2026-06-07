# 最小化 RAG Demo

200 行核心代码看清 RAG (Retrieval-Augmented Generation) 全流程。
**默认零模型下载, 立即可跑**;可选升级到神经 embedding + LLM 调用。

---

## 为什么需要 RAG?

LLM (GPT/Qwen/DeepSeek) 训练数据有截止日期, **不知道**:

- 公司内部研报、合同、合规文档
- 最新政策、新闻、财报

且 LLM 容易**幻觉**(瞎编看似合理但错误的答案)。

→ **RAG = 给 LLM 一本"参考书", 让它基于参考书回答, 不再瞎编。**

---

## 核心思路 (4 步)

```
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
```

| 步骤 | 名字 | 本 demo 做的 |
|------|------|-------------|
| 1 | Chunking | 文档已经是块 (没切) |
| 2 | Embedding | TF-IDF 或 sentence-transformers |
| 3 | Retrieval | numpy 点积 + top-k |
| 4 | Generation | 装配 prompt + OpenAI 兼容 API (可选) |

---

## 极简实现

| 组件 | 选择 | 理由 |
|------|------|------|
| 文档库 | 7 段假数据 (2026 年的, LLM 不可能见过) | 不需要文件 IO |
| Embedding | **TF-IDF (默认)** 或 sentence-transformers | TF-IDF 零下载 |
| 向量 DB | `numpy` 数组 + cosine 相似度 | 不需要 Milvus |
| LLM | OpenAI 兼容 API (可选) | 支持 DeepSeek/Moonshot/Ollama |

---

## 快速开始

### 1. 安装(默认依赖, 不需联网下载 AI 模型)

```bash
cd rag/minimal-demo
uv sync
```

### 2. 跑 demo (默认 TF-IDF + 不调 LLM)

```bash
uv run python rag_demo.py
```

→ **3 秒内出结果**, 看到每个问题的 top-3 相关文档和相似度分数。

### 3. 升级到神经 embedding

```bash
uv sync --extra neural    # 装 sentence-transformers + torch (~2GB)
uv run python rag_demo.py --neural
```

第一次跑会下载 ~480MB 模型(可能需要科学上网或用 HF 镜像)。

### 4. 调 LLM 生成回答

复制配置模板:

```bash
cp .env.example .env
# 编辑 .env, 填入你的 API key
```

推荐 **DeepSeek**(国内友好,便宜):

```
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

然后:

```bash
uv run python rag_demo.py --llm                # TF-IDF + LLM
uv run python rag_demo.py --neural --llm       # 神经 embedding + LLM
```

---

## 验证过的输出 (默认 TF-IDF, 无 LLM)

```
[使用 embedding: TF-IDF (sklearn)]
[已编码 7 个文档, 向量维度 = 1108]
======================================================================
问题: 蚂蚁集团 2026 年 AI 投入金额是多少, 重点投向哪里?
======================================================================

[检索结果] top-3 (按相似度排序):
  #1  score=0.4370
        2026 Q1 蚂蚁集团 AI 战略报告: 公司计划投入 50 亿元用于金融大模型研发...
  #2  score=0.0969
        中国人民银行 2026 年 3 月发布《金融业人工智能应用监管指引(2026)》...
  #3  score=0.0546
        招商银行私募基金部 2026 年部署内部 RAG 系统...

问题: 为什么中国金融大模型几乎都是私有部署?
  #1  score=0.2479
        中国金融行业 2026 年大模型私有部署率达到 92%, 主要受三个因素驱动...

问题: DeepSeek 在金融评测上的表现如何?
  #1  score=0.3373
        DeepSeek-V3.5 模型在金融问答 benchmark CFinBench 2026 上达到 89.2%...
```

→ **每个问题的 top-1 都是真正相关的文档,top-2/3 明显较低,说明检索区分度好。**

---

## TF-IDF vs 神经 embedding 对比

| | TF-IDF | sentence-transformers |
|---|--------|----------------------|
| 模型下载 | 0 (零) | ~480MB |
| 启动速度 | 立即 | 几秒 |
| 懂同义词 | ❌ 不懂 | ✅ 懂 |
| 懂跨语言 | ❌ 不懂 | ✅ 多语言 |
| 适合 demo | ✓ 适合 | 也适合 |
| 适合生产 | 部分场景 (混合检索的关键词部分) | 主流方案 |

→ **本 demo 默认 TF-IDF 是为了"开箱即跑";真实项目用神经 embedding 或两者结合。**

---

## 关键代码段速览

### TF-IDF Embedding(默认,零下载)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
doc_vectors = vec.fit_transform(docs).toarray()
```

### 神经 Embedding(可选升级)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
doc_vectors = encoder.encode(docs, normalize_embeddings=True)
```

### Retrieval(检索 — 核心 3 行,两种 embedding 都一样)

```python
q_vec = encode(query)
scores = doc_vectors @ q_vec   # cosine 相似度
top_idx = np.argsort(scores)[::-1][:top_k]
```

### Prompt 装配(RAG 的"魂"在这里)

```python
prompt = f"""请基于以下文档回答问题。如果文档里没有相关信息, 回答"未提及"。

参考文档:
{context_block}

问题: {query}
回答:"""
```

→ **这就是 RAG 的全部核心。** 其他都是工程细节(更好的 chunking、更大的模型、向量 DB、reranker)。

---

## 从 demo 到生产 (下一步)

| 阶段 | demo 做的 | 生产应该做的 |
|------|----------|-------------|
| Chunking | 无 (文档本身就是一块) | 按段落/语义切, 处理 PDF/Word/HTML |
| Embedding | TF-IDF / MiniLM-L12 | BGE-large / OpenAI text-embedding-3-large |
| 向量 DB | numpy 数组 | Milvus / Qdrant / pgvector |
| 检索 | 纯向量相似度 | 混合检索 (向量 + BM25 + reranker) |
| Prompt | 简单拼接 | 系统提示、few-shot、citation |
| 评估 | 肉眼看 | RAGAS, faithfulness, retrieval recall |

学完这个 demo 后, 推荐:

1. **加 chunking**: 用 `langchain.text_splitter` 切真实 PDF
2. **换向量 DB**: 改用 Qdrant 或 ChromaDB (一样 API)
3. **加 reranker**: 用 BGE-reranker 提升 top-k 准确度
4. **评估**: 用 RAGAS 算 faithfulness / context precision

---

## 在金融场景的应用方向

| 场景 | 文档库 | 价值 |
|------|--------|------|
| 投研助手 | 历史研报 + 财报 + 行业报告 | 节省研究员翻阅时间 |
| 合规问答 | 监管文件 + 公司内规 | 替代法务咨询初筛 |
| 客户服务 | 产品说明书 + FAQ | 24/7 智能客服 |
| 风控审查 | 历史案例 + 合规要点 | 加快人工审核 |

→ **这就是金融 AI 工程师最常做的事**, 也是当前需求最大的方向之一。
