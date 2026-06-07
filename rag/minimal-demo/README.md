# 最小化 RAG Demo

50 行核心代码看清 RAG (Retrieval-Augmented Generation) 全流程。
单文件、无数据库依赖、可选 LLM 调用。

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

1. **Chunking** — 把文档切成小块
2. **Embedding** — 把每块转成向量
3. **Retrieval** — 用户问问题时, 找最相关的几块
4. **Generation** — 把这几块作为"上下文"喂给 LLM 生成回答

---

## 本 demo 的极简实现

| 组件 | 选择 | 理由 |
|------|------|------|
| 文档库 | 7 段假数据 (内嵌 Python list) | 不需要文件 IO |
| Embedding | `sentence-transformers` 多语言模型 | 一行调用, 100MB |
| 向量 DB | `numpy` 数组 + cosine 相似度 | 不需要 Milvus |
| LLM | OpenAI 兼容 API (可选) | 支持 DeepSeek/Moonshot/Ollama |

**关键文件**:`rag_demo.py` (200 行, 一半是注释)

---

## 快速开始

### 1. 安装依赖

```bash
cd rag/minimal-demo
uv sync
```

第一次跑会下载 embedding 模型(约 480MB),后续从本地缓存读。

### 2. 不调 LLM, 只看检索

```bash
uv run python rag_demo.py
```

会输出 3 个测试问题的检索结果(top-3 相关文档 + 相似度分数),
最后告诉你"加 `--llm` 让 LLM 基于这些文档回答"。

### 3. 调 LLM 生成回答

复制配置模板:

```bash
cp .env.example .env
# 编辑 .env, 填入你的 API key
```

推荐 **DeepSeek**(中国友好,便宜):

```
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

然后跑:

```bash
uv run python rag_demo.py --llm
```

---

## 期望输出 (无 LLM 模式)

```
[加载 embedding 模型: paraphrase-multilingual-MiniLM-L12-v2]
[已编码 7 个文档, 向量维度 = 384]
======================================================================
问题: 蚂蚁集团 2026 年 AI 投入金额是多少, 重点投向哪里?
======================================================================

[检索结果] top-3 (按相似度排序):
  #1  score=0.7245
        2026 Q1 蚂蚁集团 AI 战略报告: 公司计划投入 50 亿元用于金融大模型...
  #2  score=0.3812
        中国金融行业 2026 年大模型私有部署率达到 92%, 主要受三个因素驱动...
  #3  score=0.3501
        FinGPT 是哥伦比亚大学维护的开源金融大模型项目...

[未启用 LLM 调用] 加 --llm 参数让 LLM 基于上下文生成回答
```

→ **可以看到 #1 是真正相关的, 相似度 0.72; 其他两个明显不相关 (~0.35)**

---

## 期望输出 (调 LLM 后)

```
问题: 蚂蚁集团 2026 年 AI 投入金额是多少, 重点投向哪里?
...
[LLM 回答]
蚂蚁集团 2026 年计划投入 50 亿元用于金融大模型研发, 重点投向反欺诈和
智能投顾两个方向, 预计 2027 年完成内部推广。
```

→ LLM 准确复述了文档内容, 不再瞎编。

**关键验证**:这些 2026 年的"假数据"是 LLM 训练时不可能见过的。
如果不用 RAG 直接问 LLM, 它要么说"不知道",要么瞎编一个金额。
**这就是 RAG 的价值。**

---

## 关键代码段速览

### Embedding (编码)

```python
self.encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
self.doc_vectors = self.encoder.encode(docs, normalize_embeddings=True)
```

### Retrieval (检索 — 核心 3 行)

```python
q_vec = self.encoder.encode([query], normalize_embeddings=True)[0]
scores = self.doc_vectors @ q_vec   # cosine similarity (因为已归一化)
top_idx = np.argsort(scores)[::-1][:top_k]
```

### Prompt 装配

```python
prompt = f"""请基于以下文档回答问题。如果文档里没有相关信息, 回答"未提及"。

参考文档:
{context_block}

问题: {query}
回答:"""
```

→ **这就是 RAG 的全部核心。** 其他都是工程细节(更好的 chunking、更大的模型、向量 DB)。

---

## 从 demo 到生产 (下一步)

| 阶段 | demo 做的 | 生产应该做的 |
|------|----------|-------------|
| Chunking | 无 (文档本身就是一块) | 按段落/语义切, 处理 PDF/Word |
| Embedding | sentence-transformers 小模型 | BGE-large / OpenAI text-embedding-3-large |
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
