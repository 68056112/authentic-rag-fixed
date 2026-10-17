"""จำลองระบบเดิม (aekanun2020/2025-authenticRAG: authenticRAG.py + onlysearchAuthenticRAG.py) ให้เหมือนทุกจุด
เพื่อวัดเทียบกับฉบับแก้ไข — ใช้ index ชื่อ baseline-* แยกจากของจริง

ทุกอย่างในไฟล์นี้ตั้งใจคงข้อผิดพลาดของโค้ดเดิมไว้ อย่านำไปใช้งานจริง
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag_core  # noqa: E402

VEC, BM25 = "baseline-vector-index", "baseline-bm25-index"

TEACHER_CONTEXT_PROMPT = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval. Answer only with the succinct context and nothing else.
"""

TEACHER_ANSWER_PROMPT = """You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""


def chunks(chunk_size=256):
    """MarkdownNodeParser(chunk_size=256) แบบ load_documents() เดิม"""
    from llama_index.core import Document
    from llama_index.core.node_parser import MarkdownNodeParser
    docs = [Document(text=t, metadata={"source": f}) for f, t in rag_core.read_corpus().items()]
    nodes = MarkdownNodeParser(chunk_size=chunk_size).get_nodes_from_documents(docs)
    return [{"source": n.metadata["source"], "content": n.text} for n in nodes]


def context_key(c):
    return "baseline_ctx|" + rag_core.text_key(f"{c['source']}|{c['content'][:200]}")


def context_prompt(c):
    # โค้ดเดิม: full_doc_content = doc.page_content  -> <document> คือ chunk ตัวเอง
    return TEACHER_CONTEXT_PROMPT.format(document=c["content"], chunk=c["content"])


def build_index(cs, client):
    """index 2 ตัวแบบ _create_or_update_indices() + add_documents_with_context() เดิม"""
    for n in (VEC, BM25):
        if client.indices.exists(index=n):
            client.indices.delete(index=n)
    client.indices.create(index=VEC, body={"settings": {"index.knn": True, "index.knn.space_type": "cosinesimil"},
                                           "mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": 1024}}}})
    client.indices.create(index=BM25, body={"mappings": {"properties": {
        "content": {"type": "text", "analyzer": "standard"}, "contextualized_content": {"type": "text", "analyzer": "standard"}}}})
    embs = rag_core.embedder().encode([f"{c['content']}\n\n{c['context']}" for c in cs], batch_size=8, show_progress_bar=False)
    body = []
    for i, (c, e) in enumerate(zip(cs, embs)):
        src = {"content": c["content"], "contextualized_content": c["context"], "source": c["source"]}
        body += [{"index": {"_index": VEC, "_id": f"doc_{i}"}}, {**src, "embedding": e.tolist()},
                 {"index": {"_index": BM25, "_id": f"doc_{i}"}}, src]
    assert not client.bulk(body=body, refresh=True)["errors"]


def sparse(q, client, k=10):
    hits = client.search(index=BM25, body={"size": k, "query": {"multi_match": {
        "query": q, "fields": ["content", "contextualized_content"], "type": "best_fields"}}})["hits"]["hits"]
    return [(h["_id"], h["_source"]) for h in hits]


def dense(q, client, k=10):
    hits = client.search(index=VEC, body={"size": k, "_source": {"excludes": ["embedding"]}, "query": {
        "knn": {"embedding": {"vector": rag_core.embedder().encode(q).tolist(), "k": k}}}})["hits"]["hits"]
    return [(h["_id"], h["_source"]) for h in hits]


def hybrid(q, client, k=5):
    s, src = {}, {}
    for ranking in (sparse(q, client, 2 * k), dense(q, client, 2 * k)):
        for rank, (d, so) in enumerate(ranking):          # rank เริ่ม 0 แบบเดิม
            s[d] = s.get(d, 0) + 1 / (60 + rank)
            src[d] = so
    return [(d, src[d]) for d in sorted(s, key=s.get, reverse=True)][:k]


def search(q, mode, client):
    if mode == "bm25":
        return sparse(q, client)[:5]
    if mode == "dense":
        return dense(q, client)[:5]
    return hybrid(q, client)


def estimate_tokens(context):
    """truncate_context() เดิม: นับคำด้วยช่องว่าง x 1.3"""
    return len(context.split()) * 1.3


def build_context(hits):
    ctx = "".join(f"Content: {s['content']}\nContext: {s['contextualized_content']}\n\n" for _, s in hits)
    return ctx[:12000]   # ตัดที่ 12k ตัวอักษรเพื่อให้อยู่ใน quota (truncate_context เดิมไม่ตัดเลย)
