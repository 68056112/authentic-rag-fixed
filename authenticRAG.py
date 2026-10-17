"""นำเอกสารเข้า OpenSearch (ฉบับแก้ไขของ authenticRAG.py)

    python authenticRAG.py

ตัด chunk -> ให้ LLM เขียน context ต่อ chunk (cache ไว้ที่ results/contexts_cache.json)
-> embed ด้วย bge-m3 -> index เดียวที่มีทั้ง BM25 (thai analyzer) และ vector
"""
import rag_core

if __name__ == "__main__":
    rag_core.ingest()
