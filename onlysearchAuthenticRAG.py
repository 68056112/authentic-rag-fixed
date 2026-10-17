"""ถาม-ตอบจาก index ที่ authenticRAG.py สร้าง (ฉบับแก้ไขของ onlysearchAuthenticRAG.py)

    python onlysearchAuthenticRAG.py                      # ชุดคำถามเดิม 5 ข้อ -> authentic_rag_search_results.json
    python onlysearchAuthenticRAG.py "คำถาม"              # ถามข้อเดียว
    python onlysearchAuthenticRAG.py "คำถาม" hybrid       # เลือกโหมดค้น: bm25 | dense | hybrid | rerank
"""
import json
import sys

import rag_core

QUESTIONS = [
    "โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?",
    "อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน",
    "ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?",
    "ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?",
    "โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?",
]


def ask(question, mode="rerank"):
    hits, _, ans = rag_core.answer(question, mode)
    return {"question": question, "answer": ans,
            "results": [{"chunk_id": d, "source": s["source"], "breadcrumb": s["breadcrumb"],
                         "content": s["content"], "context": s["context"]} for d, s in hits]}


def main(argv):
    if argv:
        r = ask(argv[0], *(argv[1:2] or ["rerank"]))
        for i, h in enumerate(r["results"], 1):
            print(f"[{i}] {h['source']} > {h['breadcrumb']}\n    {h['content'][:100]!r}")
        print("\nคำตอบ:", r["answer"])
        return
    results = []
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n===== คำถามที่ {i}/{len(QUESTIONS)}: {q}")
        results.append(ask(q))
        print(results[-1]["answer"])
    with open("authentic_rag_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nบันทึกผลลัพธ์ไปยัง authentic_rag_search_results.json")


if __name__ == "__main__":
    main(sys.argv[1:])
