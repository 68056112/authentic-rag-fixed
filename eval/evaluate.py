"""วัดผลระบบเดิม (baseline) เทียบฉบับแก้ไข (fixed) ทีละขั้น -> results/metrics.md + results/metrics.json

    python eval/evaluate.py            # ครบทุกขั้น (ขั้น generation เรียก LLM; ผลทุกครั้ง cache ใน results/llm_cache.json)
    python eval/evaluate.py --no-llm   # ข้ามขั้น generation

ต้องรัน python authenticRAG.py (ฉบับแก้ไข) ก่อน เพื่อให้มี index ของ fixed
ชุดคำถาม eval/questions.json: chunk ถือว่า "ถูก" ถ้ามีข้อความ evidence (คัดลอกตรงจากเอกสาร) อยู่ข้างใน
"""
import json
import re
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rag_core  # noqa: E402
import baseline  # noqa: E402

QUESTIONS = json.loads((Path(__file__).with_name("questions.json")).read_text(encoding="utf-8"))
LABELLED = [q for q in QUESTIONS if q["evidence"]]
K = 5
NAMES = {"1.md": ["หัดเยอรมัน", "rubella", "german measles"], "2.md": ["อหิวาต", "cholera"],
         "44.md": ["ต้อกระจก", "cataract"], "5555.md": ["กรดไหลย้อน", "gerd", "reflux"]}
CACHE_FILE = rag_core.RESULTS / f"llm_cache.{rag_core._slug}.json"      # แยก cache ตามโมเดล
WORKERS = int(rag_core.os.environ.get("LLM_WORKERS", 4))
_lock = threading.Lock()
cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
client = rag_core.opensearch()
corpus = rag_core.read_corpus()
tok_count = rag_core.count_tokens
M = {}            # ตัวเลขทั้งหมด -> metrics.json
md = []           # ตาราง -> metrics.md


def cached_llm(key, prompt, **kw):
    if not cache.get(key):
        text = rag_core.llm(prompt, **kw)
        with _lock:
            cache[key] = text
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return cache[key]


def parallel(fn, items, label):
    """เรียก fn กับทุก item พร้อมกัน WORKERS ตัว (งาน LLM เท่านั้น) คืนผลตามลำดับเดิม"""
    out = []
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(fn, items), 1):
            out.append(r)
            print(f"  {label} {i}/{len(items)}", end="\r")
    print()
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def num(x, d=2):
    return "–" if x is None else (f"{x:,.{d}f}" if isinstance(x, float) else f"{x:,}")


def table(title, rows, note=""):
    md.append(f"\n## {title}\n")
    if note:
        md.append(note + "\n")
    md.append("| ตัวชี้วัด | ระบบเดิม | ฉบับแก้ไข | ทิศที่ดี |\n|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join(map(str, r)) + " |")
    print(f"\n=== {title}")
    for r in rows:
        print(f"  {r[0]:50} {r[1]:>14} {r[2]:>14}")


def names_disease(src, text):
    return any(n in text.lower() for n in NAMES[src])


def thai_share(t):
    letters = [ch for ch in t if ch.isalpha()]
    return sum("฀" <= ch <= "๿" for ch in letters) / max(len(letters), 1)


# ================================================================ เตรียมข้อมูลทั้งสองระบบ
base_chunks = baseline.chunks()
for c, text in zip(base_chunks, parallel(lambda c: cached_llm(baseline.context_key(c), baseline.context_prompt(c), temperature=0.1),
                                         base_chunks, "baseline context")):
    c["context"] = text
fixed_chunks = rag_core.load_chunks()
ctx_cache = json.loads(rag_core.CONTEXT_CACHE.read_text(encoding="utf-8"))
for c in fixed_chunks:
    c["context"] = ctx_cache.get(rag_core.context_key(c), "")
if not client.indices.exists(index=rag_core.INDEX):
    raise SystemExit("ยังไม่มี index ของฉบับแก้ไข — รัน python authenticRAG.py ก่อน")
baseline.build_index(base_chunks, client)

# ================================================================ S1 Chunking
def chunk_stats(cs):
    t = [tok_count(c["content"]) for c in cs]
    ev = [(q, e) for q in LABELLED for e in q["evidence"]]
    split = sum(not any(e in c["content"] for c in cs if c["source"] == q["file"]) for q, e in ev)
    return {"chunks": len(cs), "tok_median": statistics.median(t), "tok_max": max(t),
            "over_256": np.mean([x > 256 for x in t]), "over_512": np.mean([x > 512 for x in t]),
            "evidence_split": split / len(ev)}


s1 = {"baseline": chunk_stats(base_chunks), "fixed": chunk_stats(fixed_chunks)}
s1["baseline"]["chunk_size_has_effect"] = [c["content"] for c in baseline.chunks(4096)] != [c["content"] for c in base_chunks]
M["S1_chunking"] = s1
b, f = s1["baseline"], s1["fixed"]
table("S1 Chunking", [
    ("จำนวน chunk", b["chunks"], f["chunks"], ""),
    ("token ต่อ chunk (median)", num(b["tok_median"], 0), num(f["tok_median"], 0), ""),
    ("token ต่อ chunk (max)", num(b["tok_max"]), num(f["tok_max"]), "ต่ำ"),
    ("chunk ที่เกิน 256 token", pct(b["over_256"]), pct(f["over_256"]), "ต่ำ"),
    ("chunk ที่เกิน 512 token", pct(b["over_512"]), pct(f["over_512"]), "ต่ำ"),
    ("evidence ที่ถูกตัดขาดข้าม chunk", pct(b["evidence_split"]), pct(f["evidence_split"]), "ต่ำ"),
    ("ค่า chunk_size มีผลต่อการตัด", "ไม่มี" if not b["chunk_size_has_effect"] else "มี", "มี (MAX_CHARS)", ""),
], "token นับด้วย tokenizer ของ bge-m3")

# ================================================================ S2 Contextual Retrieval
def context_stats(cs, seen):
    orph = [c for c in cs if not names_disease(c["source"], c["content"])]
    ctx = [c["context"] for c in cs]
    return {"doc_seen_median": statistics.median(seen),
            "names_disease": np.mean([names_disease(c["source"], c["context"]) for c in cs]),
            "orphans": len(orph), "orphans_fixed": np.mean([names_disease(c["source"], c["context"]) for c in orph]) if orph else None,
            "thai": np.mean([thai_share(t) > 0.5 for t in ctx]),
            "whole_doc_confusion": np.mean([bool(re.search(r"entire document|whole document", t, re.I)) for t in ctx]),
            "empty": np.mean([not t for t in ctx])}


def outline_seen(c):
    if rag_core.CONTEXT_MODE == "full":
        return 1.0
    outline = "\n".join(l for l in corpus[c["source"]].splitlines() if l.startswith("#"))
    return min(1.0, (len(outline) + len(c["section"][:3000])) / len(corpus[c["source"]]))


s2 = {"baseline": context_stats(base_chunks, [len(c["content"]) / len(corpus[c["source"]]) for c in base_chunks]),
      "fixed": context_stats(fixed_chunks, [outline_seen(c) for c in fixed_chunks])}
M["S2_context"] = s2
b, f = s2["baseline"], s2["fixed"]
table("S2 Contextual Retrieval (context ที่ LLM เขียนให้แต่ละ chunk)", [
    ("สัดส่วนเอกสารที่ LLM เห็นตอนเขียน context (median)", pct(b["doc_seen_median"]), pct(f["doc_seen_median"]), "สูง"),
    ("context ที่ระบุชื่อโรค", pct(b["names_disease"]), pct(f["names_disease"]), "สูง"),
    ("chunk ไม่มีชื่อโรค ที่ context เติมชื่อโรคให้", f"{pct(b['orphans_fixed'])} (n={b['orphans']})",
     f"{pct(f['orphans_fixed'])} (n={f['orphans']})", "สูง"),
    ("context เป็นภาษาไทย", pct(b["thai"]), pct(f["thai"]), "สูง"),
    ("context ที่เข้าใจว่า chunk คือ 'ทั้งเอกสาร'", pct(b["whole_doc_confusion"]), pct(f["whole_doc_confusion"]), "ต่ำ"),
    ("chunk ที่ไม่มี context (สร้างไม่สำเร็จ)", pct(b["empty"]), pct(f["empty"]), "ต่ำ"),
], f"LLM: {rag_core.LLM_MODEL} · ฉบับแก้ไขใช้ CONTEXT_MODE={rag_core.CONTEXT_MODE} (ชื่อเรื่อง + สารบัญ + section แม่)")

# ================================================================ S3 Analyzer
def analyze(analyzer, text):
    return [t["token"] for t in client.indices.analyze(body={"analyzer": analyzer, "text": text})["tokens"]]


def is_thai(t):
    return any("฀" <= ch <= "๿" for ch in t)


def analyzer_stats(an, cs):
    vocab = {t for c in cs for t in analyze(an, c["content"][:5000]) if is_thai(t)}
    qt = [t for q in LABELLED for t in analyze(an, q["question"]) if is_thai(t)]
    return {"term_len_median": statistics.median(map(len, vocab)), "query_terms_in_index": np.mean([t in vocab for t in qt]),
            "query_terms_per_q": len(qt) / len(LABELLED)}


s3 = {"baseline": analyzer_stats("standard", base_chunks), "fixed": analyzer_stats("thai", fixed_chunks)}
M["S3_analyzer"] = s3
b, f = s3["baseline"], s3["fixed"]
table("S3 Analyzer (BM25)", [
    ("analyzer", "standard", "thai", ""),
    ("ความยาว term ภาษาไทยใน index (median, ตัวอักษร)", num(b["term_len_median"], 0), num(f["term_len_median"], 0), "สั้น ≈ เป็นคำ"),
    ("term ภาษาไทยต่อคำถาม (เฉลี่ย)", num(b["query_terms_per_q"], 1), num(f["query_terms_per_q"], 1), ""),
    ("term ในคำถามที่มีอยู่ใน index", pct(b["query_terms_in_index"]), pct(f["query_terms_in_index"]), "สูง"),
])

# ================================================================ S4 Vector index
def knn_check(index):
    fmap = client.indices.get_mapping(index=index)[index]["mappings"]["properties"]["embedding"]
    q = LABELLED[0]["question"]
    qv = rag_core.embedder().encode(q, normalize_embeddings=True)
    h = client.search(index=index, body={"size": 1, "query": {"knn": {"embedding": {"vector": qv.tolist(), "k": 1}}}})["hits"]["hits"][0]
    qv, dv = qv.astype(np.float64), np.array(h["_source"]["embedding"], dtype=np.float64)
    with np.errstate(all="ignore"):             # numpy บน macOS เตือน matmul ผิด ๆ ทั้งที่ค่าถูก
        cos = float(qv @ dv / np.linalg.norm(qv) / np.linalg.norm(dv))
    l2 = float(np.sum((qv - dv) ** 2))
    space = "cosinesimil" if abs(h["_score"] - (1 + cos) / 2) < abs(h["_score"] - 1 / (1 + l2)) else "l2"
    return {"method_in_mapping": bool(fmap.get("method")), "space_used": space, "norm": float(np.linalg.norm(dv))}


s4 = {"baseline": knn_check(baseline.VEC), "fixed": knn_check(rag_core.INDEX)}
M["S4_vector"] = s4
table("S4 Vector index", [
    ("ระบุ method/space_type ใน mapping ของ field", "ไม่" if not s4["baseline"]["method_in_mapping"] else "ใช่",
     "ใช่" if s4["fixed"]["method_in_mapping"] else "ไม่", ""),
    ("space ที่ใช้จริง (ตรวจจาก score)", s4["baseline"]["space_used"], s4["fixed"]["space_used"], "cosinesimil"),
    ("ความยาว vector", num(s4["baseline"]["norm"], 3), num(s4["fixed"]["norm"], 3), "1.000"),
], "ระบบเดิมได้ cosine จริงเพราะ OpenSearch 2.19 ยังอ่าน index.knn.space_type — ไม่ใช่ข้อผิดพลาด แต่เปราะ (ค่านี้ deprecated)")

# ================================================================ S5 Retrieval
def first_hit(texts, evidence):
    for rank, t in enumerate(texts[:K], 1):
        if any(e in t for e in evidence):
            return rank
    return None


def retrieval(run):
    ranks, purity, zero = [], [], 0
    for q in LABELLED:
        hits = run(q["question"])
        zero += not hits
        ranks.append(first_hit([s["content"] for _, s in hits], q["evidence"]))
        purity.append(np.mean([s["source"] == q["file"] for _, s in hits]) if hits else 0.0)
    n = len(LABELLED)
    return {"Hit@1": sum(r == 1 for r in ranks) / n, "Hit@5": sum(r is not None for r in ranks) / n,
            "MRR@5": sum(1 / r for r in ranks if r) / n, "FilePurity@5": float(np.mean(purity)), "ZeroHits": zero / n}


s5 = {"baseline": {m: retrieval(lambda q, m=m: baseline.search(q, m, client)) for m in ("bm25", "dense", "hybrid")},
      "fixed": {m: retrieval(lambda q, m=m: rag_core.search(q, m, client=client)) for m in ("bm25", "dense", "hybrid", "rerank")}}
M["S5_retrieval"] = s5
md.append(f"\n## S5 Retrieval ({len(LABELLED)} คำถาม, top-{K})\n")
md.append("| ระบบ | โหมด | Hit@1 | Hit@5 | MRR@5 | chunk มาจากไฟล์ที่ถูก | ค้นไม่เจอเลย |\n|---|---|---|---|---|---|---|")
print(f"\n=== S5 Retrieval")
for sysname, modes in (("ระบบเดิม", s5["baseline"]), ("ฉบับแก้ไข", s5["fixed"])):
    for m, x in modes.items():
        line = f"| {sysname} | {m} | {x['Hit@1']:.2f} | {x['Hit@5']:.2f} | {x['MRR@5']:.2f} | {pct(x['FilePurity@5'])} | {pct(x['ZeroHits'])} |"
        md.append(line)
        print("  " + line)
md.append("\nโหมดที่ระบบใช้จริง: ระบบเดิม = hybrid, ฉบับแก้ไข = rerank")

# ================================================================ S6 Context ที่ส่งให้ LLM ตอนตอบ
base_ctx = {q["id"]: baseline.build_context(baseline.search(q["question"], "hybrid", client)) for q in QUESTIONS}
fixed_hits = {q["id"]: rag_core.search(q["question"], "rerank", client=client) for q in QUESTIONS}
fixed_ctx = {i: rag_core.build_context(h) for i, h in fixed_hits.items()}
full_base = {q["id"]: "".join(f"Content: {s['content']}\nContext: {s['contextualized_content']}\n\n"
                              for _, s in baseline.search(q["question"], "hybrid", client)) for q in QUESTIONS}
real = [tok_count(t) for t in full_base.values()]
est = [baseline.estimate_tokens(t) for t in full_base.values()]
s6 = {"baseline": {"real_tokens_median": statistics.median(real), "est_tokens_median": statistics.median(est),
                   "underestimate_x": statistics.median([r / e for r, e in zip(real, est)]),
                   "over_budget": np.mean([r > 4000 for r in real]), "truncated": np.mean([e > 4000 for e in est])},
      "fixed": {"real_tokens_median": statistics.median([tok_count(t) for t in fixed_ctx.values()]),
                "est_tokens_median": statistics.median([tok_count(t) for t in fixed_ctx.values()]),
                "underestimate_x": 1.0,
                "over_budget": np.mean([tok_count(t) > rag_core.MAX_CONTEXT_TOKENS for t in fixed_ctx.values()]),
                "truncated": None}}
M["S6_context_budget"] = s6
b, f = s6["baseline"], s6["fixed"]
table("S6 Context ที่ส่งให้ LLM ตอนตอบ", [
    ("งบ token", "4,000 (ประมาณ)", f"{rag_core.MAX_CONTEXT_TOKENS:,} (นับจริง)", ""),
    ("token จริงต่อคำถาม (median)", num(b["real_tokens_median"], 0), num(f["real_tokens_median"], 0), "ต่ำ"),
    ("ตัวนับคิดว่าใช้ (median)", num(b["est_tokens_median"], 0), num(f["est_tokens_median"], 0), "= ค่าจริง"),
    ("นับต่ำกว่าจริง (เท่า)", f"{b['underestimate_x']:.1f}x", "1.0x", "1.0x"),
    ("คำถามที่ context จริงเกินงบ", pct(b["over_budget"]), pct(f["over_budget"]), "0%"),
])

# ================================================================ S7 Generation
if "--no-llm" not in sys.argv:
    JUDGE = """You are grading a RAG system's answer. Reply with JSON only.

Question: {question}
Reference answer (ground truth): {reference}

Retrieved context given to the system:
<context>
{context}
</context>

System answer:
<answer>
{answer}
</answer>

Grade:
- "correct": 2 if the answer contains the key facts of the reference answer, 1 if partially, 0 if wrong or missing.
  If the reference says the information is not in the documents, give 2 only if the answer says it cannot be found in the context.
- "claims": number of distinct factual claims in the answer.
- "unsupported": how many of those claims are NOT supported by the retrieved context.
{{"correct": <0|1|2>, "claims": <int>, "unsupported": <int>}}"""

    def judge(system, q, ctx, ans):
        for attempt in range(3):
            raw = cached_llm(f"judge|{system}|{q['id']}|{attempt}", JUDGE.format(
                question=q["question"], reference=q["reference_answer"], context=ctx, answer=ans[:3000]),
                temperature=0, max_tokens=8000)
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                return json.loads(m.group())
            except Exception:
                continue
        return {"correct": None, "claims": None, "unsupported": None}

    # อันดับที่เจอ evidence (ค้น OpenSearch/embedding ทีละข้อ) แล้วค่อยเรียก LLM แบบขนาน
    base_rank = {q["id"]: first_hit([s["content"] for _, s in baseline.search(q["question"], "hybrid", client)], q["evidence"])
                 if q["evidence"] else None for q in QUESTIONS}
    fixed_rank = {q["id"]: first_hit([s["content"] for _, s in fixed_hits[q["id"]]], q["evidence"])
                  if q["evidence"] else None for q in QUESTIONS}

    def run(q):
        ans_b = cached_llm(f"baseline_answer|{q['id']}", baseline.TEACHER_ANSWER_PROMPT.format(
            context=base_ctx[q["id"]], question=q["question"]), temperature=0.6)
        ans_f = cached_llm(f"fixed_answer|{q['id']}", rag_core.ANSWER_PROMPT.format(
            context=fixed_ctx[q["id"]], question=q["question"]), temperature=0.3)
        return [{"id": q["id"], "type": q["type"], "rank": rank, "answer": ans, **judge(system, q, ctx, ans)}
                for system, ctx, ans, rank in (("baseline", base_ctx[q["id"]], ans_b, base_rank[q["id"]]),
                                               ("fixed", fixed_ctx[q["id"]], ans_f, fixed_rank[q["id"]]))]

    pairs = parallel(run, QUESTIONS, "generation")
    rows = {"baseline": [p[0] for p in pairs], "fixed": [p[1] for p in pairs]}

    def gen_stats(rs):
        g = [r for r in rs if r["correct"] is not None]
        lab = [r for r in g if r["id"] != "q29"]
        claims = sum(r["claims"] or 0 for r in g)
        q29 = next((r for r in g if r["id"] == "q29"), None)
        return {"graded": len(g), "correct": np.mean([r["correct"] == 2 for r in lab]),
                "partial": np.mean([r["correct"] == 1 for r in lab]), "wrong": np.mean([r["correct"] == 0 for r in lab]),
                "score": np.mean([r["correct"] / 2 for r in lab]),
                "unsupported_rate": sum(r["unsupported"] or 0 for r in g) / max(claims, 1),
                "answers_with_unsupported": np.mean([(r["unsupported"] or 0) > 0 for r in g]),
                "miss_retrieval": sum(r["rank"] is None and r["correct"] < 2 for r in lab),
                "miss_generation": sum(r["rank"] is not None and r["correct"] < 2 for r in lab),
                "q29_refused": None if q29 is None else q29["correct"] == 2,
                "rows": rs}

    s7 = {s: gen_stats(r) for s, r in rows.items()}
    M["S7_generation"] = s7
    b, f = s7["baseline"], s7["fixed"]
    yn = {True: "ใช่", False: "ไม่ (แต่งคำตอบ)", None: "–"}
    table(f"S7 Generation ({len(QUESTIONS)} คำถาม; q29 ไม่มีคำตอบในเอกสาร)", [
        ("ตอบถูกครบ (correct=2)", pct(b["correct"]), pct(f["correct"]), "สูง"),
        ("ตอบถูกบางส่วน (correct=1)", pct(b["partial"]), pct(f["partial"]), ""),
        ("ตอบผิด (correct=0)", pct(b["wrong"]), pct(f["wrong"]), "ต่ำ"),
        ("คะแนนเฉลี่ย (0-1)", f"{b['score']:.2f}", f"{f['score']:.2f}", "สูง"),
        ("claim ที่ไม่มีใน context", pct(b["unsupported_rate"]), pct(f["unsupported_rate"]), "ต่ำ"),
        ("คำตอบที่มี claim นอก context ≥ 1", pct(b["answers_with_unsupported"]), pct(f["answers_with_unsupported"]), "ต่ำ"),
        ("ไม่ถูกครบ เพราะ retrieval ไม่เจอ evidence", f"{b['miss_retrieval']} ข้อ", f"{f['miss_retrieval']} ข้อ", "ต่ำ"),
        ("ไม่ถูกครบ ทั้งที่ retrieval เจอ evidence", f"{b['miss_generation']} ข้อ", f"{f['miss_generation']} ข้อ", "ต่ำ"),
        ("q29: บอกว่าไม่มีข้อมูลในเอกสาร", yn[b["q29_refused"]], yn[f["q29_refused"]], "ใช่"),
        ("คำถามที่ judge ให้คะแนนได้", f"{b['graded']}/{len(QUESTIONS)}", f"{f['graded']}/{len(QUESTIONS)}", ""),
    ], f"judge = {rag_core.LLM_MODEL} ตัวเดียวกับที่ตอบ เห็น context เต็มที่ส่งให้ระบบ"
       + (" · ผ่าน Claude Code CLI จึงตั้ง temperature ไม่ได้ (prompt ของทั้งสองระบบได้ค่า default เท่ากัน)"
          if rag_core.LLM_PROVIDER == "claude-cli" else " (temperature 0)"))

# ================================================================ บันทึก
for n in (baseline.VEC, baseline.BM25):
    client.indices.delete(index=n)
head = [f"# ตารางวัดผล: ระบบเดิม vs ฉบับแก้ไข\n",
        f"สร้างโดย `python eval/evaluate.py` · embedding BAAI/bge-m3 · LLM {rag_core.LLM_MODEL} · "
        f"คำถาม {len(QUESTIONS)} ข้อ (มี evidence {len(LABELLED)} ข้อ)"]
rag_core.RESULTS.mkdir(exist_ok=True)
(rag_core.RESULTS / "metrics.md").write_text("\n".join(head + md) + "\n", encoding="utf-8")
(rag_core.RESULTS / "metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("\nบันทึกที่ results/metrics.md และ results/metrics.json")
