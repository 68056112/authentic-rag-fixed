"""Authentic RAG (ฉบับแก้ไข) — ของที่ authenticRAG.py และ onlysearchAuthenticRAG.py ใช้ร่วมกัน

แต่ละจุดที่แก้มีหมายเลข [F1]..[F8] ตรงกับหัวข้อใน REPORT.md

ตั้งค่าผ่าน environment หรือไฟล์ .env ที่โฟลเดอร์นี้ (หรือชี้ไฟล์อื่นด้วย RAG_ENV_FILE=/path/to/.env):
    OPENROUTER_API_KEY=sk-or-...                 # ค่าเริ่มต้น: Qwen ผ่าน OpenRouter แบบโค้ดเดิม
  หรือ LLM เจ้าอื่นที่รองรับ OpenAI SDK:
    LLM_API_KEY=...
    LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    LLM_MODEL=gemma-4-26b-a4b-it
  หรือ Claude ผ่าน Claude Code CLI (ใช้บัญชีที่ล็อกอินด้วย `claude auth login` ไม่ต้องมี API key):
    LLM_PROVIDER=claude-cli
    LLM_MODEL=claude-haiku-5-5
    CLAUDE_BIN=claude                            # path ของคำสั่ง claude ถ้าไม่ได้อยู่ใน PATH
    CONTEXT_MODE=outline | full                  # [F2] full = ส่งเอกสารเต็ม (ต้องมี quota token/นาทีพอ)
"""
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus_input"
RESULTS = ROOT / "results"

for _env in (Path(os.environ.get("RAG_ENV_FILE", ROOT / ".env")),):
    if _env.exists():
        for line in _env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.split(" #", 1)[0].strip())   # ตัด comment ท้ายบรรทัด

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")        # openai (OpenAI SDK) | claude-cli
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-haiku-5-5" if LLM_PROVIDER == "claude-cli" else "qwen/qwen-2.5-72b-instruct")
CONTEXT_MODE = os.environ.get("CONTEXT_MODE", "outline")
_slug = re.sub(r"[^A-Za-z0-9.-]+", "-", LLM_MODEL)
INDEX = "authentic-rag-fixed"
MAX_CHARS = 800                 # [F1] ขนาด chunk สูงสุด (ตัวอักษร) ≈ 250-300 token ของ bge-m3
MAX_CONTEXT_TOKENS = 3000       # [F6] งบ token ของ context ที่ส่งให้ LLM ตอนตอบ
CONTEXT_CACHE = RESULTS / f"contexts_cache.{_slug}.{CONTEXT_MODE}.json"   # แยก cache ตามโมเดลและโหมด


# ---------------------------------------------------------------- LLM / OpenSearch / embedding
_client = None


_cli_dir = None


def _claude_cli(prompt, system):
    """เรียก `claude -p` แบบไม่มี tool / MCP / skills / settings และรันในโฟลเดอร์ว่าง (ไม่อ่าน CLAUDE.md ของ repo)
    CLI ไม่รับ temperature / max_tokens จึงไม่ได้ส่งต่อ"""
    global _cli_dir
    _cli_dir = _cli_dir or tempfile.mkdtemp(prefix="rag-claude-")
    cmd = [os.environ.get("CLAUDE_BIN", "claude"), "-p", "--output-format", "json", "--model", LLM_MODEL,
           "--system-prompt", system, "--tools", "", "--strict-mcp-config", "--disable-slash-commands",
           "--setting-sources", "", "--no-session-persistence"]
    for attempt in range(5):
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=_cli_dir, timeout=600)
        try:
            out = json.loads(r.stdout)
        except json.JSONDecodeError:
            out = {"is_error": True, "result": (r.stderr or r.stdout)[:300]}
        if not out.get("is_error") and (out.get("result") or "").strip():
            return out["result"].strip()
        msg = str(out.get("result"))
        if "Not logged in" in msg:
            raise SystemExit("Claude CLI ยังไม่ได้ล็อกอิน — รัน: claude auth login")
        wait = 30 * (attempt + 1)
        print(f"  (claude-cli: {msg[:80]!r}, รอ {wait} วินาที)")
        time.sleep(wait)
    raise RuntimeError(f"claude-cli ล้มเหลวซ้ำ: {msg[:200]}")


def llm(prompt, max_tokens=4000, temperature=0.2, system="You are a helpful assistant."):
    """เรียก LLM; ชน rate limit แล้วรอตามที่ server บอก, ได้ข้อความว่าง (โมเดลใช้ token หมดกับการคิด) ก็ลองใหม่
    [F8] ล้มเหลวจริงให้ raise ไม่คืนข้อความ error กลับไปเหมือนโค้ดเดิม (ซึ่งทำให้ error ถูกเก็บลง index)"""
    if LLM_PROVIDER == "claude-cli":
        return _claude_cli(prompt, system)
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("ยังไม่ได้ตั้ง OPENROUTER_API_KEY หรือ LLM_API_KEY (ใน environment หรือไฟล์ .env)")
        # timeout 3 นาที: ค่า default 10 นาทีทำให้ request ที่ค้างรอนานเกินไป (APITimeoutError ถูกลองใหม่ด้านล่าง)
        _client = OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
                         timeout=180, max_retries=0)
    from openai import APIConnectionError, InternalServerError, RateLimitError
    for attempt in range(8):
        try:
            r = _client.chat.completions.create(
                model=LLM_MODEL, max_tokens=max_tokens, temperature=temperature,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
        except (RateLimitError, InternalServerError, APIConnectionError) as e:
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 2 if m else 30
            if wait > 300 or attempt == 7:
                raise
            print(f"  ({type(e).__name__}, รอ {wait:.0f} วินาที)")
            time.sleep(wait)
            continue
        text = re.sub(r"(?s)<thought>.*?(</thought>|$)", "", r.choices[0].message.content or "").strip()
        if text:
            return text
    raise RuntimeError("LLM คืนข้อความว่างซ้ำหลายครั้ง")


def opensearch(host="localhost", port=9200):
    from opensearchpy import OpenSearch
    return OpenSearch(hosts=[{"host": host, "port": port}], use_ssl=False, timeout=60)


_embedder = _reranker = None


def embedder():
    """BAAI/bge-m3 ผ่าน sentence-transformers (โค้ดเดิมใช้ตัวนี้ ไม่ได้ใช้ Ollama ตามที่ README เดิมบอก)"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-m3")
    return _embedder


def count_tokens(text):
    return len(embedder().tokenizer.encode(text, add_special_tokens=False))


def text_key(s):
    """key ของ cache เป็น hash เพื่อไม่ให้ไฟล์ cache มีข้อความจากบทความติดไปด้วย"""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]


def read_corpus():
    files = sorted(CORPUS.glob("*.md"))
    if not files:
        raise SystemExit("ไม่พบเอกสารใน corpus_input/ — รัน python fetch_corpus.py ก่อน")
    return {p.name: p.read_text(encoding="utf-8") for p in files}


# ---------------------------------------------------------------- [F1] Chunking
def chunk_markdown(text, source):
    """ตัดตามหัวข้อ markdown ก่อน แล้วจำกัดขนาดจริง (MAX_CHARS) + overlap 1 บรรทัด + breadcrumb
    โค้ดเดิม: MarkdownNodeParser(chunk_size=256) ไม่ใช้ chunk_size เลย -> chunk ยาวถึง 2,431 token"""
    title, chunks, stack, lines = "", [], [], []

    def flush():
        body = [l for l in lines if l.strip()]
        if not body:
            return
        crumb, buf = " > ".join(stack), []
        for line in body:
            while len(line) > MAX_CHARS:                      # บรรทัดยาวเกิน -> ตัดที่ช่องว่าง
                cut = line.rfind(" ", 0, MAX_CHARS)
                cut = cut if cut > MAX_CHARS // 2 else MAX_CHARS
                piece, line = line[:cut], line[cut:].strip()
                buf = emit(buf, piece, crumb, body)
            buf = emit(buf, line, crumb, body)
        if buf:
            chunks.append(make(buf, crumb, body))

    def emit(buf, line, crumb, body):
        if buf and sum(map(len, buf)) + len(line) > MAX_CHARS:
            chunks.append(make(buf, crumb, body))
            buf = buf[-1:] if len(buf[-1]) < 300 else []        # overlap 1 บรรทัด
        return buf + [line]

    def make(buf, crumb, body):
        return {"source": source, "title": title, "breadcrumb": crumb,
                "content": "\n".join(buf), "section": "\n".join(body)}

    for raw in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", raw)
        if m:
            flush()
            lines, level, head = [], len(m.group(1)), m.group(2).strip()
            title = title or head
            stack = stack[: level - 1] + [head]                 # breadcrumb: เอกสาร > หัวข้อ > หัวข้อย่อย
        else:
            lines.append(raw)
    flush()
    return chunks


def load_chunks():
    chunks = []
    for name, text in read_corpus().items():
        chunks += chunk_markdown(text, name)
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{c['source']}#{i}"                     # [F8] id ผูกกับไฟล์ ไม่ใช่ doc_{i} ที่ชนกันข้ามรอบ
    return chunks


# ---------------------------------------------------------------- [F2] Contextual Retrieval
CONTEXT_PROMPT = """<document_title>{title}</document_title>
<document_outline>
{outline}
</document_outline>
<section path="{crumb}">
{section}
</section>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences, in Thai) to situate this chunk within the overall document for the purposes of improving search retrieval. Mention the disease name and what aspect this chunk covers (e.g. cause, symptom, treatment), and key synonyms a reader might search with. Answer only with the succinct context and nothing else."""

CONTEXT_PROMPT_FULL = """<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Please give a short succinct context (1-2 sentences, in Thai) to situate this chunk within the overall document for the purposes of improving search retrieval. Mention the disease name and what aspect this chunk covers (e.g. cause, symptom, treatment), and key synonyms a reader might search with. Answer only with the succinct context and nothing else."""


def context_prompt(c, corpus):
    """โค้ดเดิม: <document> = chunk ตัวเอง (LLM เห็นเอกสารแค่ 5.6%) และได้ context ภาษาอังกฤษ
    แก้: outline = ชื่อเรื่อง + สารบัญ + section แม่ (พอดี free tier 16k token/นาที), full = เอกสารเต็มแบบ Anthropic"""
    if CONTEXT_MODE == "full":
        return CONTEXT_PROMPT_FULL.format(document=corpus[c["source"]], chunk=c["content"])
    outline = "\n".join(l for l in corpus[c["source"]].splitlines() if l.startswith("#"))
    return CONTEXT_PROMPT.format(title=c["title"], outline=outline, crumb=c["breadcrumb"],
                                 section=c["section"][:3000], chunk=c["content"])


def context_key(c):
    return text_key(("full|" if CONTEXT_MODE == "full" else "") + c["source"] + "|" + c["content"][:200])


def contextualize(chunks, workers=int(os.environ.get("LLM_WORKERS", 4))):
    corpus = read_corpus()
    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8")) if CONTEXT_CACHE.exists() else {}
    failed, lock = [], threading.Lock()
    RESULTS.mkdir(exist_ok=True)

    def save():
        CONTEXT_CACHE.write_text(json.dumps({k: v for k, v in cache.items() if v}, ensure_ascii=False, indent=1),
                                 encoding="utf-8")

    def one(c):
        key = context_key(c)
        if not cache.get(key):
            try:
                text = llm(context_prompt(c, corpus), temperature=0.1)
            except Exception as e:                                  # [F8] ไม่เก็บข้อความ error ลง index
                failed.append((c["chunk_id"], str(e)[:120]))
                text = None
            with lock:
                cache[key] = text
                save()                                              # บันทึกทุกครั้ง หยุดกลางทางแล้วรันต่อได้
        c["context"] = cache[key] or ""

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(one, chunks), 1):
            print(f"  context {i}/{len(chunks)}", end="\r")
    save()
    print()
    if failed:
        print(f"  สร้าง context ไม่สำเร็จ {len(failed)} chunk (index โดยไม่มี context, รันซ้ำเพื่อลองใหม่): {failed[:3]}")
    return chunks


# ---------------------------------------------------------------- [F3][F4] Index
def create_index(client, name=INDEX):
    if client.indices.exists(index=name):
        client.indices.delete(index=name)
    thai = {"type": "text", "analyzer": "thai"}                    # [F3] ตัดคำไทยด้วยพจนานุกรม (เดิม standard)
    client.indices.create(index=name, body={
        "settings": {"index.knn": True},
        "mappings": {"properties": {
            # [F4] ระบุ method/space_type ที่ field ตรง ๆ (เดิมพึ่ง index.knn.space_type ระดับ settings)
            "embedding": {"type": "knn_vector", "dimension": 1024,
                          "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
            "content": thai, "context": thai, "breadcrumb": thai,
            "source": {"type": "keyword"}, "chunk_id": {"type": "keyword"}}}})


def ingest(client=None):
    client = client or opensearch()
    chunks = load_chunks()
    print(f"{len(chunks)} chunks จาก {len(read_corpus())} ไฟล์")
    contextualize(chunks)
    create_index(client)
    texts = [f"{c['breadcrumb']}\n{c['context']}\n{c['content']}" for c in chunks]   # embed หัวข้อ + context + เนื้อหา
    vecs = embedder().encode(texts, batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    from opensearchpy import helpers
    ok, errors = helpers.bulk(client, ({"_index": INDEX, "_id": c["chunk_id"], "embedding": v.tolist(),
                                        **{k: c[k] for k in ("chunk_id", "source", "breadcrumb", "content", "context")}}
                                       for c, v in zip(chunks, vecs)), raise_on_error=False)
    client.indices.refresh(index=INDEX)
    print(f"index {ok} chunk ลง {INDEX}" + (f", error {len(errors)}" if errors else ""))
    return chunks


# ---------------------------------------------------------------- [F5] Search
def bm25(q, k=20, client=None):
    r = (client or opensearch()).search(index=INDEX, body={"size": k, "_source": {"excludes": ["embedding"]}, "query": {
        "multi_match": {"query": q, "fields": ["content", "context", "breadcrumb^2"], "type": "most_fields"}}})
    return [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]


def dense(q, k=20, client=None):
    v = embedder().encode(q, normalize_embeddings=True).tolist()
    r = (client or opensearch()).search(index=INDEX, body={"size": k, "_source": {"excludes": ["embedding"]},
                                                           "query": {"knn": {"embedding": {"vector": v, "k": k}}}})
    return [(h["_id"], h["_source"]) for h in r["hits"]["hits"]]


def rrf(rankings, k=60):
    """RRF สูตรมาตรฐาน rank เริ่ม 1 (เดิมเริ่ม 0) และ candidate ฝั่งละ 20 (เดิม 10)"""
    scores, docs = {}, {}
    for ranking in rankings:
        for rank, (doc_id, src) in enumerate(ranking, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            docs[doc_id] = src
    return [(d, docs[d]) for d in sorted(scores, key=scores.get, reverse=True)]


def rerank(q, cands):
    """cross-encoder อ่าน (คำถาม, chunk) คู่กันแล้วจัดอันดับใหม่"""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    scores = _reranker.predict([(q, f"{s['breadcrumb']}\n{s['content']}") for _, s in cands])
    return [c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])]


def search(q, mode="rerank", k=5, client=None):
    """mode: bm25 | dense | hybrid | rerank (ค่าเริ่มต้น)"""
    if mode == "bm25":
        return bm25(q, client=client)[:k]
    if mode == "dense":
        return dense(q, client=client)[:k]
    fused = rrf([bm25(q, client=client), dense(q, client=client)])
    return fused[:k] if mode == "hybrid" else rerank(q, fused[:20])[:k]


# ---------------------------------------------------------------- [F6][F7] Answer
ANSWER_PROMPT = """You are a question answering assistant for Thai medical articles. Answer in Thai using ONLY the context below.
Cite sources like [1]. Do not add facts from outside the context.
If the context does not contain the answer, say clearly that the documents do not contain this information instead of guessing.

Context:
{context}

Question: {question}"""


def build_context(hits, max_tokens=MAX_CONTEXT_TOKENS):
    """[F6] นับ token จริงด้วย tokenizer (เดิมนับคำด้วยช่องว่าง x1.3 ซึ่งนับไทยต่ำไป 4.2 เท่า)
    ใส่ chunk ทีละตัวจนเต็มงบ ไม่ตัดกลาง chunk และคงการขึ้นบรรทัดไว้"""
    parts, used = [], 0
    for i, (_, s) in enumerate(hits, 1):
        part = f"[{i}] แหล่ง: {s['source']} > {s['breadcrumb']}\nบริบท: {s.get('context', '')}\n{s['content']}"
        n = count_tokens(part)
        if parts and used + n > max_tokens:
            break
        parts.append(part)
        used += n
    return "\n\n".join(parts)


def answer(q, mode="rerank", k=5, client=None):
    hits = search(q, mode, k, client)
    if not hits:
        return hits, "", "ไม่พบข้อมูลที่เกี่ยวข้องในเอกสาร"
    ctx = build_context(hits)
    return hits, ctx, llm(ANSWER_PROMPT.format(context=ctx, question=q), temperature=0.3)   # [F7]
