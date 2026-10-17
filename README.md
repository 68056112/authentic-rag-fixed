# Authentic RAG — ฉบับแก้ไข

ระบบ RAG ภาษาไทยที่แก้จาก [aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG)
(`authenticRAG.py` + `onlysearchAuthenticRAG.py`) โดยวัดผลทีละขั้นก่อนแก้ และวัดซ้ำหลังแก้

- **รายงาน (ปัญหาที่เจอ, วิธีแก้, ผลวัด):** [REPORT.md](REPORT.md)
- **ตารางวัดผลที่สคริปต์สร้าง:** [results/metrics.md](results/metrics.md)

## สิ่งที่ต้องมี

- Python 3.10 ขึ้นไป
- OpenSearch 2.19.1 ที่ `localhost:9200` เช่น

  ```bash
  docker run -d --name opensearch -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:2.19.1
  ```

- API key ของ LLM ที่รองรับ OpenAI SDK: OpenRouter (Qwen แบบโค้ดเดิม) หรือ Gemini API (Gemma 4)
- ไม่ต้องใช้ Ollama: embedding ใช้ `BAAI/bge-m3` ผ่าน sentence-transformers (โหลดครั้งแรกประมาณ 2 GB) และ reranker ใช้ `BAAI/bge-reranker-v2-m3`

## ติดตั้งและใช้งาน

```bash
pip install -r requirements.txt
```

```bash
python fetch_corpus.py
```

```bash
cp .env.example .env
```

`fetch_corpus.py` ดาวน์โหลดบทความ 4 เรื่องจาก repo ต้นฉบับ (ล็อก commit และตรวจ sha256) เพราะ repo นี้ไม่แจกจ่ายเนื้อหาของเว็บเมดไทยซ้ำ

แก้ไฟล์ `.env` ใส่ key ของตัวเอง แล้วรัน:

```bash
python authenticRAG.py
```

```bash
python onlysearchAuthenticRAG.py "ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน"
```

- `python authenticRAG.py` นำเอกสารเข้า index ชื่อ `authentic-rag-fixed` โดย context ที่ LLM สร้างจะ cache ไว้ใน `results/contexts_cache.json` ซึ่ง repo มี cache ของเอกสาร 4 ไฟล์มาให้แล้ว ถ้าไม่ได้เปลี่ยนเอกสารก็ไม่ต้องเรียก LLM เลย
- `python onlysearchAuthenticRAG.py` (ไม่ใส่คำถาม) จะตอบชุดคำถามเดิม 5 ข้อ แล้วบันทึกลง `authentic_rag_search_results.json`
- ใส่โหมดค้นเป็นอาร์กิวเมนต์ที่สองได้: `bm25`, `dense`, `hybrid` หรือ `rerank` (ค่าเริ่มต้น)

## วัดผล

```bash
python eval/evaluate.py
```

- จำลองระบบเดิมใน index `baseline-*` (`eval/baseline.py`) แล้ววัดเทียบกับฉบับแก้ไขด้วยชุดคำถาม `eval/questions.json` (29 ข้อ)
- เขียนผลลง `results/metrics.md` และ `results/metrics.json`
- ผลที่ได้จาก LLM ทุกครั้ง cache ไว้ใน `results/llm_cache.json` รันซ้ำได้โดยไม่ต้องเรียก LLM ใหม่
- ใส่ `--no-llm` เพื่อข้ามขั้นตอบคำถามและให้คะแนน

## โครงสร้าง

```
rag_core.py                 ทุกขั้นของระบบ มีหมายเลข [F1]..[F8] ตรงกับ REPORT.md
authenticRAG.py             นำเอกสารเข้า index
onlysearchAuthenticRAG.py   ค้นและตอบคำถาม
fetch_corpus.py             ดาวน์โหลดบทความ 4 เรื่อง (หัดเยอรมัน, อหิวาตกโรค, ต้อกระจก, กรดไหลย้อน) ลง corpus_input/
eval/baseline.py            จำลองระบบเดิมพร้อมข้อผิดพลาดเดิมทุกจุด (ไว้วัดเทียบ ห้ามใช้งานจริง)
eval/evaluate.py            วัดผลทีละขั้น
eval/questions.json         ชุดคำถาม + evidence + คำตอบอ้างอิง
results/                    ผลวัดและ cache
```
