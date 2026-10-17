"""ดาวน์โหลดบทความ 4 เรื่องจาก repo ต้นฉบับ (aekanun2020/2025-authenticRAG) มาไว้ที่ corpus_input/

    python fetch_corpus.py

repo นี้ไม่แจกจ่ายบทความซ้ำ (เนื้อหาจากเว็บเมดไทย) — ล็อก commit และ sha256 ไว้ เพื่อให้ผลวัดทำซ้ำได้ตรงกับ REPORT.md
"""
import hashlib
import urllib.request
from pathlib import Path

COMMIT = "4567d3d5d377f6e38ac18afd62133476425f98bf"
FILES = {
    "1.md": "98d477174ea02b4768094510490e744d21326be7a3f5da0f39a21c6eaf3ef22f",
    "2.md": "95950a88253488d6f79d8ff3249a4137ea2dfac90fa6b09bbf9a3cd06515b517",
    "44.md": "ec67f8e6724a3e5ec63569f604065aa504555864cffab851be34f25b72778313",
    "5555.md": "d5a7db4966b091311b000c07e86035cc46b8ac1a9bc69e11a0e5567301df562c",
}
OUT = Path(__file__).resolve().parent / "corpus_input"

OUT.mkdir(exist_ok=True)
for name, sha in FILES.items():
    url = f"https://raw.githubusercontent.com/aekanun2020/2025-authenticRAG/{COMMIT}/corpus_input/{name}"
    data = urllib.request.urlopen(url, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{name}: sha256 ไม่ตรง — ไฟล์ต้นฉบับอาจถูกแก้")
    (OUT / name).write_bytes(data)
    print(f"{name}  {len(data):,} bytes  OK")
