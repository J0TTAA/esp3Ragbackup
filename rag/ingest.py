import argparse
from pathlib import Path
import re
import pandas as pd
from pypdf import PdfReader
from bs4 import BeautifulSoup
import csv

def clean_text(txt: str) -> str:
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n{3,}', '\n\n', txt)
    return txt.strip()

def chunks_by_words(text: str, chunk_size:int=900, overlap:int=120):
    words = text.split()
    res = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        res.append(" ".join(words[i:i+chunk_size]))
        i += step
    return res

def extract_pdf_text(path: Path):
    reader = PdfReader(str(path))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

def extract_html_text(path:Path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    return [soup.get_text("\n")]

def load_sources(path:Path):
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = (r.get("filename") or "").strip()
            if fname:
                out[fname] = r
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--sources", default="data/sources.csv")
    ap.add_argument("--out", default="data/processed/chunks.parquet")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    raw = Path(args.raw)
    sources = load_sources(Path(args.sources))
    records = []
    for fp in sorted(raw.glob("*")):
        if not fp.is_file(): continue
        ext = fp.suffix.lower()
        print("Procesando:", fp.name)
        try:
            if ext == ".pdf":
                pages = extract_pdf_text(fp)
            elif ext in (".html",".htm"):
                pages = extract_html_text(fp)
            else:
                continue
        except Exception as e:
            print("Error extrayendo:", e)
            continue

        meta = sources.get(fp.name, {})
        doc_id = meta.get("doc_id", fp.stem)
        title = meta.get("title", fp.stem)
        url = meta.get("url","")
        vigencia = meta.get("vigencia","")

        for pno, ptext in enumerate(pages, start=1):
            text = clean_text(ptext)
            if not text: continue
            chs = chunks_by_words(text, args.chunk_size, args.overlap)
            for i,ch in enumerate(chs):
                records.append({
                    "chunk_id": f"{doc_id}_p{pno}_c{i}",
                    "doc_id": doc_id,
                    "title": title,
                    "page": pno,
                    "url": url,
                    "vigencia": vigencia,
                    "text": ch,
                    "filename": fp.name
                })

    df = pd.DataFrame.from_records(records)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        print("No se generaron chunks.")
    else:
        df.to_parquet(outp, index=False)
        print(f"Guardado {len(df)} chunks en {outp}")

if __name__ == "__main__":
    main()
