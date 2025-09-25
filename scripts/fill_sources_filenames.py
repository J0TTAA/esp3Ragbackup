#!/usr/bin/env python3
import csv
from pathlib import Path
import difflib

DATA_DIR = Path("data/raw")
SOURCES = Path("data/sources.csv")

def load_files():
    return [p.name for p in DATA_DIR.glob("*") if p.suffix.lower() in (".pdf", ".html", ".htm")]

def best_match(key, candidates):
    if not key: return (None, 0.0)
    scores = [(c, difflib.SequenceMatcher(None, key.lower(), c.lower()).ratio()) for c in candidates]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0] if scores else (None, 0.0)

def main():
    files = load_files()
    print("Archivos detectados en data/raw/:", files)
    rows = []
    
    # Definir fieldnames por defecto si el archivo está vacío
    default_fieldnames = ["doc_id", "title", "filename", "description"]
    
    if SOURCES.exists() and SOURCES.stat().st_size > 0:
        with SOURCES.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or default_fieldnames
            for r in reader:
                rows.append(r)
    else:
        # Si el archivo está vacío o no existe, usar fieldnames por defecto
        fieldnames = default_fieldnames
        print("Archivo sources.csv vacío o no existe. Usando estructura por defecto.")

    updated = []
    for r in rows:
        if r.get("filename"):
            updated.append(r); continue
        key = r.get("doc_id") or r.get("title") or ""
        cand, score = best_match(key, files)
        if score >= 0.4:
            print(f"[MATCH] {key} -> {cand} (score={score:.2f})")
            r["filename"] = cand
        else:
            print(f"[NO MATCH] {key} (dejar filename vacío para rellenar manualmente)")
            r["filename"] = r.get("filename","")
        updated.append(r)

    # backup
    bak = SOURCES.with_suffix(".bak.csv")
    SOURCES.replace(bak)
    with SOURCES.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)
    print("Actualizado data/sources.csv (backup:", bak, ")")

if __name__ == "__main__":
    main()
