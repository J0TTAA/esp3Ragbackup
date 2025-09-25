import os
import csv
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_CSV = Path("data/sources.csv")

FIELDNAMES = ["filename", "doc_id", "title", "url", "vigencia"]

def slugify(filename: str) -> str:
    """Genera un doc_id a partir del nombre del archivo."""
    return filename.lower().replace(" ", "-").replace("_", "-").replace(".pdf", "")

def generate_title(filename: str) -> str:
    """Genera un título legible a partir del nombre del archivo."""
    stem = Path(filename).stem
    return stem.replace("-", " ").replace("_", " ").title()

def update_sources_csv():
    # Crear el archivo si no existe
    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0

    if not file_exists:
        with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    # Cargar los existentes
    existing_files = set()
    with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_files.add(row["filename"])

    # Buscar nuevos PDFs
    raw_files = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(".pdf")]
    new_files = [f for f in raw_files if f not in existing_files]

    if not new_files:
        print("✅ No se encontraron nuevos PDFs para agregar.")
        return

    print(f"⏳ Agregando {len(new_files)} archivos a sources.csv...")

    with OUTPUT_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)

        for filename in sorted(new_files):
            doc_id = slugify(filename)
            title = generate_title(filename)

            writer.writerow({
                "filename": filename,
                "doc_id": doc_id,
                "title": title,
                "url": "",
                "vigencia": ""
            })
            print(f"  - Añadido: {filename}")

    print("🎉 sources.csv actualizado correctamente.")

if __name__ == "__main__":
    update_sources_csv()
