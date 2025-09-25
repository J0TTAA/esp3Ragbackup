#!/usr/bin/env python3
"""
Script para ejecutar el proceso de embeddings y FAISS
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path para importar módulos
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from rag.embed import main as embed_main

if __name__ == "__main__":
    embed_main()