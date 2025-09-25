import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import json
from typing import List, Dict, Tuple
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Inicializa el generador de embeddings
        
        Args:
            model_name: Nombre del modelo de SentenceTransformers a usar
        """
        self.model_name = model_name
        self.model = None
        self.dimension = None
        
    def load_model(self):
        """Carga el modelo de embeddings"""
        logger.info(f"Cargando modelo: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Modelo cargado. Dimensión de embeddings: {self.dimension}")
        return self
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos
        
        Args:
            texts: Lista de textos a convertir en embeddings
            
        Returns:
            Array numpy con los embeddings
        """
        if self.model is None:
            self.load_model()
            
        logger.info(f"Generando embeddings para {len(texts)} textos...")
        start_time = time.time()
        
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=32
        )
        
        elapsed = time.time() - start_time
        logger.info(f"Embeddings generados en {elapsed:.2f} segundos")
        
        return embeddings

class FAISSIndexBuilder:
    def __init__(self, dimension: int):
        """
        Inicializa el constructor de índice FAISS
        
        Args:
            dimension: Dimensión de los embeddings
        """
        self.dimension = dimension
        self.index = None
        
    def build_index(self, embeddings: np.ndarray, index_type: str = "FlatL2") -> faiss.Index:
        """
        Construye el índice FAISS
        
        Args:
            embeddings: Array numpy con los embeddings
            index_type: Tipo de índice FAISS ("FlatL2" o "FlatIP")
            
        Returns:
            Índice FAISS
        """
        logger.info(f"Construyendo índice FAISS {index_type}...")
        
        if index_type == "FlatL2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "FlatIP":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Tipo de índice no soportado: {index_type}")
        
        if index_type == "FlatIP":
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Índice construido con {self.index.ntotal} vectores")
        return self.index
    
    def save_index(self, index_path: Path, metadata: List[Dict] = None):
        """
        Guarda el índice FAISS y metadatos
        
        Args:
            index_path: Ruta donde guardar el índice
            metadata: Metadatos opcionales para guardar
        """
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Índice guardado en: {index_path}")
        
        if metadata is not None:
            metadata_path = index_path.parent / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"Metadatos guardados en: {metadata_path}")

def load_chunks_data(chunks_path: Path) -> Tuple[List[str], List[Dict]]:
    """
    Carga los chunks y metadatos desde el archivo parquet
    
    Args:
        chunks_path: Ruta al archivo parquet con los chunks
        
    Returns:
        Tupla con (textos, metadatos)
    """
    logger.info(f"Cargando chunks desde: {chunks_path}")
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {chunks_path}")
    
    df = pd.read_parquet(chunks_path)
    
    texts = df['text'].tolist()
    
    metadata = []
    for _, row in df.iterrows():
        metadata.append({
            'chunk_id': row['chunk_id'],
            'doc_id': row['doc_id'],
            'title': row['title'],
            'page': row['page'],
            'url': row['url'],
            'vigencia': row['vigencia'],
            'filename': row['filename']
        })
    
    logger.info(f"Cargados {len(texts)} chunks con sus metadatos")
    return texts, metadata

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar embeddings y construir índice FAISS')
    parser.add_argument('--chunks-path', type=Path, default='data/processed/chunks.parquet',
                       help='Ruta al archivo parquet con los chunks')
    parser.add_argument('--output-dir', type=Path, default='data/processed',
                       help='Directorio donde guardar el índice y metadatos')
    parser.add_argument('--model-name', type=str, default='all-MiniLM-L6-v2',
                       help='Nombre del modelo de SentenceTransformers a usar')
    parser.add_argument('--index-type', type=str, default='FlatL2',
                       choices=['FlatL2', 'FlatIP'],
                       help='Tipo de índice FAISS a construir')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        texts, metadata = load_chunks_data(args.chunks_path)
        
        embedder = EmbeddingGenerator(args.model_name)
        embeddings = embedder.generate_embeddings(texts)
        
        index_builder = FAISSIndexBuilder(embeddings.shape[1])
        index = index_builder.build_index(embeddings, args.index_type)
        
        index_path = args.output_dir / "index.faiss"
        index_builder.save_index(index_path, metadata)
        
        logger.info("Proceso H3 (Embeddings & FAISS) completado exitosamente")
        
    except Exception as e:
        logger.error(f"Error en el proceso: {str(e)}")
        raise

if __name__ == "__main__":
    main()