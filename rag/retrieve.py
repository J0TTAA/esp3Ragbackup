# retrieve.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Cargar variables de entorno (.env)
load_dotenv()


class QdrantRetriever:
    """
    Cliente para recuperar chunks desde Qdrant usando embeddings.
    El modelo de embeddings se carga solo una vez.
    """

    def __init__(self, collection_name="ufro_normativa"):
        # Conectar a Qdrant
        self.qdrant_client = QdrantClient(
            url=os.environ.get("QDRANT_HOST"),
            api_key=os.environ.get("QDRANT_API_KEY")
        )

        # ⚡ Cargar el modelo de embeddings una sola vez
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        self.collection_name = collection_name
        print(f"[Retriever] Conectado a Qdrant en colección '{self.collection_name}'")

    def retrieve(self, query: str, k: int = 4):
        """
        Realiza búsqueda semántica en Qdrant y devuelve los chunks relevantes.
        """
        # 1. Convertir la query a vector
        query_vector = self.embedding_model.encode(query).tolist()

        # 2. Buscar en la colección de Qdrant
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            with_payload=True
        )

        # 3. Formatear resultados
        retrieved_chunks = [
            {
                "text": r.payload.get("text"),
                "score": r.score,
                "doc_id": r.payload.get("doc_id"),
                "title": r.payload.get("title"),
                "page": r.payload.get("page"),
                "url": r.payload.get("url"),
                "vigencia": r.payload.get("vigencia"),
            }
            for r in search_result
        ]

        return retrieved_chunks


# ✅ Crear una sola instancia global del retriever
retriever = QdrantRetriever()


if __name__ == "__main__":
    # Test rápido en terminal
    query = "¿Qué es el Periodo de Inactividad Académica (PIA)?"
    chunks = retriever.retrieve(query, k=3)

    if not chunks:
        print("No se encontraron resultados.")
    else:
        print("\n--- Resultados ---")
        for i, chunk in enumerate(chunks, 1):
            print(f"[{i}] {chunk['title']} (pág. {chunk['page']})")
            print(f"   Score: {chunk['score']:.4f}")
            print(f"   Texto: {chunk['text'][:120]}...\n")
