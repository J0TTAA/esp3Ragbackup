import os
import time
from dotenv import load_dotenv
import argparse
from typing import List, Tuple, Dict, Any

# Importar los componentes que creaste
from rag.retrieve import QdrantRetriever
from providers.deepseek import DeepSeekProvider
from providers.openrouter import OpenRouterProvider

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# -------------------------------
# Inicialización global (1 sola vez)
# -------------------------------
retriever = QdrantRetriever()
deepseek_provider = DeepSeekProvider()
openrouter_provider = OpenRouterProvider()

# Definimos el tipo para los metadatos de citación
CitationMetadata = Dict[str, Any]


# MODIFICACIÓN CLAVE: AGREGAR 'k: int = 4' a la firma de la función.
def rag_pipeline(query: str, provider: str = "openrouter", k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """
    Ejecuta el pipeline RAG completo para una consulta de usuario.

    Args:
        query: La pregunta del usuario.
        provider: El nombre del proveedor LLM a usar ("deepseek" u "openrouter").
        k: Número de fragmentos a recuperar.

    Returns:
        tuple: (
            generated_answer: str,
            retrieved_texts: list[str],
            citation_metadata: list[dict],
            tokens_used: int
        )
    """
    # Seleccionar el proveedor ya inicializado
    if provider == "deepseek":
        llm = deepseek_provider
    else:
        llm = openrouter_provider

    # Paso de Recuperación (Retrieval)
    chunks = retriever.retrieve(query, k=k)  # Se pasa 'k' al retriever

    if not chunks:
        # DEVOLVEMOS LISTA VACÍA DE CITACIONES EN CASO DE NO ENCONTRAR NADA
        return (
            "No pude encontrar información relevante en la base de datos para responder a tu pregunta.",
            [],
            [],  # CITACIONES VACÍAS
            0,
        )

    # 1. Extraer el contenido de texto para el prompt
    retrieved_texts = [chunk["text"] for chunk in chunks]

    # 2. LÓGICA DE CITACIÓN: Extraer y filtrar metadatos de citación (para evitar duplicados)
    unique_citations_set = set()
    citation_metadata: List[CitationMetadata] = []

    for chunk in chunks:
        source_key = (chunk.get("title"), chunk.get("page"), chunk.get("url"))

        if source_key not in unique_citations_set:
            unique_citations_set.add(source_key)
            citation_metadata.append({
                "title": chunk.get("title", "Documento Desconocido"),
                "page": chunk.get("page", "N/D"),
                "url": chunk.get("url", "#")
            })

    # Paso de Aumento de Contexto (Augmentation)
    context = "\n\n".join(retrieved_texts)

    # PROMPT MEJORADO: Política de Abstención más clara
    augmented_prompt = (
        f"Basado EXCLUSIVAMENTE en la siguiente información de la normativa de la UFRO, responde la pregunta del usuario. "
        f"Si la información no es suficiente o no permite una respuesta completa, indica claramente que no puedes responderla con el contexto proporcionado (Política de Abstención).\n\n"
        f"### Contexto:\n{context}\n\n### Pregunta del usuario:\n{query}"
    )

    # Paso de Generación (Generation)
    response = llm.chat(messages=[{"role": "user", "content": augmented_prompt}])

    # Estimación de tokens (puedes reemplazarlo con tiktoken si usas OpenAI-compatible)
    tokens_used = len(augmented_prompt.split()) + len(response.split())

    # DEVOLVEMOS LOS METADATOS DE CITACIÓN
    return response, retrieved_texts, citation_metadata, tokens_used


# MODIFICAMOS LAS FIRMAS DE LAS ENVOLTURAS
def call_rag_chatgpt(query: str, k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """Función de envoltura para llamar al pipeline RAG con el proveedor OpenRouter (ChatGPT)."""
    return rag_pipeline(query=query, provider="openrouter", k=k)


def call_rag_deepseek(query: str, k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """Función de envoltura para llamar al pipeline RAG con el proveedor DeepSeek."""
    return rag_pipeline(query=query, provider="deepseek", k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline RAG de la UFRO.")
    parser.add_argument("query", type=str, help="La pregunta del usuario a procesar.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        help="El proveedor LLM a usar ('openrouter' o 'deepseek').",
    )
    # Manejo de k
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Número de fragmentos (chunks) a recuperar de la base de datos (top-k).",
    )

    args = parser.parse_args()

    print("--- 1. Inicializando componentes RAG ---")
    start_time = time.time()

    # ACTUALIZAMOS EL LLAMADO Y DESEMPAQUE DE LA TUPLA
    final_response, retrieved_texts, citations, tokens = rag_pipeline(
        query=args.query, provider=args.provider, k=args.k  # Se pasa el valor de 'k'
    )

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    print("\n--- ¡Proceso Completado! ---")
    print("\nRespuesta Final:")
    print(final_response)

    # LÓGICA DE CITAS: Impresión de referencias en el formato requerido
    if citations:
        print("\n### Referencias:")
        for citation in citations:
            # Formato requerido: [Documento, p.xx] e ID/URL
            print(f"  - [{citation['title']}, p.{citation['page']}] (URL: {citation['url']})")
    else:
        print("\n### Referencias: No se encontraron fuentes.")

    # ACTUALIZAMOS LAS MÉTRICAS FINALES
    print(f"\nModelo usado: {args.provider.upper()}")
    print(f"Fragmentos recuperados (k): {len(retrieved_texts)}")
    print(f"Tokens usados (estimado): {tokens}")
    print(f"Latencia: {latency_ms:.2f} ms")
