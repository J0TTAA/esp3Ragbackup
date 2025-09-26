import os
import time
from dotenv import load_dotenv
import argparse
from typing import List, Tuple

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


def rag_pipeline(query: str, provider: str = "openrouter") -> Tuple[str, List[str], int]:
    """
    Ejecuta el pipeline RAG completo para una consulta de usuario.

    Args:
        query: La pregunta del usuario.
        provider: El nombre del proveedor LLM a usar ("deepseek" u "openrouter").

    Returns:
        tuple: (generated_answer: str, retrieved_sources: list, tokens_used: int)
    """
    # Seleccionar el proveedor ya inicializado
    if provider == "deepseek":
        llm = deepseek_provider
    else:
        llm = openrouter_provider

    # Paso de Recuperación (Retrieval)
    chunks = retriever.retrieve(query)

    if not chunks:
        return (
            "No pude encontrar información relevante en la base de datos para responder a tu pregunta.",
            [],
            0,
        )

    # CORRECCIÓN: Extraer el contenido de texto real de los chunks para las métricas
    retrieved_sources = [chunk["text"] for chunk in chunks]

    # Paso de Aumento de Contexto (Augmentation)
    context = "\n\n".join(retrieved_sources)
    augmented_prompt = (
        f"Basado en la siguiente información de la normativa de la UFRO, responde la pregunta del usuario. "
        f"Si la información no es suficiente, indica que no puedes responderla con el contexto proporcionado.\n\n"
        f"### Contexto:\n{context}\n\n### Pregunta del usuario:\n{query}"
    )

    # Paso de Generación (Generation)
    response = llm.chat(messages=[{"role": "user", "content": augmented_prompt}])

    # Estimación de tokens (puedes reemplazarlo con tiktoken si usas OpenAI-compatible)
    tokens_used = len(augmented_prompt.split()) + len(response.split())

    return response, retrieved_sources, tokens_used


def call_rag_chatgpt(query: str) -> Tuple[str, List[str], int]:
    """Función de envoltura para llamar al pipeline RAG con el proveedor OpenRouter (ChatGPT)."""
    return rag_pipeline(query=query, provider="openrouter")


def call_rag_deepseek(query: str) -> Tuple[str, List[str], int]:
    """Función de envoltura para llamar al pipeline RAG con el proveedor DeepSeek."""
    return rag_pipeline(query=query, provider="deepseek")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline RAG de la UFRO.")
    parser.add_argument("query", type=str, help="La pregunta del usuario a procesar.")
    parser.add_argument(
        "--provider",
        type=str,
        default="openrouter",
        help="El proveedor LLM a usar ('openrouter' o 'deepseek').",
    )

    args = parser.parse_args()

    print("--- 1. Inicializando componentes RAG ---")
    start_time = time.time()

    final_response, sources, tokens = rag_pipeline(
        query=args.query, provider=args.provider
    )

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    print("\n--- ¡Proceso Completado! ---")
    print("\nRespuesta Final:")
    print(final_response)
    print(f"\nFuentes recuperadas: {sources}")
    print(f"Tokens usados (estimado): {tokens}")
    print(f"Latencia: {latency_ms:.2f} ms")
