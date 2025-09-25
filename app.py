import os
from dotenv import load_dotenv
import argparse  # ¡Nueva importación!

# Importar los componentes que creaste
from rag.retrieve import QdrantRetriever
from providers.deepseek import DeepSeekProvider
from providers.openrouter import OpenRouterProvider

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

def rag_pipeline(query: str, provider: str = "openrouter"):
    """
    Ejecuta el pipeline RAG completo para una consulta de usuario.
    
    Args:
        query: La pregunta del usuario.
        provider: El nombre del proveedor LLM a usar ("deepseek" u "openrouter").
    """
    print("--- 1. Inicializando componentes RAG ---")
    retriever = QdrantRetriever()
    
    if provider == "deepseek":
        llm = DeepSeekProvider()
    elif provider == "openrouter":
        llm = OpenRouterProvider()
    else:
        print(f"Error: Proveedor LLM '{provider}' no soportado.")
        return "Lo siento, ese proveedor LLM no está disponible."
        
    print(f"--- 2. Buscando documentos relevantes para: '{query}' ---")
    
    # Paso de Recuperación (Retrieval)
    chunks = retriever.retrieve(query)
    
    if not chunks:
        print("No se encontraron documentos relevantes.")
        return "No pude encontrar información relevante en la base de datos para responder a tu pregunta."

    # Paso de Aumento de Contexto (Augmentation)
    context = "\n\n".join([chunk["text"] for chunk in chunks])
    
    # Generamos un prompt para el LLM con el contexto y la consulta
    augmented_prompt = f"Basado en la siguiente información de la normativa de la UFRO, responde la pregunta del usuario. Si la información no es suficiente, indica que no puedes responderla con el contexto proporcionado.\n\n### Contexto:\n{context}\n\n### Pregunta del usuario:\n{query}"
    
    print("--- 3. Generando respuesta con el LLM ---")
    
    # Paso de Generación (Generation)
    response = llm.chat(messages=[{"role": "user", "content": augmented_prompt}])
    
    print("\n--- ¡Proceso Completado! ---")
    return response

if __name__ == "__main__":
    # --- Configurar argparse para leer la pregunta desde la terminal ---
    parser = argparse.ArgumentParser(description="Ejecuta el pipeline RAG de la UFRO.")
    parser.add_argument("query", type=str, help="La pregunta del usuario a procesar.")
    
    args = parser.parse_args()
    
    final_response = rag_pipeline(query=args.query, provider="openrouter") # El proveedor se mantiene fijo aquí
    
    print("\nRespuesta Final:")
    print(final_response)