import os
import time
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
from flask import Flask, render_template, request, jsonify

# Importar los componentes que creaste (Asegúrate que las rutas sean correctas)
from rag.retrieve import QdrantRetriever
from providers.deepseek import DeepSeekProvider
from providers.openrouter import OpenRouterProvider

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# -------------------------------
# Inicialización global (DUPLICADA para independencia)
# -------------------------------
retriever = QdrantRetriever()
deepseek_provider = DeepSeekProvider()
openrouter_provider = OpenRouterProvider()

# Definimos el tipo para los metadatos de citación
CitationMetadata = Dict[str, Any]

# Inicialización de Flask
app = Flask(__name__)

# -------------------------------
# Pipeline RAG (DUPLICADO)
# -------------------------------
def rag_pipeline(query: str, provider: str = "openrouter", k: int = 4) -> Tuple[str, List[str], List[CitationMetadata], int]:
    """
    Ejecuta el pipeline RAG completo para una consulta de usuario.
    (Copia EXACTA de la función rag_pipeline de tu app.py)
    """
    # Seleccionar el proveedor ya inicializado
    if provider == "deepseek":
        llm = deepseek_provider
    else:
        llm = openrouter_provider

    # Paso de Recuperación (Retrieval)
    chunks = retriever.retrieve(query, k=k)

    if not chunks:
        # DEVOLVEMOS LISTA VACÍA DE CITACIONES EN CASO DE NO ENCONTRAR NADA
        return (
            "No pude encontrar información relevante en la base de datos para responder a tu pregunta.",
            [],
            [],
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

    # Estimación de tokens
    tokens_used = len(augmented_prompt.split()) + len(response.split())

    # DEVOLVEMOS LOS METADATOS DE CITACIÓN
    return response, retrieved_texts, citation_metadata, tokens_used
# -------------------------------


# RUTAS DE FLASK

@app.route("/", methods=["GET"])
def index():
    """Ruta para la página principal."""
    
    # Asumimos que retriever.collection_name existe o usamos un valor predeterminado
    collection_name = getattr(retriever, 'collection_name', 'ufro_normativa') 
    system_status = "Listo" # Se asume listo porque la inicialización global ya ocurrió.

    return render_template(
        "index.html",
        system_status=system_status,
        collection_name=collection_name
    )

@app.route("/api/query", methods=["POST"])
def api_query():
    """Ruta API para manejar la consulta RAG y devolver una respuesta JSON."""
    data = request.json
    query = data.get("query", "")
    provider = data.get("provider", "openrouter")
    k = data.get("k", 4)

    if not query:
        return jsonify({"error": "No se proporcionó la consulta."}), 400

    try:
        k_int = int(k)
    except ValueError:
        return jsonify({"error": "El valor de 'k' debe ser un número entero."}), 400

    start_time = time.time()
    
    # Llama a la pipeline RAG duplicada en este mismo archivo
    final_response, retrieved_texts, citations, tokens = rag_pipeline(
        query=query, 
        provider=provider, 
        k=k_int
    )
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # Prepara el resultado para la respuesta JSON
    result = {
        "answer": final_response,
        "citations": citations,
        "metrics": {
            "provider": provider.upper(),
            "k": len(retrieved_texts),
            "tokens_used": tokens,
            "latency_ms": f"{latency_ms:.2f}"
        }
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)