import sys
import os
import csv
import time
import math
from ragas import evaluate
from datasets import Dataset
# Importa las métricas correctas
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from concurrent.futures import ThreadPoolExecutor
import functools

# 🔑 Nuevos imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Asegurarse de que el directorio padre esté en el camino de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa tus funciones RAG
from app import call_rag_chatgpt, call_rag_deepseek

# --- Cargar variables de entorno ---
load_dotenv()

# --- Configuración de LLM para Ragas ---
if os.getenv("OPENAI_API_KEY"):
    ragas_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
elif os.getenv("OPENROUTER_API_KEY"):
    ragas_llm = ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        temperature=0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
else:
    raise ValueError("❌ No encontré ni OPENAI_API_KEY ni OPENROUTER_API_KEY en tu .env")

# --- Configuración de Embeddings para Ragas (gratis con HuggingFace) ---
ragas_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def _process_single_query(row: dict, model_name: str, call_function):
    query = row['query']
    ground_truth = row['ground_truth']

    start_time = time.time()
    try:
        generated_answer, retrieved_sources_list, tokens_used = call_function(query)
        end_time = time.time()
        latency = end_time - start_time
    except Exception as e:
        print(f"Error al procesar la pregunta '{query}': {e}")
        generated_answer, retrieved_sources_list, latency, tokens_used = "Error en la generación.", [], 0.0, 0
    
    return {
        'question': query,
        'answer': generated_answer,
        'contexts': retrieved_sources_list,
        'ground_truth': ground_truth,
        'latency': latency,
        'tokens_used': tokens_used
    }

def evaluate_rag_model(model_name: str, test_set_path: str):
    print(f"\n--- Iniciando evaluación para el modelo: {model_name} ---")

    # 1. Cargar el conjunto de datos
    print("Cargando el conjunto de datos de prueba...")
    if not os.path.exists(test_set_path):
        print(f"Error: El archivo '{test_set_path}' no se encuentra.")
        return

    with open(test_set_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 2. Procesar concurrentemente
    print(f"Procesando {len(data)} preguntas de forma concurrente...")
    call_function = call_rag_chatgpt if model_name == 'ChatGPT' else call_rag_deepseek
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        func = functools.partial(_process_single_query, model_name=model_name, call_function=call_function)
        results = list(executor.map(func, data))

    # 3. Consolidar resultados
    questions = [res['question'] for res in results]
    ground_truths = [res['ground_truth'] for res in results]
    latencies = [res['latency'] for res in results]
    generated_answers = [res['answer'] for res in results]
    retrieved_sources = [res['contexts'] for res in results]
    tokens_used_list = [res['tokens_used'] for res in results]

    dataset = Dataset.from_dict({
        'question': questions,
        'answer': generated_answers,
        'contexts': retrieved_sources,
        'ground_truth': ground_truths
    })

    # 4. Evaluar con ragas
    print("\nEvaluando métricas Ragas (puede tomar unos minutos)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings   # ✅ Ahora usa embeddings de HuggingFace
    )

    df = result.to_pandas()

    # 5. Métricas adicionales
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_tokens = sum(tokens_used_list)
    avg_tokens = total_tokens / len(tokens_used_list) if tokens_used_list else 0

    # 6. Mostrar resultados
    print("\n--- Resultados de la Evaluación ---")
    print(f"Métricas del Modelo {model_name}:")
    print(f"   Puntaje de Fidelidad: {df['faithfulness'].mean():.2f}")
    print(f"   Puntaje de Relevancia de la Respuesta: {df['answer_relevancy'].mean():.2f}")
    print(f"   Puntaje de Precisión del Contexto: {df['context_precision'].mean():.2f}")
    print(f"   Puntaje de Recall del Contexto: {df['context_recall'].mean():.2f}")
    print(f"   Latencia Promedio: {avg_latency:.2f} segundos")
    print(f"   Tokens Totales Usados: {total_tokens}")
    print(f"   Tokens Promedio Usados: {math.floor(avg_tokens)}")

    output_path = f"evaluation_results_{model_name.lower()}.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResultados detallados guardados en '{output_path}'")

if __name__ == "__main__":
    gold_set_path = os.path.join("data", "gold_set.csv")
    if not os.path.exists(gold_set_path):
        print(f"Error: El archivo '{gold_set_path}' no se encuentra.")
        print("Por favor, crea este archivo o verifica la ruta.")
    else:
        evaluate_rag_model(model_name="ChatGPT", test_set_path=gold_set_path)
        evaluate_rag_model(model_name="DeepSeek", test_set_path=gold_set_path)
