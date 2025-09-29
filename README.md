````markdown
## 2. Instalación y Configuración del Entorno

### 2.1 Prerrequisitos
- Python 3.9+
- Servidor Qdrant (por defecto en [http://localhost:6333](http://localhost:6333))
- Documentos PDF en la carpeta `data/`

### 2.2 Configuración de Claves API (`.env`)

Crea un archivo llamado `.env` en el directorio raíz de tu proyecto:

```bash
# Claves de Acceso a APIs
OPENAI_API_KEY="tu_clave_de_openai_para_embeddings"
DEEPSEEK_API_KEY="tu_clave_de_deepseek"
OPENROUTER_API_KEY="tu_clave_de_openrouter_para_gpt"

# Configuración de Qdrant
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="ufro_normativa"
````

```
```


### 2.3 Instalación de Dependencias

```bash
pip install -r requirements.txt
```

---

## 3. Preparación de la Base de Datos (S2: Ingesta y Retriever)

Construye el índice vectorial en Qdrant. Este paso debe ejecutarse una sola vez o cada vez que se añadan nuevos documentos.

```bash
# Asume que el script de ingesta se llama ingest.py
python ingest.py
```

---

## 4. Uso y Demo del Pipeline RAG (S3/H9 - CLI)

### 4.1 Parámetros de Ejecución

```bash
python app.py [QUERY] --provider [PROVIDER] --k [VALUE]
```

**Parámetros:**

* `[QUERY]` *(Obligatorio)*: La pregunta sobre la normativa UFRO (texto entre comillas).
* `--provider` *(Obligatorio)*: Selecciona el LLM a utilizar (`deepseek` o `openrouter`).
* `--k` *(Opcional)*: Número de fragmentos a recuperar de Qdrant.

**Ejemplos de Demo (H9):**

| Propósito            | Comando                                                                              | Criterios Cubiertos       |
| -------------------- | ------------------------------------------------------------------------------------ | ------------------------- |
| Integración GPT (S3) | `python app.py "¿Quién aprueba la política financiera?" --provider openrouter --k 4` | Citas (S3), API Dual (S3) |
| Robustez (H8)        | `python app.py "Explica la teoría de la relatividad." --provider deepseek --k 4`     | Abstención (H8)           |

---

### 4.2 Modo Batch (Generación de Reporte - S3/S4)

Para generar métricas comparativas (Latencia, Costo, Fidelidad), usa:

```bash
# Generar reporte para DeepSeek
python eval/evaluate.py --provider deepseek 

# Generar reporte para OpenRouter (ChatGPT)
python eval/evaluate.py --provider openrouter
```

---

## 5. Ética, Limitaciones y Trazabilidad (S5)

### 5.1 Principios Éticos y Limitaciones (S5)

* **Abstención Explícita (H8):** El modelo responde *"No puedo responder..."* si la información no está en el contexto recuperado.
* **Vigencia Normativa:** Solo se consideran los documentos PDF indexados al momento de la ingesta. El usuario debe verificar la URL de la cita.
* **Privacidad:** Las consultas y el contexto normativo se envían a APIs de terceros (DeepSeek, OpenRouter).

### 5.2 Trazabilidad y Citas Verificables (S3, S5)

Todas las respuestas incluyen una sección:

```markdown
### Referencias:
[Título del Documento, p.XX] (URL: URL_VÁLIDA)
```

**Ejemplo de Metadatos Indexados:**

| ID Interno (doc_id) | Título del Documento                              | URL / Ubicación de Origen                                                                          | Vigencia     |
| ------------------- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------ |
| Reglamento Pregrado | Reglamento de Régimen de Estudios de Pregrado     | [https://pregrado.ufro.cl/...pdf](https://pregrado.ufro.cl/...pdf)                                 | Versión 2023 |
| Magíster 2024       | Nuevo Reglamento General de Programas de Magíster | [https://magistercienciassociales.ufro.cl/...pdf](https://magistercienciassociales.ufro.cl/...pdf) | Versión 2024 |

---

```
```
