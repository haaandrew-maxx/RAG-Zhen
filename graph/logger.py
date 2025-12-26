# graph/logger.py
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("./rag_logs.jsonl")


def log_interaction(question, documents, generation):
    """
    Guarda una interacción RAG en formato JSONL para análisis y evaluación.
    """

    # Contenidos de los chunks
    contexts = []
    # Origen de cada chunk (normalmente ruta de fichero)
    sources = []

    for doc in documents:
        # page_content es el texto, metadata suele tener "source"
        content = getattr(doc, "page_content", None)
        metadata = getattr(doc, "metadata", {}) or {}

        contexts.append(content)
        sources.append(metadata.get("source", None))

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "contexts": contexts,
        "sources": sources,
        "answer": generation,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Logger] Interacción registrada en {LOG_PATH}")