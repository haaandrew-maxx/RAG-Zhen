import json
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()
from ragas import evaluate
from ragas.metrics import (
    # context_precision,   #  De momento los quitamos
    # context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.embeddings import HuggingfaceEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
from ragas.llms import llm_factory
llm = llm_factory("gpt-4.1", client=OpenAI())

embedding = OpenAIEmbeddings(model="text-embedding-ada-002")  
LOG_PATH = "rag_logs.jsonl"


def cargar_logs():
    datos = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            item = json.loads(linea.strip())
            datos.append(
                {
                    "question": item["question"],
                    "answer": item["answer"],
                    "contexts": item["contexts"],
                }
            )
    return datos


def main():
    print(" Cargando registros de interacción RAG desde:", LOG_PATH)
    registros = cargar_logs()

    if len(registros) == 0:
        print(" No se encontraron logs. Ejecuta el sistema RAG primero.")
        return

    ds = Dataset.from_list(registros)

    print("\nEjecutando evaluación con RAGAS...\n")

    # 🔧 De momento solo usamos métricas que NO requieren 'reference'
    resultado = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy],
        embeddings=embedding,
    )

    print("\nResultados de la evaluación:")
    print(resultado)

    print("\nExplicación de las métricas usadas ahora mismo:")
    print("""
    - faithfulness: Comprueba si la respuesta está realmente apoyada en los 
      documentos. Un valor bajo indica alucinaciones o información inventada.

    - answer_relevancy: Mide si la respuesta realmente aborda la pregunta. 
      Un valor bajo significa que la respuesta es irrelevante, incompleta o vaga.
    """)

    print("\nNota:")
    print("""
    Las métricas 'context_precision' y 'context_recall' en esta versión de RAGAS
    necesitan una columna adicional 'reference' (respuesta de referencia
    anotada a mano). Cuando tengas un conjunto de evaluación con respuestas
    de referencia, podremos añadir:

        - context_precision
        - context_recall

    y así evaluar la calidad del recuperador de forma más completa.
    """)

    print("\nEvaluación completada. Ajusta el recuperador (top-k, embeddings, "
          "tamaño de chunk) y vuelve a ejecutar para comparar mejoras.")


if __name__ == "__main__":
    main()