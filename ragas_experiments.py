# ragas_experiments.py
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

EXPERIMENTOS = [
    {
        "nombre": "topk_3",
        "ruta_logs": "logs/topk3.jsonl",
    },
    {
        "nombre": "topk_5",
        "ruta_logs": "logs/topk5.jsonl",
    },
    {
        "nombre": "topk_8",
        "ruta_logs": "logs/topk8.jsonl",
    },
    # {"nombre": "bge_large", "ruta_logs": "logs/bge_large.jsonl"},
]


def cargar_logs(path):
    datos = []
    with open(path, "r", encoding="utf-8") as f:
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


def evaluar_experimento(nombre, ruta_logs):
    print(f"\nEjecutando experimento: {nombre}")
    print(f"   Fichero de logs: {ruta_logs}")

    registros = cargar_logs(ruta_logs)
    if len(registros) == 0:
        print("No hay registros en este fichero. Se omite.")
        return None

    ds = Dataset.from_list(registros)

    resultado = evaluate(
        dataset=ds,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    # resultado suele ser un objeto que ya imprime un resumen,
    # pero aquí asumimos que se comporta como un dict con medias.
    print("   Evaluación completada.")
    print("   Resultado bruto:")
    print(resultado)

    return resultado


def main():
    print("Iniciando batería de experimentos con RAGAS...\n")

    resultados = {}

    for exp in EXPERIMENTOS:
        nombre = exp["nombre"]
        ruta = exp["ruta_logs"]
        res = evaluar_experimento(nombre, ruta)
        if res is not None:
            resultados[nombre] = res

    print("\nResumen comparativo (alto nivel):")
    if not resultados:
        print("No se ha podido evaluar ningún experimento.")
        return

    # Aquí simplemente los mostramos uno por uno.
    # Si quieres, puedes adaptar esto para extraer métricas concretas.
    for nombre, res in resultados.items():
        print("────────────────────────────────────────")
        print(f"Experimento: {nombre}")
        print(res)

    print("\nFin de la batería de experimentos.")
    print("   Usa estos resultados para elegir la mejor configuración de retriever.")


if __name__ == "__main__":
    main()