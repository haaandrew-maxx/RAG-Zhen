# doc_stats.py
import json
from collections import Counter, defaultdict

LOG_PATH = "rag_logs.jsonl"


def cargar_logs(path):
    registros = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            item = json.loads(linea.strip())
            registros.append(item)
    return registros


def main():
    print("Cargando logs desde:", LOG_PATH)
    registros = cargar_logs(LOG_PATH)

    if not registros:
        print("No hay registros en el fichero de logs.")
        return

    # Contadores
    hits_por_doc = Counter()
    preguntas_por_doc = defaultdict(set)

    for idx, item in enumerate(registros):
        question = item.get("question", f"q_{idx}")
        sources = item.get("sources")

        # Compatibilidad: si no hay 'sources', saltamos
        if not sources:
            continue

        for src in sources:
            if src is None:
                continue
            hits_por_doc[src] += 1
            preguntas_por_doc[src].add(question)

    if not hits_por_doc:
        print("No se encontraron 'sources' en los logs. "
              "Asegúrate de haber actualizado el logger.")
        return

    total_hits = sum(hits_por_doc.values())
    print("\nFrecuencia de uso de documentos (Document Hit Frequency):\n")
    print(f"Total de hits (chunks recuperados con 'source' válido): {total_hits}\n")

    # Ordenar de más usado a menos
    for src, count in hits_por_doc.most_common():
        num_preguntas = len(preguntas_por_doc[src])
        porcentaje = 100.0 * count / total_hits if total_hits > 0 else 0.0

        print("────────────────────────────────────────")
        print(f"Documento: {src}")
        print(f"   ➤ Hits totales: {count}")
        print(f"   ➤ Porcentaje de todos los hits: {porcentaje:.2f}%")
        print(f"   ➤ Número de preguntas distintas que lo usan: {num_preguntas}")

    print("\nAnálisis completado.")
    print("   Si los 2-3 primeros documentos concentran >60-70% de los hits,")
    print("   es muy probable que el recuperador esté sesgado hacia ellos.")


if __name__ == "__main__":
    main()