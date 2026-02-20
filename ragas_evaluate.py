import json
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import llm_factory
from ragas.run_config import RunConfig
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI as OpenAIClient

# Initialize embeddings
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# RAGAS judge LLM config (stability + avoid truncation)
run_config = RunConfig(
    timeout=300,
    max_retries=10,
    max_wait=60,
    max_workers=4,
    seed=42,
)

# IMPORTANT: RAGAS expects an instance of `openai.OpenAI` / `openai.AsyncOpenAI`
judge_client = OpenAIClient()

# Use a strong judge model with deterministic decoding and a higher output budget
ragas_llm = llm_factory(
    "gpt-4.1",
    client=judge_client,
    temperature=1e-8,
    max_tokens=4096,
    n=1,
)

# Bind the judge LLM to the metrics so the evaluation is consistent across runs
faithfulness.llm = ragas_llm
answer_relevancy.llm = ragas_llm

# Paths
LOG_PATH = "rag_logs.jsonl"
CSV_PATH = "Groundtruth.csv"



# Note: RAGAS metrics used:
# - faithfulness: answer supported by contexts
# - answer_relevancy: answer addresses the question


# ============================================================================
# Helper Functions
# ============================================================================

def load_ground_truths_from_csv(path: str) -> Dict[str, str]:
    """
    Load ground truths from CSV file and return as dict keyed by question.
    
    Returns:
        Dict mapping question text to ground_truth
    """
    try:
        df = pd.read_csv(path)
        # Create mapping: query -> groundtruth
        ground_truth_map = {}
        for _, row in df.iterrows():
            question = str(row["query"]).strip()
            ground_truth = str(row["groundtruth"]).strip()
            ground_truth_map[question] = ground_truth
        return ground_truth_map
    except Exception as e:
        print(f"  No se pudo cargar ground_truths desde CSV: {e}")
        print("   Continuando sin ground_truths...")
        return {}


def cargar_logs(ground_truth_map: Dict[str, str]):
    """Load interaction logs from JSONL file and match with ground truths."""
    datos = []
    error_count = 0
    
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            item = json.loads(linea.strip())
            question = item["question"]
            
            # Parse answer from JSON string
            answer_text = item["answer"]
            try:
                if isinstance(answer_text, str) and answer_text.startswith("{"):
                    answer_json = json.loads(answer_text)
                    answer_text = answer_json.get("answer", answer_text)
            except json.JSONDecodeError:
                pass  # Keep original if not JSON
            
            # Skip error entries
            if item.get("error") or answer_text.startswith("ERROR:"):
                error_count += 1
                print(f"   Saltando entrada con error: {question[:60]}...")
                continue
            
            # Match ground_truth from CSV by question text
            ground_truth = ground_truth_map.get(question, "")
            
            if not ground_truth:
                print(f"   Sin ground_truth para: {question[:60]}...")
            
            datos.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "contexts": item["contexts"],
                    "ground_truth": ground_truth,
                }
            )
    
    if error_count > 0:
        print(f"\n   Se encontraron {error_count} entradas con errores (no se evaluarán)\n")
    
    return datos


def main():
    print("=" * 80)
    print("RAG EVALUATION WITH RAGAS METRICS")
    print("=" * 80)
    print(f"\nCargando ground truths desde: {CSV_PATH}")
    
    # Load ground truths from CSV
    ground_truth_map = load_ground_truths_from_csv(CSV_PATH)
    print(f"Cargados {len(ground_truth_map)} ground truths\n")
    
    print(f"Cargando registros desde: {LOG_PATH}\n")
    
    registros = cargar_logs(ground_truth_map)

    if len(registros) == 0:
        print("No se encontraron logs. Ejecuta el sistema RAG primero.")
        return

    print(f"Cargados {len(registros)} registros\n")

    # ========================================================================
    # Part 1: RAGAS Built-in Metrics (NO context metrics)
    # ========================================================================
    print("-" * 80)
    print("PARTE 1: Métricas RAGAS Estándar (sin context metrics)")
    print("-" * 80)
    
    ds = Dataset.from_list(registros)
    
    print("\nEjecutando evaluación RAGAS...\n")
    
    # Use 3 metrics in evaluate()
    resultado_ragas = evaluate(
        dataset=ds,
        metrics=[faithfulness, answer_relevancy],
        embeddings=embedding,
        run_config=run_config,
    )

    print("\nResultados RAGAS:")
    print(resultado_ragas)

    # ========================================================================
    # Part 2: Summary Statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("RESUMEN DE MÉTRICAS")
    print("=" * 80)
    
    print("\n RAGAS Metrics:")
    # EvaluationResult from RAGAS 0.4+ - print full object to see structure
    print(f"   {resultado_ragas}")
    
    # ========================================================================
    # Part 3: Save Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("GUARDANDO RESULTADOS")
    print("=" * 80)
    
    from datetime import datetime
    summary_file = f"ragas_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAGAS EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total registros evaluados: {len(registros)}\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("MÉTRICAS RAGAS\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"{resultado_ragas}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("EXPLICACIÓN DE MÉTRICAS\n")
        f.write("-" * 80 + "\n\n")
        
        f.write("""
faithfulness (0-1):
  Mide si la respuesta está apoyada en los contextos recuperados.
  Valor alto = respuesta fiel a los documentos (sin alucinaciones).

answer_relevancy (0-1):
  Mide si la respuesta aborda la pregunta correctamente.
  Valor alto = respuesta relevante y completa.
        """)
        
    
    print(f"\n✓ Resumen guardado en: {summary_file}\n")
    



if __name__ == "__main__":
    main()