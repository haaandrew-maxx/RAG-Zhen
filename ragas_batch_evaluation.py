from dotenv import load_dotenv
load_dotenv()

import os
import json
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any

from graph.graph import app as graph_app

# ============================================================================
# Configuration
# ============================================================================
EXCEL_PATH = "epd-questions-ground_truths.xlsx"

# Rate limiting configuration
DELAY_BETWEEN_QUERIES = 2.0  # seconds between each query (adjust based on your rate limit)
MAX_RETRIES = 3  # maximum number of retries for rate limit errors
RETRY_DELAY = 10.0  # seconds to wait before retrying after rate limit error


# ============================================================================
# Helper Functions
# ============================================================================

def load_questions_from_excel(path: str) -> List[Dict[str, Any]]:
    """
    Load questions from Excel file.
    
    Expected columns:
    - id: question ID
    - question: the question text
    
    Returns:
        List of dicts with keys: id, question
    """
    print(f"Cargando preguntas desde: {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    
    df = pd.read_excel(path)
    
    # Validate required columns
    required_cols = ["id", "question"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
    
    # Convert to list of dicts
    questions = []
    for _, row in df.iterrows():
        questions.append({
            "id": row["id"],
            "question": str(row["question"]).strip(),
        })
    
    print(f"Cargadas {len(questions)} preguntas")
    return questions


def run_rag_query_mode_b(question: str, retry_count: int = 0) -> None:
    """
    Run a single question through the RAG system in Mode B.
    LangGraph will automatically log the interaction.
    
    Args:
        question: The question to ask
        retry_count: Current retry attempt (for rate limiting)
    """
    try:
        # Build initial state for Mode B (no description/BOM)
        initial_state = {
            "question": question,
            "bom": "",
            "description": "",
            "generation": "",
            "documents": [],
            "session_docs": [],
            "mode": "B",  # Force Mode B
        }
        
        # Invoke LangGraph (will automatically log to rag_logs.jsonl)
        final_state = graph_app.invoke(initial_state)
    
    except Exception as e:
        error_str = str(e).lower()
        # Check if it's a rate limit error
        if ("rate" in error_str or "quota" in error_str or "429" in error_str) and retry_count < MAX_RETRIES:
            print(f"  Rate limit hit, esperando {RETRY_DELAY}s antes de reintentar... (intento {retry_count + 1}/{MAX_RETRIES})")
            time.sleep(RETRY_DELAY)
            return run_rag_query_mode_b(question, retry_count + 1)
        else:
            # Re-raise the exception if it's not rate limit or max retries reached
            raise


def batch_execute_questions(questions: List[Dict[str, Any]]) -> int:
    """
    Run all questions through RAG system.
    LangGraph will automatically log each interaction.
    
    Args:
        questions: List of question dicts
        
    Returns:
        Number of successfully executed questions
    """
    total = len(questions)
    success_count = 0
    error_count = 0
    
    print(f"\nEjecutando {total} preguntas en Modo B...")
    print(f" Delay entre queries: {DELAY_BETWEEN_QUERIES}s")
    print(f" Max reintentos por rate limit: {MAX_RETRIES}\n")
    
    for i, q in enumerate(questions, 1):
        question_id = q["id"]
        question_text = q["question"]
        
        print(f"[{i}/{total}] ID: {question_id}")
        print(f"   Pregunta: {question_text[:80]}...")
        
        try:
            run_rag_query_mode_b(question_text)
            print(f"   ✓ Ejecutada correctamente")
            success_count += 1
            
            # Add delay between queries to avoid rate limits (except for last query)
            if i < total:
                print(f"   ⏳ Esperando {DELAY_BETWEEN_QUERIES}s...")
                time.sleep(DELAY_BETWEEN_QUERIES)
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            error_count += 1
    
    print(f"\n{'='*80}")
    print(f"RESUMEN")
    print(f"{'='*80}")
    print(f"Total procesadas: {total}")
    print(f"Exitosas: {success_count}")
    print(f"Errores: {error_count}")
    print()
    
    return success_count


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 80)
    print("RAG BATCH QUERY EXECUTION - MODE B")
    print("=" * 80 + "\n")
    
    # 1. Load questions
    try:
        questions = load_questions_from_excel(EXCEL_PATH)
    except Exception as e:
        print(f"Error cargando preguntas: {e}")
        return
    
    # 2. Execute batch queries
    success_count = batch_execute_questions(questions)
    
    # 3. Display completion message
    print("=" * 80)
    print("EJECUCIÓN COMPLETADA")
    print("=" * 80)
    print(f"\n✓ {success_count} preguntas ejecutadas exitosamente")
    print(f"\n Siguiente paso: Ejecuta 'python ragas_evaluate.py' para evaluar los resultados")
    print(f"   Los logs fueron generados automáticamente por LangGraph en rag_logs.jsonl\n")


if __name__ == "__main__":
    main()
