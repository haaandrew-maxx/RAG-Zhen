"""
LCA Analysis Node: Step 3 - Production Process Analysis
Evaluates sustainable process alternatives
"""

from typing import Any, Dict
from graph.state import GraphState
from graph.chains.lca_analysis import step3_chain
import json


def analisis_procesos(state: GraphState) -> Dict[str, Any]:
    """
    PASO 3: Análisis de procesos productivos
    
    For each production process:
    - Evaluates alternative sustainable processes
    - Considers energy consumption
    - Assesses renewable electricity use
    - Identifies process optimization opportunities
    - Indicates expected environmental improvement level
    """
    print("---PASO 3: ANÁLISIS DE PROCESOS PRODUCTIVOS---")
    
    resumen_proyecto = state.get("resumen_proyecto", {})
    dataset_materiales = state.get("dataset_materiales", "")
    documents = state.get("documents", [])
    
    # Format RAG documents
    rag_docs_text = []
    for doc in documents:
        src = doc.metadata.get("source", "desconocido")
        page = doc.metadata.get("page", "N/A")
        rag_docs_text.append(
            f"[FUENTE: {src} | PÁGINA: {page}]\n{doc.page_content}"
        )
    
    rag_documents_str = "\n\n".join(rag_docs_text) if rag_docs_text else "No se recuperaron documentos RAG"
    
    # Convert resumen_proyecto to string
    if isinstance(resumen_proyecto, dict):
        resumen_str = json.dumps(resumen_proyecto, indent=2, ensure_ascii=False)
    else:
        resumen_str = str(resumen_proyecto)
    
    dataset_str = dataset_materiales if dataset_materiales else "No se proporcionó dataset de procesos"
    
    print(f"   - Documentos RAG disponibles: {len(documents)}")
    print(f"   - Dataset de procesos: {'Sí' if dataset_materiales else 'No'}")
    
    try:
        # Invoke chain to analyze processes
        analisis = step3_chain.invoke({
            "resumen_proyecto": resumen_str,
            "dataset_materiales": dataset_str,
            "rag_documents": rag_documents_str,
        })
        
        procesos_analizados = analisis.get("analisis_procesos", [])
        print(f"   ✓ Análisis de {len(procesos_analizados)} procesos completado")
        
        for proc in procesos_analizados[:3]:  # Show first 3
            print(f"   - Proceso: {proc.get('proceso_actual', 'N/A')}")
            alternativas = proc.get("alternativas", [])
            print(f"     Alternativas encontradas: {len(alternativas)}")
        
        return {
            "analisis_procesos": procesos_analizados,
        }
    
    except Exception as e:
        print(f"   ✗ Error en análisis de procesos: {str(e)}")
        return {
            "analisis_procesos": [{
                "error": str(e),
                "proceso_actual": "Error en análisis",
            }]
        }
