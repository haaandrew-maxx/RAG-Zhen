"""
LCA Analysis Node: Step 4 - Waste Management Analysis
Proposes waste management improvements
"""

from typing import Any, Dict
from graph.state import GraphState
from graph.chains.lca_analysis import step4_chain
import json


def gestion_residuos(state: GraphState) -> Dict[str, Any]:
    """
    PASO 4: Gestión de residuos
    
    Analyzes waste management:
    - Evaluates current reuse, recycling, or valorization
    - Proposes improvements based on:
      * Component reuse
      * Critical material recycling
      * Design for disassembly
    - Links proposals with RAG documentation when available
    """
    print("---PASO 4: GESTIÓN DE RESIDUOS---")
    
    resumen_proyecto = state.get("resumen_proyecto", {})
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
    
    print(f"   - Documentos RAG disponibles: {len(documents)}")
    
    try:
        # Invoke chain to analyze waste management
        analisis = step4_chain.invoke({
            "resumen_proyecto": resumen_str,
            "rag_documents": rag_documents_str,
        })
        
        mejoras = analisis.get("mejoras_propuestas", [])
        print(f"   ✓ Análisis de gestión de residuos completado")
        print(f"   - Mejoras propuestas: {len(mejoras)}")
        
        gestion_actual = analisis.get("gestion_residuos_actual", {})
        print(f"   - Reutilización actual: {gestion_actual.get('reutilizacion', False)}")
        print(f"   - Reciclaje actual: {gestion_actual.get('reciclaje', False)}")
        print(f"   - Valorización actual: {gestion_actual.get('valorizacion', False)}")
        
        return {
            "analisis_residuos": analisis,
        }
    
    except Exception as e:
        print(f"   ✗ Error en análisis de residuos: {str(e)}")
        return {
            "analisis_residuos": {
                "error": str(e),
                "gestion_residuos_actual": {"descripcion": "Error en análisis"},
            }
        }
