"""
LCA Analysis Node: Step 2 - Material Analysis
Identifies sustainable material alternatives
"""

from typing import Any, Dict
from graph.state import GraphState
from graph.chains.lca_analysis import step2_chain
import json


def analisis_materiales(state: GraphState) -> Dict[str, Any]:
    """
    PASO 2: Análisis de materiales
    
    For each main material:
    - Identifies sustainable alternatives from dataset or RAG
    - Evaluates recycled/reused material possibilities
    - Estimates CO2 reduction potential
    - Assesses circular economy alignment
    """
    print("---PASO 2: ANÁLISIS DE MATERIALES---")
    
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
    
    dataset_str = dataset_materiales if dataset_materiales else "No se proporcionó dataset de materiales"
    
    print(f"   - Documentos RAG disponibles: {len(documents)}")
    print(f"   - Dataset de materiales: {'Sí' if dataset_materiales else 'No'}")
    
    try:
        # Invoke chain to analyze materials
        analisis = step2_chain.invoke({
            "resumen_proyecto": resumen_str,
            "dataset_materiales": dataset_str,
            "rag_documents": rag_documents_str,
        })
        
        materiales_analizados = analisis.get("analisis_materiales", [])
        print(f"   ✓ Análisis de {len(materiales_analizados)} materiales completado")
        
        for mat in materiales_analizados[:3]:  # Show first 3
            print(f"   - Material: {mat.get('material_actual', 'N/A')}")
            alternativas = mat.get("alternativas", [])
            print(f"     Alternativas encontradas: {len(alternativas)}")
        
        return {
            "analisis_materiales": materiales_analizados,
        }
    
    except Exception as e:
        print(f"   ✗ Error en análisis de materiales: {str(e)}")
        return {
            "analisis_materiales": [{
                "error": str(e),
                "material_actual": "Error en análisis",
            }]
        }
