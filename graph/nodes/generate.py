from typing import Any, Dict
from graph.chains.generation import generation_chain, generation_chain_mode_b
from graph.state import GraphState
from langchain_core.documents import Document


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates answer based on the operation mode and available analysis results.
    
    Mode A: Uses description + BOM + knowledge base documents + analysis results (if any)
    Mode B: Uses only knowledge base documents
    """
    print("---GENERATE ANSWER (JSON MODE)---")

    question = state["question"]
    description = state.get("description", "")
    bom = state.get("bom", "")
    documents = state["documents"]
    mode = state.get("mode", "B")
    
    # Check if we have LCA analysis results
    resumen_proyecto = state.get("resumen_proyecto")
    analisis_materiales = state.get("analisis_materiales")
    analisis_procesos = state.get("analisis_procesos")
    analisis_residuos = state.get("analisis_residuos")
    recomendaciones_finales = state.get("recomendaciones_finales")
    
    # If recomendaciones_finales was generated, use that directly
    if recomendaciones_finales:
        print("---USING FINAL RECOMMENDATIONS AS GENERATION---")
        import json
        generation_json = json.dumps(recomendaciones_finales, indent=2, ensure_ascii=False)
        return {
            "generation": generation_json,
            "question": question,
            "documents": documents,
        }
    
    print(f"---GENERATING IN MODE {mode}---")

    # Build context from retrieved documents
    docs_as_text = []
    for d in documents:
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "N/A")
        docs_as_text.append(
            f"[SOURCE: {src} | PAGE: {page}]\n{d.page_content}"
        )

    # Mode A: Include description + BOM + analysis results in context
    if mode == "A":
        context_parts = ["=== Project Description ===\n" + description]
        context_parts.append("\n\n=== BOM ===\n" + bom)
        
        # Add analysis results if available
        if resumen_proyecto:
            import json
            context_parts.append("\n\n=== Project Analysis ===\n" + json.dumps(resumen_proyecto, indent=2, ensure_ascii=False))
        if analisis_materiales:
            import json
            context_parts.append("\n\n=== Materials Analysis ===\n" + json.dumps(analisis_materiales, indent=2, ensure_ascii=False))
        if analisis_procesos:
            import json
            context_parts.append("\n\n=== Process Analysis ===\n" + json.dumps(analisis_procesos, indent=2, ensure_ascii=False))
        if analisis_residuos:
            import json
            context_parts.append("\n\n=== Waste Management Analysis ===\n" + json.dumps(analisis_residuos, indent=2, ensure_ascii=False))
        
        context_parts.append("\n\n=== Retrieved Documents ===\n" + "\n\n".join(docs_as_text))
        final_context = "".join(context_parts)
        
        generation = generation_chain.invoke({
            "context": final_context,
            "question": question,
        })
    
    # Mode B: Only knowledge base documents
    else:
        final_context = (
            "=== Retrieved Documents ===\n"
            + "\n\n".join(docs_as_text)
        )
        
        generation = generation_chain_mode_b.invoke({
            "context": final_context,
            "question": question,
        })

    return {
        "generation": generation,
        "question": question,
        "documents": documents,
    }