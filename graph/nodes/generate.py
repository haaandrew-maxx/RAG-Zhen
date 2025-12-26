from typing import Any, Dict
from graph.chains.generation import generation_chain, generation_chain_mode_b
from graph.state import GraphState
from langchain_core.documents import Document


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generates answer based on the operation mode.
    
    Mode A: Uses description + BOM + knowledge base documents
    Mode B: Uses only knowledge base documents
    """
    print("---GENERATE ANSWER (JSON MODE)---")

    question = state["question"]
    description = state.get("description", "")
    bom = state.get("bom", "")
    documents = state["documents"]
    mode = state.get("mode", "B")
    
    print(f"---GENERATING IN MODE {mode}---")

    # Build context from retrieved documents
    docs_as_text = []
    for d in documents:
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "N/A")
        docs_as_text.append(
            f"[SOURCE: {src} | PAGE: {page}]\n{d.page_content}"
        )

    # Mode A: Include description + BOM in context
    if mode == "A":
        final_context = (
            "=== Project Description ===\n"
            + description
            + "\n\n=== BOM ===\n"
            + bom
            + "\n\n=== Retrieved Documents ===\n"
            + "\n\n".join(docs_as_text)
        )
        
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