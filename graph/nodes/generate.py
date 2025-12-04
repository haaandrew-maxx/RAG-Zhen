from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState
from langchain_core.documents import Document


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE ANSWER (JSON MODE)---")

    question = state["question"]
    description = state["description"]
    bom = state["bom"]
    documents = state["documents"]

    # Build context
    docs_as_text = []
    for d in documents:
        src = d.metadata.get("source", "desconocido")
        page = d.metadata.get("page", "N/A")
        docs_as_text.append(
            f"[SOURCE: {src} | PAGE: {page}]\n{d.page_content}"
        )

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

    return {
        "generation": generation,
        "question": question,
        "documents": documents,
    }