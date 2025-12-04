from typing import List, TypedDict, Optional
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: user question
        bom: BOM table (string or JSON-encoded string)
        description: project description text
        generation: LLM JSON generation (string)
        documents: retrieved documents (base + session)
        session_docs: per-session uploaded docs (description + BOM)
    """

    question: str
    bom: str
    description: str

    # The final model output (JSON string)
    generation: str

    # Retrieved documents from RAG
    documents: List[Document]

    # NEW: uploaded files in this session (used for session-level vectorstore)
    session_docs: Optional[List[Document]]