from typing import List, TypedDict, Optional, Dict, Any
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Represents the state of our graph for Life Cycle Assessment (LCA) analysis.

    Attributes:
        question: user question
        bom: BOM table (string or JSON-encoded string)
        description: project description text
        generation: LLM JSON generation (string)
        documents: retrieved documents (base + session)
        session_docs: per-session uploaded docs (description + BOM)
        mode: operation mode - "A" (with description+BOM) or "B" (knowledge base only)
        
        # LCA-specific fields:
        dataset_materiales: Excel dataset of alternative sustainable materials/processes
        datos_proyecto: Structured project data (environmental impact, materials, transport, processes, waste)
        resumen_proyecto: Step 1 output - Project understanding summary
        analisis_materiales: Step 2 output - Material analysis and alternatives
        analisis_procesos: Step 3 output - Process analysis and improvements
        analisis_residuos: Step 4 output - Waste management analysis
        recomendaciones_finales: Step 5 output - Final structured recommendations
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
    
    # NEW: Operation mode - "A" or "B"
    mode: str
    
    # LCA-specific fields
    dataset_materiales: Optional[str]  # Excel data as string or JSON
    datos_proyecto: Optional[Dict[str, Any]]  # Structured project data
    resumen_proyecto: Optional[Dict[str, Any]]  # Step 1 output
    analisis_materiales: Optional[List[Dict[str, Any]]]  # Step 2 output
    analisis_procesos: Optional[List[Dict[str, Any]]]  # Step 3 output
    analisis_residuos: Optional[Dict[str, Any]]  # Step 4 output
    recomendaciones_finales: Optional[Dict[str, Any]]  # Step 5 output