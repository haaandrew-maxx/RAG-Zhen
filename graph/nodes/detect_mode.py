from typing import Any, Dict
from graph.state import GraphState


def detect_mode(state: GraphState) -> Dict[str, Any]:
    """
    Detects the operation mode based on uploaded files.
    
    Mode A: Both description and BOM are present (project-specific analysis)
    Mode B: No description or BOM (knowledge base only query)
    
    Returns:
        Updated state with 'mode' set to "A" or "B"
    """
    print("---DETECT MODE---")
    
    description = state.get("description", "")
    bom = state.get("bom", "")
    session_docs = state.get("session_docs", [])
    
    # Check if we have both description and BOM
    has_description = bool(description and description.strip())
    has_bom = bool(bom and bom.strip())
    
    # Mode A: Both files present
    if has_description and has_bom:
        mode = "A"
        print(f"---MODE A DETECTED: Description + BOM present---")
        print(f"   - Description length: {len(description)} chars")
        print(f"   - BOM length: {len(bom)} chars")
        print(f"   - Session docs: {len(session_docs)} documents")
    # Mode B: Pure knowledge base query
    else:
        mode = "B"
        print(f"---MODE B DETECTED: Knowledge base only---")
        if not has_description:
            print("   - No description provided")
        if not has_bom:
            print("   - No BOM provided")
    
    return {
        "mode": mode,
        "question": state["question"],
    }
