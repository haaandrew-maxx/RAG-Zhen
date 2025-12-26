from dotenv import load_dotenv

from langgraph.graph import END, StateGraph


from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import DETECT_MODE, RETRIEVE, GRADE_DOCUMENTS, GENERATE
from graph.nodes import detect_mode, generate, grade_documents, retrieve
from graph.state import GraphState
from graph.logger import log_interaction
load_dotenv()

# ============================================================================
# TOGGLE: Enable/Disable Hallucination Check
# ============================================================================
# Set to False to disable hallucination checking and prevent re-generation loops
ENABLE_HALLUCINATION_CHECK = False  # TOGGLE: Change to True to re-enable


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Grades the generation for hallucinations and relevance.
    
    If ENABLE_HALLUCINATION_CHECK is False, always returns "useful" (no retry loop).
    If True, performs full hallucination and answer grading.
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    # Always log interaction regardless of hallucination check
    log_interaction(question, documents, generation)
    
    # ========================================================================
    # DISABLED MODE: Skip hallucination check, always accept generation
    # ========================================================================
    if not ENABLE_HALLUCINATION_CHECK:
        print("---HALLUCINATION CHECK DISABLED---")
        print("---ACCEPTING GENERATION WITHOUT VALIDATION---")
        return "useful"
    
    # ========================================================================
    # ENABLED MODE: Full hallucination and answer grading
    # ========================================================================
    print("---CHECK HALLUCINATIONS---")
    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"



def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")
    print("---DECISION: GENERATE---")
    return GENERATE


# ============================================================================
# Build the graph with mode detection as entry point
# ============================================================================
workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node(DETECT_MODE, detect_mode)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)

# Set mode detection as entry point
workflow.set_entry_point(DETECT_MODE)

# Flow: DETECT_MODE -> RETRIEVE -> GRADE_DOCUMENTS -> GENERATE
workflow.add_edge(DETECT_MODE, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
    },
)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")