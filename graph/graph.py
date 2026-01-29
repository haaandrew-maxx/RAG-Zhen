from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import (
    DETECT_MODE, RETRIEVE, GRADE_DOCUMENTS, GENERATE, AGENT_ROUTER,
    ANALISIS_PROYECTO, ANALISIS_MATERIALES, ANALISIS_PROCESOS,
    GESTION_RESIDUOS, RECOMENDACIONES_FINALES, GENERAR_INFORME
)
from graph.nodes import (
    detect_mode, generate, grade_documents, retrieve, agent_router_node,
    analisis_proyecto, analisis_materiales, analisis_procesos,
    gestion_residuos, recomendaciones_finales, generar_informe
)
from graph.state import GraphState
from graph.logger import log_interaction

load_dotenv()

# ============================================================================
# TOGGLE: Enable/Disable Hallucination Check
# ============================================================================
ENABLE_HALLUCINATION_CHECK = False

# ============================================================================
# TOGGLE: Enable/Disable Agentic LCA Analysis
# ============================================================================
ENABLE_AGENTIC_LCA = True  # True = agent decides routing, False = standard generation


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    Grades the generation for hallucinations and relevance.
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    log_interaction(question, documents, generation)
    
    if not ENABLE_HALLUCINATION_CHECK:
        print("---HALLUCINATION CHECK DISABLED---")
        return "useful"
    
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


def decide_to_generate(state: GraphState) -> str:
    """
    Decides whether to use agentic LCA workflow or standard generation.
    """
    print("---ASSESS GRADED DOCUMENTS---")
    
    mode = state.get("mode", "B")
    
    if ENABLE_AGENTIC_LCA and mode == "A":
        print("---DECISION: AGENTIC LCA WORKFLOW---")
        return AGENT_ROUTER
    else:
        print("---DECISION: STANDARD GENERATION---")
        return GENERATE


def route_from_agent(state: GraphState) -> str:
    """
    Routes from agent_router to the next node based on agent's decision.
    """
    next_node = state.get("next_node", GENERATE)
    reasoning = state.get("routing_reasoning", "No reasoning")
    
    print(f"---ROUTING TO: {next_node}---")
    print(f"Reasoning: {reasoning}")
    
    return next_node


def route_after_analysis(state: GraphState) -> str:
    """
    After completing an analysis node, check if there are pending nodes.
    If yes, go to AGENT_ROUTER to decide next step.
    If no pending nodes, check if more analysis is needed or go to GENERATE.
    """
    pending_nodes = state.get("pending_nodes", [])
    
    if pending_nodes:
        print(f"---PENDING NODES REMAINING: {pending_nodes}---")
        print("---RETURNING TO AGENT ROUTER---")
        return AGENT_ROUTER
    else:
        # No more pending nodes - agent will decide if done or needs more analysis
        print("---NO PENDING NODES, RETURNING TO AGENT ROUTER FOR DECISION---")
        return AGENT_ROUTER


# ============================================================================
# Build the Agentic RAG Graph
# ============================================================================
workflow = StateGraph(GraphState)

# Add all nodes
workflow.add_node(DETECT_MODE, detect_mode)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(AGENT_ROUTER, agent_router_node)

# Add LCA analysis nodes
workflow.add_node(ANALISIS_PROYECTO, analisis_proyecto)
workflow.add_node(ANALISIS_MATERIALES, analisis_materiales)
workflow.add_node(ANALISIS_PROCESOS, analisis_procesos)
workflow.add_node(GESTION_RESIDUOS, gestion_residuos)
workflow.add_node(RECOMENDACIONES_FINALES, recomendaciones_finales)

# Add report generation node
workflow.add_node(GENERAR_INFORME, generar_informe)

# Set entry point
workflow.set_entry_point(DETECT_MODE)

# Standard flow
workflow.add_edge(DETECT_MODE, RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

# Conditional: Agentic LCA or standard generation
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
        AGENT_ROUTER: AGENT_ROUTER,
    },
)

# Agent router decides which node to execute next
workflow.add_conditional_edges(
    AGENT_ROUTER,
    route_from_agent,
    {
        GENERATE: GENERATE,
        ANALISIS_PROYECTO: ANALISIS_PROYECTO,
        ANALISIS_MATERIALES: ANALISIS_MATERIALES,
        ANALISIS_PROCESOS: ANALISIS_PROCESOS,
        GESTION_RESIDUOS: GESTION_RESIDUOS,
        RECOMENDACIONES_FINALES: RECOMENDACIONES_FINALES,
        GENERAR_INFORME: GENERAR_INFORME,
    },
)

# After each analysis node, check for pending nodes or return to agent
workflow.add_conditional_edges(
    ANALISIS_PROYECTO,
    route_after_analysis,
    {
        AGENT_ROUTER: AGENT_ROUTER,
    },
)

workflow.add_conditional_edges(
    ANALISIS_MATERIALES,
    route_after_analysis,
    {
        AGENT_ROUTER: AGENT_ROUTER,
    },
)

workflow.add_conditional_edges(
    ANALISIS_PROCESOS,
    route_after_analysis,
    {
        AGENT_ROUTER: AGENT_ROUTER,
    },
)

workflow.add_conditional_edges(
    GESTION_RESIDUOS,
    route_after_analysis,
    {
        AGENT_ROUTER: AGENT_ROUTER,
    },
)

workflow.add_conditional_edges(
    RECOMENDACIONES_FINALES,
    route_after_analysis,
    {
        AGENT_ROUTER: AGENT_ROUTER,
    },
)

# Generation validation
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
    },
)

# Report generation goes directly to END (no hallucination check needed for structured reports)
workflow.add_edge(GENERAR_INFORME, END)

app = workflow.compile()

# Generate graph visualization when run directly
if __name__ == "__main__":
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
    print("✓ Graph visualization saved to graph.png")
