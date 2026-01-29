"""
Agent Router Node for Agentic RAG
Intelligently decides which analysis nodes to execute based on question and current state
"""

from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from graph.state import GraphState
from graph.consts import (
    GENERATE, ANALISIS_PROYECTO, ANALISIS_MATERIALES, 
    ANALISIS_PROCESOS, GESTION_RESIDUOS, RECOMENDACIONES_FINALES,
    GENERAR_INFORME
)
import json


# Node name normalization mapping
NODE_NAME_MAP = {
    "generate": GENERATE,
    "analisis_proyecto": ANALISIS_PROYECTO,
    "analisis_materiales": ANALISIS_MATERIALES,
    "analisis_procesos": ANALISIS_PROCESOS,
    "gestion_residuos": GESTION_RESIDUOS,
    "recomendaciones_finales": RECOMENDACIONES_FINALES,
    "generar_informe": GENERAR_INFORME,
    # Also handle uppercase variants
    "GENERATE": GENERATE,
    "ANALISIS_PROYECTO": ANALISIS_PROYECTO,
    "ANALISIS_MATERIALES": ANALISIS_MATERIALES,
    "ANALISIS_PROCESOS": ANALISIS_PROCESOS,
    "GESTION_RESIDUOS": GESTION_RESIDUOS,
    "RECOMENDACIONES_FINALES": RECOMENDACIONES_FINALES,
    "GENERAR_INFORME": GENERAR_INFORME,
}


class RoutingDecision(BaseModel):
    """Structured output for routing decisions"""
    next_nodes: List[str] = Field(description="List of nodes to execute next, in order")
    reasoning: str = Field(description="Explanation of why these nodes are needed")


ROUTER_SYSTEM_PROMPT = """You are an intelligent routing agent for a Life Cycle Assessment (LCA) RAG system.

Your task: Analyze the user's question and current analysis state, then decide which analysis node(s) to execute next.

Available nodes (use these EXACT names in lowercase):
1. analisis_proyecto - Analyze overall project environmental impact (first step for comprehensive analysis)
2. analisis_materiales - Analyze material sustainability (requires project analysis)
3. analisis_procesos - Analyze process optimizations (requires materials analysis)
4. gestion_residuos - Analyze waste management (requires process analysis)
5. recomendaciones_finales - Generate final recommendations (required for comprehensive reports)
6. generar_informe - Generate comprehensive markdown report (use when user asks for "report", "informe", or comprehensive documentation)
7. generate - Generate final answer (when enough analysis is done for regular questions)

Routing rules:
- For simple questions about specific aspects: route directly to relevant node, then generate
- For comprehensive LCA analysis: route through nodes in dependency order
- When user asks for "report", "informe", "comprehensive report", or "documentation": 
  → Route ALL 5 analysis nodes (analisis_proyecto, analisis_materiales, analisis_procesos, gestion_residuos, recomendaciones_finales)
  → Then route to generar_informe (NOT generate)
- For regular questions: end with "generate"
- For report requests: end with "generar_informe"
- Don't execute unnecessary nodes if the question doesn't require them

Examples:
Q: "What materials are used?" → ["analisis_proyecto", "generate"]
Q: "How can we reduce material waste?" → ["analisis_proyecto", "analisis_materiales", "gestion_residuos", "generate"]
Q: "Generate a comprehensive report" → ["analisis_proyecto", "analisis_materiales", "analisis_procesos", "gestion_residuos", "recomendaciones_finales", "generar_informe"]
Q: "I need an informe" → ["analisis_proyecto", "analisis_materiales", "analisis_procesos", "gestion_residuos", "recomendaciones_finales", "generar_informe"]
Q: "Create documentation of the LCA" → ["analisis_proyecto", "analisis_materiales", "analisis_procesos", "gestion_residuos", "recomendaciones_finales", "generar_informe"]
Q: "What is the environmental impact?" → ["analisis_proyecto", "generate"]

CRITICAL: Always use lowercase node names exactly as shown above.

Output format (JSON):
{
    "next_nodes": ["NODE_NAME_1", "NODE_NAME_2"],
    "reasoning": "Why these nodes are needed"
}

If analysis is complete and ready to generate answer:
{
    "next_nodes": ["GENERATE"],
    "reasoning": "Sufficient analysis completed"
}

If report is requested:
{
    "next_nodes": ["analisis_proyecto", "analisis_materiales", "analisis_procesos", "gestion_residuos", "recomendaciones_finales", "generar_informe"],
    "reasoning": "Comprehensive report requested - all analysis nodes + report generation"
}

IMPORTANT: Return ONLY the JSON object, no markdown code blocks, no explanations before or after.
"""


def create_router_llm():
    """Create LLM for routing decisions with structured output"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    return llm.with_structured_output(RoutingDecision)


def agent_router_node(state: GraphState) -> Dict[str, Any]:
    """
    Agentic Router Node - Decides which analysis nodes to execute
    
    Returns updated state with:
    - next_node: The next node to execute
    - routing_reasoning: Explanation of routing decision
    """
    question = state["question"]
    
    # Check what analysis has been completed
    completed_analysis = []
    if state.get("resumen_proyecto"):
        completed_analysis.append("ANALISIS_PROYECTO")
    if state.get("analisis_materiales"):
        completed_analysis.append("ANALISIS_MATERIALES")
    if state.get("analisis_procesos"):
        completed_analysis.append("ANALISIS_PROCESOS")
    if state.get("analisis_residuos"):
        completed_analysis.append("GESTION_RESIDUOS")
    if state.get("recomendaciones_finales"):
        completed_analysis.append("RECOMENDACIONES_FINALES")
    
    # Build routing prompt
    routing_prompt = f"""
Question: {question}

Completed analysis nodes: {completed_analysis if completed_analysis else "None"}

Available context:
- Project data: {"Yes" if state.get("datos_proyecto") else "No"}
- BOM data: {"Yes" if state.get("bom") else "No"}
- Retrieved documents: {len(state.get("documents", []))} docs

Decide which node(s) to execute next to answer this question.
"""
    
    llm = create_router_llm()
    
    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=routing_prompt)
    ]
    # Get structured output
    try:
        decision: RoutingDecision = llm.invoke(messages)
        next_nodes = decision.next_nodes if decision.next_nodes else ["generate"]
        reasoning = decision.reasoning
        
        # Normalize node names using mapping
        next_nodes = [NODE_NAME_MAP.get(node, node) for node in next_nodes]
        
    except Exception as e:
        # Fallback if structured output fails
        print(f"Warning: Router LLM failed: {e}")
        next_nodes = [GENERATE]
        reasoning = "Fallback to generate due to LLM error"
    
    print(f"---AGENT ROUTER DECISION---")
    print(f"Next nodes: {next_nodes}")
    print(f"Reasoning: {reasoning}")
    
    return {
        "next_node": next_nodes[0] if next_nodes else GENERATE,
        "pending_nodes": next_nodes[1:] if len(next_nodes) > 1 else [],
        "routing_reasoning": reasoning
    }
