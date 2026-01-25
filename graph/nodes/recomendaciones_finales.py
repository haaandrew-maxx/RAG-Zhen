"""
LCA Analysis Node: Step 5 - Final Recommendations
Consolidates all analysis into structured recommendations
"""

from typing import Any, Dict
from graph.state import GraphState
from graph.chains.lca_analysis import step5_chain
import json


def recomendaciones_finales(state: GraphState) -> Dict[str, Any]:
    """
    PASO 5: Recomendaciones finales
    
    Generates a structured list of final recommendations, separated by:
    - Materials
    - Production processes
    - Waste management
    
    For each recommendation:
    - Clear description
    - Environmental justification
    - Knowledge source (Excel / RAG Document)
    - Expected environmental impact (qualitative)
    """
    print("---PASO 5: RECOMENDACIONES FINALES---")
    
    resumen_proyecto = state.get("resumen_proyecto", {})
    analisis_materiales = state.get("analisis_materiales", [])
    analisis_procesos = state.get("analisis_procesos", [])
    analisis_residuos = state.get("analisis_residuos", {})
    
    # Convert all to strings for the prompt
    resumen_str = json.dumps(resumen_proyecto, indent=2, ensure_ascii=False)
    materiales_str = json.dumps(analisis_materiales, indent=2, ensure_ascii=False)
    procesos_str = json.dumps(analisis_procesos, indent=2, ensure_ascii=False)
    residuos_str = json.dumps(analisis_residuos, indent=2, ensure_ascii=False)
    
    # Safe length checks
    num_materiales = len(analisis_materiales) if analisis_materiales else 0
    num_procesos = len(analisis_procesos) if analisis_procesos else 0
    
    print(f"   - Materiales analizados: {num_materiales}")
    print(f"   - Procesos analizados: {num_procesos}")
    
    try:
        # Invoke chain to generate final recommendations
        recomendaciones = step5_chain.invoke({
            "resumen_proyecto": resumen_str,
            "analisis_materiales": materiales_str,
            "analisis_procesos": procesos_str,
            "analisis_residuos": residuos_str,
        })
        
        print("   ✓ Recomendaciones finales generadas exitosamente")
        print(f"   - Recomendaciones de materiales: {len(recomendaciones.get('recomendaciones_materiales', []))}")
        print(f"   - Recomendaciones de procesos: {len(recomendaciones.get('recomendaciones_procesos', []))}")
        print(f"   - Recomendaciones de residuos: {len(recomendaciones.get('recomendaciones_gestion_residuos', []))}")
        
        # Convert to JSON string for generation field
        generation_json = json.dumps(recomendaciones, indent=2, ensure_ascii=False)
        
        return {
            "recomendaciones_finales": recomendaciones,
            "generation": generation_json,  # Final output
        }
    
    except Exception as e:
        print(f"   ✗ Error en generación de recomendaciones finales: {str(e)}")
        error_output = {
            "error": str(e),
            "resumen_proyecto": {},
            "recomendaciones_materiales": [],
            "recomendaciones_procesos": [],
            "recomendaciones_gestion_residuos": [],
            "conclusiones": "Error al generar recomendaciones finales"
        }
        return {
            "recomendaciones_finales": error_output,
            "generation": json.dumps(error_output, indent=2, ensure_ascii=False),
        }
