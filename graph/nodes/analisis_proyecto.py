"""
LCA Analysis Node: Step 1 - Project Understanding
Analyzes and summarizes the project's environmental impact
"""

from typing import Any, Dict
from graph.state import GraphState
from graph.chains.lca_analysis import step1_chain
import json


def analisis_proyecto(state: GraphState) -> Dict[str, Any]:
    """
    PASO 1: Comprensión del proyecto
    
    Analyzes the project and summarizes:
    - Product or system evaluated
    - Total environmental impact (CO2 eq)
    - Impact distribution by phases
    - Most relevant materials and processes by impact
    """
    print("---PASO 1: COMPRENSIÓN DEL PROYECTO---")
    
    description = state.get("description", "")
    bom = state.get("bom", "")
    datos_proyecto = state.get("datos_proyecto", {})
    
    # Convert datos_proyecto to string if it's a dict
    if isinstance(datos_proyecto, dict):
        datos_proyecto_str = json.dumps(datos_proyecto, indent=2, ensure_ascii=False)
    else:
        datos_proyecto_str = str(datos_proyecto) if datos_proyecto else "No se proporcionaron datos estructurados del proyecto"
    
    print(f"   - Datos del proyecto: {len(datos_proyecto_str)} chars")
    print(f"   - Descripción: {len(description)} chars")
    print(f"   - BOM: {len(bom)} chars")
    
    try:
        # Invoke chain to analyze project
        resumen = step1_chain.invoke({
            "datos_proyecto": datos_proyecto_str,
            "description": description,
            "bom": bom,
        })
        
        print("   ✓ Resumen del proyecto generado exitosamente")
        print(f"   - Producto/Sistema: {resumen.get('producto_sistema', 'N/A')}")
        
        if "impacto_ambiental_total" in resumen:
            impacto = resumen["impacto_ambiental_total"]
            print(f"   - Impacto total: {impacto.get('co2_eq', 'N/A')} {impacto.get('unidad', '')}")
        
        return {
            "resumen_proyecto": resumen,
        }
    
    except Exception as e:
        print(f"   ✗ Error en análisis del proyecto: {str(e)}")
        return {
            "resumen_proyecto": {
                "error": str(e),
                "producto_sistema": "Error en análisis",
            }
        }
