"""
Generar Informe Node
Genera un informe completo en markdown basado en los resultados del análisis LCA
"""

from typing import Any, Dict
from graph.state import GraphState
import json


def generar_informe(state: GraphState) -> Dict[str, Any]:
    """Genera un informe completo en español (markdown)."""
    print("---GENERANDO INFORME COMPLETO EN MARKDOWN---")
    
    question = state["question"]
    documents = state["documents"]
    resumen_proyecto = state.get("resumen_proyecto", {})
    analisis_materiales = state.get("analisis_materiales", [])
    analisis_procesos = state.get("analisis_procesos", [])
    analisis_residuos = state.get("analisis_residuos", {})
    recomendaciones_finales = state.get("recomendaciones_finales", {})
    
    lines = []
    lines.append("# Informe Completo de Análisis de Ciclo de Vida (ACV)")
    lines.append("")
    lines.append(f"**Consulta:** {question}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Section 1: Project Analysis
    if resumen_proyecto and isinstance(resumen_proyecto, dict):
        lines.append("## 1. Análisis del Proyecto")
        lines.append("")
        
        if "producto_sistema" in resumen_proyecto:
            lines.append("### Producto/Sistema")
            lines.append(f"{resumen_proyecto['producto_sistema']}")
            lines.append("")
        
        if "impacto_ambiental_total" in resumen_proyecto:
            impact = resumen_proyecto["impacto_ambiental_total"]
            lines.append("### Impacto Ambiental Total")
            if isinstance(impact, dict):
                co2 = impact.get('co2_eq')
                unidad = impact.get('unidad', 'kg CO2 eq')
                if co2 is not None:
                    lines.append(f"**{co2} {unidad}**")
                else:
                    lines.append("*Datos no disponibles en la documentación fuente*")
            else:
                lines.append(f"{impact}")
            lines.append("")
        
        if "distribucion_impacto" in resumen_proyecto:
            distrib = resumen_proyecto["distribucion_impacto"]
            lines.append("### Distribución de Impacto por Fase")
            if isinstance(distrib, dict):
                for fase, datos in distrib.items():
                    if isinstance(datos, dict):
                        pct = datos.get('porcentaje', 'N/A')
                        co2 = datos.get('co2_eq', 'N/A')
                        lines.append(f"- **{fase.replace('_', ' ').title()}:** {pct}% ({co2} kg CO2 eq)")
            lines.append("")
        
        if "materiales_relevantes" in resumen_proyecto:
            mats = resumen_proyecto["materiales_relevantes"]
            lines.append("### Materiales Clave por Impacto")
            if isinstance(mats, list) and mats:
                for mat in mats:
                    if isinstance(mat, dict):
                        nombre = mat.get('material', 'N/A')
                        cantidad = mat.get('cantidad', '')
                        unidad = mat.get('unidad', '')
                        impacto = mat.get('impacto_co2', 'N/A')
                        pct = mat.get('porcentaje_impacto_total', '')
                        
                        mat_info = f"**{nombre}**"
                        if cantidad and unidad:
                            mat_info += f" ({cantidad} {unidad})"
                        mat_info += f": {impacto} kg CO2 eq"
                        if pct:
                            mat_info += f" ({pct}% del total)"
                        lines.append(f"- {mat_info}")
            lines.append("")
        
        if "procesos_relevantes" in resumen_proyecto:
            procs = resumen_proyecto["procesos_relevantes"]
            lines.append("### Procesos Clave por Impacto")
            if isinstance(procs, list) and procs:
                for proc in procs:
                    if isinstance(proc, dict):
                        nombre = proc.get('proceso', 'N/A')
                        energia = proc.get('energia_consumida', 'N/A')
                        tipo = proc.get('tipo_energia', 'N/A')
                        impacto = proc.get('impacto_co2', 'N/A')
                        lines.append(f"- **{nombre}**: {impacto} kg CO2 eq (Energía: {energia}, Tipo: {tipo})")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Section 2: Materials Analysis
    if analisis_materiales and isinstance(analisis_materiales, list):
        lines.append("## 2. Análisis de Materiales")
        lines.append("")
        
        for idx, mat in enumerate(analisis_materiales, 1):
            if isinstance(mat, dict):
                mat_name = mat.get('material_actual', 'Desconocido')
                lines.append(f"### Material {idx}: {mat_name}")
                lines.append("")
                
                # Mostrar función del material
                funcion = mat.get('funcion_material', '')
                if funcion:
                    lines.append(f"**Función:** {funcion}")
                    lines.append("")
                
                # Mostrar cantidad y factores de emisión
                cantidad = mat.get('cantidad_kg', '')
                factor_actual = mat.get('factor_emision_actual', '')
                co2 = mat.get('impacto_actual_co2')
                
                if cantidad:
                    lines.append(f"**Cantidad:** {cantidad} kg")
                if factor_actual:
                    lines.append(f"**Factor de Emisión:** {factor_actual}")
                if co2 is not None:
                    lines.append(f"**Impacto CO2 Total:** {co2} kg CO2 eq")
                else:
                    lines.append(f"**Impacto CO2 Total:** *Datos no disponibles*")
                lines.append("")
                
                # Alternatives - solo mostrar si reducen emisiones
                alternativas = mat.get('alternativas', [])
                if alternativas and isinstance(alternativas, list):
                    # Filtrar solo alternativas con menor factor de emisión
                    alternativas_validas = []
                    for alt in alternativas:
                        if isinstance(alt, dict):
                            factor_alt = alt.get('factor_emision_alternativo', '')
                            if factor_actual and factor_alt:
                                try:
                                    fact_act = float(factor_actual)
                                    fact_alt = float(factor_alt)
                                    # Solo incluir si el factor alternativo es MENOR que el actual
                                    if fact_alt < fact_act:
                                        alternativas_validas.append(alt)
                                except:
                                    pass
                    
                    if alternativas_validas:
                        lines.append("**Alternativas Sostenibles:**")
                        for alt in alternativas_validas:
                            alt_name = alt.get('material_alternativo', alt.get('alternativa', 'N/A'))
                            factor_alt = alt.get('factor_emision_alternativo', '')
                            reduccion_kg = alt.get('reduccion_co2_por_kg', '')
                            reduccion = alt.get('reduccion_co2_estimada', alt.get('reduccion_co2', ''))
                            porcentaje = alt.get('porcentaje_reduccion', '')
                            viabilidad = alt.get('viabilidad', '')
                            justif = alt.get('justificacion', alt.get('beneficio_ambiental', 'N/A'))
                            consideraciones = alt.get('consideraciones_tecnicas', '')
                            
                            lines.append(f"- **{alt_name}**")
                            
                            # Factor de emisión
                            if factor_alt:
                                lines.append(f"  - **Factor de Emisión:** {factor_alt}")
                            
                            # Comparación visual (siempre será menor porque ya filtramos)
                            if factor_actual and factor_alt:
                                try:
                                    fact_act = float(factor_actual)
                                    fact_alt = float(factor_alt)
                                    lines.append(f"  - ✅ Menor emisión por kg ({((fact_act - fact_alt) / fact_act * 100):.1f}% menos)")
                                except:
                                    pass
                            
                            # Viabilidad
                            if viabilidad:
                                viab_icons = {'alta': '✅ Alta', 'media': '⚡ Media', 'baja': '⚠️ Baja'}
                                lines.append(f"  - **Viabilidad:** {viab_icons.get(viabilidad.lower(), viabilidad)}")
                            
                            # Reducción total (solo mostrar si es positiva)
                            if reduccion:
                                try:
                                    red_val = float(reduccion)
                                    if red_val > 0:
                                        red_text = f"✅ Reducción de {reduccion} kg CO2 eq"
                                        if porcentaje:
                                            red_text += f" ({porcentaje}%)"
                                        lines.append(f"  - **Impacto:** {red_text}")
                                except:
                                    lines.append(f"  - **Reducción CO2:** {reduccion} kg CO2 eq")
                            
                            lines.append(f"  - **Justificación:** {justif}")
                            
                            if consideraciones:
                                lines.append(f"  - **Consideraciones:** {consideraciones}")
                            
                            # Economía circular
                            econ_circ = alt.get('economia_circular', {})
                            if econ_circ and isinstance(econ_circ, dict):
                                circ_info = []
                                if econ_circ.get('reutilizable'): circ_info.append("Reutilizable")
                                if econ_circ.get('reciclable'): circ_info.append("Reciclable")
                                if econ_circ.get('biodegradable'): circ_info.append("Biodegradable")
                                if circ_info:
                                    lines.append(f"  - **Economía Circular:** {', '.join(circ_info)}")
                        lines.append("")
                    else:
                        lines.append("*No se encontraron alternativas con menor impacto de carbono.*")
                        lines.append("")
                elif mat.get('sin_alternativa'):
                    lines.append("*No se identificaron alternativas sostenibles viables.*")
                    lines.append("")
                
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Section 3: Análisis de Procesos
    if analisis_procesos and isinstance(analisis_procesos, list):
        lines.append("## 3. Análisis de Procesos")
        lines.append("")
        
        for idx, proc in enumerate(analisis_procesos, 1):
            if isinstance(proc, dict):
                proc_name = proc.get('proceso_actual', 'Desconocido')
                lines.append(f"### Proceso {idx}: {proc_name}")
                lines.append("")
                
                energia = proc.get('consumo_energia_actual')
                tipo_energia = proc.get('tipo_energia_actual', 'no especificado')
                co2 = proc.get('impacto_co2_actual')
                
                if energia is not None:
                    lines.append(f"**Consumo de Energía:** {energia}")
                else:
                    lines.append(f"**Consumo de Energía:** *Datos no disponibles*")
                lines.append(f"**Tipo de Energía:** {tipo_energia}")
                if co2 is not None:
                    lines.append(f"**Impacto CO2:** {co2} kg CO2 eq")
                else:
                    lines.append(f"**Impacto CO2:** *Datos no disponibles*")
                lines.append("")
                
                # Alternatives
                alternativas = proc.get('alternativas', [])
                if alternativas and isinstance(alternativas, list):
                    lines.append("**Oportunidades de Mejora:**")
                    for alt in alternativas:
                        if isinstance(alt, dict):
                            alt_name = alt.get('proceso_alternativo', alt.get('alternativa', 'N/A'))
                            reduccion = alt.get('reduccion_co2_estimada', alt.get('reduccion_co2', ''))
                            grado = alt.get('grado_mejora', '')
                            justif = alt.get('justificacion', alt.get('beneficio_ambiental', 'N/A'))
                            
                            lines.append(f"- **{alt_name}**")
                            if grado:
                                grado_es = {'bajo': 'Bajo', 'medio': 'Medio', 'alto': 'Alto'}.get(grado.lower(), grado)
                                lines.append(f"  - **Nivel de Mejora:** {grado_es}")
                            if reduccion:
                                lines.append(f"  - **Reducción de CO2:** {reduccion} kg CO2 eq")
                            lines.append(f"  - **Beneficio Ambiental:** {justif}")
                            
                            # Optimizations
                            opts = alt.get('optimizaciones', [])
                            if opts and isinstance(opts, list):
                                lines.append(f"  - **Optimizaciones:** {', '.join(opts)}")
                    lines.append("")
                elif proc.get('sin_alternativa'):
                    lines.append("*No se identificaron mejoras de proceso viables en los datos disponibles.*")
                    lines.append("")
                
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Section 4: Análisis de Gestión de Residuos
    if analisis_residuos and isinstance(analisis_residuos, dict):
        lines.append("## 4. Análisis de Gestión de Residuos")
        lines.append("")
        
        if "gestion_residuos_actual" in analisis_residuos:
            current = analisis_residuos["gestion_residuos_actual"]
            lines.append("### Prácticas Actuales de Gestión de Residuos")
            if isinstance(current, dict):
                reuse = current.get('reutilizacion', False)
                recycle = current.get('reciclaje', False)
                valorizacion = current.get('valorizacion', False)
                lines.append(f"- **Reutilización:** {'Sí' if reuse else 'No'}")
                lines.append(f"- **Reciclaje:** {'Sí' if recycle else 'No'}")
                lines.append(f"- **Recuperación/Valorización:** {'Sí' if valorizacion else 'No'}")
                if 'descripcion' in current:
                    lines.append(f"\n{current['descripcion']}")
            lines.append("")
        
        if "mejoras_propuestas" in analisis_residuos:
            mejoras = analisis_residuos["mejoras_propuestas"]
            lines.append("### Mejoras Propuestas")
            if isinstance(mejoras, list):
                for mejora in mejoras:
                    if isinstance(mejora, dict):
                        accion = mejora.get('accion', 'N/A')
                        impacto = mejora.get('impacto_esperado', 'N/A')
                        lines.append(f"- **{accion}**")
                        lines.append(f"  - Impacto Esperado: {impacto}")
            lines.append("")
        
        if "diseño_desmontaje" in analisis_residuos:
            diseno = analisis_residuos["diseño_desmontaje"]
            lines.append("### Diseño para Desmontaje")
            if isinstance(diseno, dict):
                factible = diseno.get('factible', False)
                lines.append(f"**Factible:** {'Sí' if factible else 'No'}")
                if 'mejoras' in diseno:
                    lines.append(f"\n**Recomendaciones:** {diseno['mejoras']}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Section 5: Recomendaciones Finales
    if recomendaciones_finales and isinstance(recomendaciones_finales, dict):
        lines.append("## 5. Recomendaciones Finales")
        lines.append("")
        
        # Material recommendations
        if "recomendaciones_materiales" in recomendaciones_finales:
            rec_mat = recomendaciones_finales["recomendaciones_materiales"]
            if rec_mat and isinstance(rec_mat, list):
                lines.append("### Recomendaciones de Materiales")
                for rec in rec_mat:
                    if isinstance(rec, dict):
                        prioridad = rec.get('prioridad', 'media').upper()
                        desc = rec.get('descripcion', 'N/A')
                        impacto = rec.get('impacto_esperado', 'N/A')
                        lines.append(f"- **[{prioridad}]** {desc}")
                        lines.append(f"  - Impacto Esperado: {impacto}")
                lines.append("")
        
        # Process recommendations
        if "recomendaciones_procesos" in recomendaciones_finales:
            rec_proc = recomendaciones_finales["recomendaciones_procesos"]
            if rec_proc and isinstance(rec_proc, list):
                lines.append("### Recomendaciones de Procesos")
                for rec in rec_proc:
                    if isinstance(rec, dict):
                        prioridad = rec.get('prioridad', 'media').upper()
                        desc = rec.get('descripcion', 'N/A')
                        impacto = rec.get('impacto_esperado', 'N/A')
                        lines.append(f"- **[{prioridad}]** {desc}")
                        lines.append(f"  - Impacto Esperado: {impacto}")
                lines.append("")
        
        # Waste management recommendations
        if "recomendaciones_gestion_residuos" in recomendaciones_finales:
            rec_res = recomendaciones_finales["recomendaciones_gestion_residuos"]
            if rec_res and isinstance(rec_res, list):
                lines.append("### Recomendaciones de Gestión de Residuos")
                for rec in rec_res:
                    if isinstance(rec, dict):
                        prioridad = rec.get('prioridad', 'media').upper()
                        desc = rec.get('descripcion', 'N/A')
                        impacto = rec.get('impacto_esperado', 'N/A')
                        lines.append(f"- **[{prioridad}]** {desc}")
                        lines.append(f"  - Impacto Esperado: {impacto}")
                lines.append("")
        
        # Conclusiones
        if "conclusiones" in recomendaciones_finales:
            lines.append("### Conclusiones")
            lines.append(f"{recomendaciones_finales['conclusiones']}")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Conclusión final
    lines.append("## Conclusión")
    lines.append("")
    lines.append("Este informe completo de Análisis de Ciclo de Vida proporciona un análisis detallado del impacto ambiental del proyecto en todas las fases, desde la selección de materiales hasta la gestión de residuos. Las recomendaciones priorizan prácticas sostenibles y ofrecen pasos concretos para reducir la huella ambiental.")
    lines.append("")
    lines.append(f"*Informe generado a partir de {len(documents)} documentos fuente*")
    
    markdown_report = "\n".join(lines)
    print(f"---REPORT GENERATED ({len(lines)} lines)---")
    
    return {
        "generation": markdown_report,
        "question": question,
        "documents": documents,
    }
