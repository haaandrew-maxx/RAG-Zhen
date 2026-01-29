"""
LCA (Life Cycle Assessment) Analysis Chains
Specialized chains for each step of the sustainability analysis process
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", base_url="https://api.openai.com/v1", temperature=0)

# ============================================================================
# STEP 1: Project Understanding (Comprensión del proyecto)
# ============================================================================
step1_prompt = PromptTemplate(
    input_variables=["datos_proyecto", "description", "bom"],
    template="""
Eres un asistente experto en Análisis de Ciclo de Vida (ACV), sostenibilidad industrial y economía circular.

**PASO 1: COMPRENSIÓN DEL PROYECTO**

Analiza el proyecto proporcionado y genera un resumen estructurado.

DATOS DEL PROYECTO:
{datos_proyecto}

DESCRIPCIÓN DEL PROYECTO:
{description}

BOM (BILL OF MATERIALS):
{bom}

INSTRUCCIONES:
- NO inventes información que no esté en los datos proporcionados
- Si falta información, indícalo explícitamente
- Sé preciso y técnico

Genera un JSON con la siguiente estructura:
{{
    "producto_sistema": "descripción del producto o sistema evaluado",
    "impacto_ambiental_total": {{
        "co2_eq": valor_numerico,
        "unidad": "kg CO2 eq o similar"
    }},
    "distribucion_impacto": {{
        "materiales": {{
            "porcentaje": valor,
            "co2_eq": valor
        }},
        "transporte": {{
            "porcentaje": valor,
            "co2_eq": valor
        }},
        "proceso_productivo": {{
            "porcentaje": valor,
            "co2_eq": valor
        }},
        "uso": {{
            "porcentaje": valor,
            "co2_eq": valor
        }},
        "residuos": {{
            "porcentaje": valor,
            "co2_eq": valor
        }}
    }},
    "materiales_relevantes": [
        {{
            "material": "nombre",
            "cantidad": valor,
            "unidad": "kg/m3/etc",
            "impacto_co2": valor,
            "porcentaje_impacto_total": valor
        }}
    ],
    "procesos_relevantes": [
        {{
            "proceso": "nombre",
            "energia_consumida": valor,
            "tipo_energia": "renovable/no renovable",
            "impacto_co2": valor
        }}
    ]
}}

Devuelve SOLO el JSON, sin texto adicional.
"""
)

step1_chain = step1_prompt | llm | JsonOutputParser()

# ============================================================================
# STEP 2: Material Analysis (Análisis de materiales)
# ============================================================================
step2_prompt = PromptTemplate(
    input_variables=["resumen_proyecto", "dataset_materiales", "rag_documents"],
    template="""
Eres un asistente experto en Análisis de Ciclo de Vida (ACV), sostenibilidad industrial y economía circular.

**PASO 2: ANÁLISIS DE MATERIALES**

RESUMEN DEL PROYECTO:
{resumen_proyecto}

DATOS DE MATERIALES DISPONIBLES:
{dataset_materiales}

DOCUMENTACIÓN ADICIONAL:
{rag_documents}

INSTRUCCIONES CRÍTICAS:
Para cada material principal del proyecto:

1. **BÚSQUEDA ACTIVA DE ALTERNATIVAS:**
   - NO te limites a buscar coincidencias exactas de nombres
   - ANALIZA la función del material en el proyecto (conductor eléctrico, estructura, aislamiento, etc.)
   - BUSCA materiales con función similar, incluso si tienen nombres diferentes
   - Ejemplos:
     * Si el material es COBRE (conductor eléctrico) → considera ALUMINIO como alternativa funcional
     * Si el material es ACERO (estructura) → considera acero reciclado o aluminio según cargas
     * Si el material es PLÁSTICO VIRGEN → busca plásticos reciclados o bioplásticos

2. **ANÁLISIS DE VIABILIDAD:**
   - Evalúa si la alternativa puede cumplir la misma función técnica
   - Compara propiedades físicas relevantes (conductividad, resistencia, peso, etc.)
   - Considera trade-offs (ej: aluminio conduce menos que cobre pero es más ligero)
   - Califica viabilidad: alta (fácil sustitución), media (requiere ajustes), baja (limitaciones técnicas)

3. **EVALUACIÓN AMBIENTAL CORRECTA:**
   - **CRÍTICO:** Compara factores de emisión POR KILOGRAMO (T CO2/kg), NO emisiones totales
   - Busca en los datos el "Factor de Emisión" de cada material
   - Ejemplo correcto:
     * Material actual: Cobre virgen = 0.004 T CO2/kg
     * Material alternativo: Aluminio = 0.012 T CO2/kg  
     * Conclusión: Aluminio tiene MAYOR emisión por kg (no es mejor ambientalmente por este criterio)
   - Para la misma cantidad de material, calcula: reduccion_co2_kg = (factor_actual - factor_alternativo) × cantidad_kg
   - Si factor_alternativo > factor_actual → reducción es NEGATIVA (peor opción ambiental)
   - Considera material reciclado vs virgen (reciclado suele tener menor factor)
   - Evalúa circularidad (reutilizable, reciclable, biodegradable)

4. **VIABILIDAD INTEGRAL:**
   - Viabilidad "alta" solo si: reduce emisiones Y es técnicamente viable
   - Viabilidad "media" si: ligero aumento de emisiones PERO otras ventajas (peso, circularidad, costo)
   - Viabilidad "baja" si: aumenta significativamente emisiones sin ventajas compensatorias
   - Incluye en "consideraciones_tecnicas" el trade-off de emisiones vs otras propiedades

5. **SOLO si NO existe ninguna alternativa funcional:**
   - Marca "sin_alternativa": true
   - Explica por qué (ej: "No hay materiales alternativos con propiedades conductivas comparables en los datos disponibles")

**IMPORTANTE:** Sé proactivo pero PRECISO con las emisiones - usa factores por kg, no totales.

Genera un JSON con la siguiente estructura:
{{
    "analisis_materiales": [
        {{
            "material_actual": "nombre del material",
            "funcion_material": "describe la función (conductor, estructura, aislamiento, etc.)",
            "impacto_actual_co2": valor_total_kg_co2,
            "factor_emision_actual": "valor T CO2/kg del material actual",
            "cantidad_kg": valor,
            "alternativas": [
                {{
                    "material_alternativo": "nombre",
                    "factor_emision_alternativo": "valor T CO2/kg del material alternativo",
                    "material_reciclado": true/false,
                    "reduccion_co2_por_kg": "valor en kg CO2 por cada kg de material (negativo si aumenta)",
                    "reduccion_co2_estimada": "valor total para la cantidad usada (puede ser negativo)",
                    "porcentaje_reduccion": "porcentaje (negativo si aumenta emisiones)",
                    "viabilidad": "alta/media/baja",
                    "justificacion": "explicación técnica de por qué puede sustituir al material actual",
                    "consideraciones_tecnicas": "trade-offs: emisiones, peso, propiedades, costo, etc.",
                    "economia_circular": {{
                        "reutilizable": true/false,
                        "reciclable": true/false,
                        "biodegradable": true/false
                    }}
                }}
            ],
            "sin_alternativa": false
        }}
    ]
}}

Devuelve SOLO el JSON, sin texto adicional.
"""
)

step2_chain = step2_prompt | llm | JsonOutputParser()

# ============================================================================
# STEP 3: Production Process Analysis (Análisis de procesos productivos)
# ============================================================================
step3_prompt = PromptTemplate(
    input_variables=["resumen_proyecto", "dataset_materiales", "rag_documents"],
    template="""
Eres un asistente experto en Análisis de Ciclo de Vida (ACV), sostenibilidad industrial y economía circular.

**PASO 3: ANÁLISIS DE PROCESOS PRODUCTIVOS**

RESUMEN DEL PROYECTO:
{resumen_proyecto}

DATASET DE PROCESOS ALTERNATIVOS:
{dataset_materiales}

DOCUMENTOS RECUPERADOS POR RAG:
{rag_documents}

INSTRUCCIONES:
Para cada proceso productivo identificado:
1. Evalúa si existen procesos alternativos más sostenibles
2. Considera:
   - Consumo energético
   - Uso de electricidad renovable
   - Optimización de procesos (reducción de energía, simplificación)
3. Indica grado de mejora ambiental esperado (bajo/medio/alto)
4. NO inventes procesos que no estén documentados

Genera un JSON con la siguiente estructura:
{{
    "analisis_procesos": [
        {{
            "proceso_actual": "nombre del proceso",
            "consumo_energia_actual": valor,
            "tipo_energia_actual": "renovable/no renovable",
            "impacto_co2_actual": valor,
            "alternativas": [
                {{
                    "proceso_alternativo": "nombre",
                    "fuente": "Excel/RAG/documento específico",
                    "consumo_energia": valor,
                    "tipo_energia": "renovable/no renovable",
                    "reduccion_co2_estimada": valor,
                    "grado_mejora": "bajo/medio/alto",
                    "justificacion": "explicación técnica",
                    "optimizaciones": [
                        "lista de mejoras propuestas"
                    ]
                }}
            ],
            "sin_alternativa": {{
                "razon": "justificación si no hay proceso alternativo viable"
            }}
        }}
    ]
}}

Devuelve SOLO el JSON, sin texto adicional.
"""
)

step3_chain = step3_prompt | llm | JsonOutputParser()

# ============================================================================
# STEP 4: Waste Management Analysis (Gestión de residuos)
# ============================================================================
step4_prompt = PromptTemplate(
    input_variables=["resumen_proyecto", "rag_documents"],
    template="""
Eres un asistente experto en Análisis de Ciclo de Vida (ACV), sostenibilidad industrial y economía circular.

**PASO 4: GESTIÓN DE RESIDUOS**

RESUMEN DEL PROYECTO:
{resumen_proyecto}

DOCUMENTOS RECUPERADOS POR RAG:
{rag_documents}

INSTRUCCIONES:
1. Analiza la gestión de residuos actual del proyecto
2. Evalúa si contempla reutilización, reciclaje o valorización
3. Propón mejoras basadas en:
   - Reutilización de componentes
   - Reciclaje de materiales críticos
   - Diseño para desmontaje
4. Relaciona las propuestas con documentación RAG si existe
5. NO inventes información no respaldada

Genera un JSON con la siguiente estructura:
{{
    "gestion_residuos_actual": {{
        "reutilizacion": true/false,
        "reciclaje": true/false,
        "valorizacion": true/false,
        "descripcion": "descripción de la gestión actual"
    }},
    "mejoras_propuestas": [
        {{
            "tipo": "reutilizacion/reciclaje/diseño_desmontaje/valorizacion",
            "descripcion": "descripción detallada",
            "materiales_criticos": [
                "lista de materiales aplicables"
            ],
            "impacto_ambiental_esperado": "reducción estimada",
            "fuente_conocimiento": "RAG documento específico / mejores prácticas",
            "viabilidad": "alta/media/baja",
            "justificacion": "explicación técnica"
        }}
    ],
    "diseño_desmontaje": {{
        "aplicable": true/false,
        "recomendaciones": [
            "lista de recomendaciones específicas"
        ]
    }}
}}

Devuelve SOLO el JSON, sin texto adicional.
"""
)

step4_chain = step4_prompt | llm | JsonOutputParser()

# ============================================================================
# STEP 5: Final Recommendations (Recomendaciones finales)
# ============================================================================
step5_prompt = PromptTemplate(
    input_variables=["resumen_proyecto", "analisis_materiales", "analisis_procesos", "analisis_residuos"],
    template="""
Eres un asistente experto en Análisis de Ciclo de Vida (ACV), sostenibilidad industrial y economía circular.

**PASO 5: RECOMENDACIONES FINALES**

RESUMEN DEL PROYECTO:
{resumen_proyecto}

ANÁLISIS DE MATERIALES:
{analisis_materiales}

ANÁLISIS DE PROCESOS:
{analisis_procesos}

ANÁLISIS DE GESTIÓN DE RESIDUOS:
{analisis_residuos}

INSTRUCCIONES:
Genera una lista estructurada de recomendaciones finales consolidadas, separadas por categoría.
Para cada recomendación indica:
- Descripción clara
- Justificación ambiental
- Fuente de conocimiento (Excel/Documento RAG)
- Impacto ambiental esperado (cualitativo)

Genera un JSON con la siguiente estructura:
{{
    "resumen_proyecto": {{
        "producto": "nombre del producto/sistema",
        "impacto_total_co2": valor,
        "principales_hotspots": [
            "lista de áreas con mayor impacto"
        ]
    }},
    "recomendaciones_materiales": [
        {{
            "prioridad": "alta/media/baja",
            "descripcion": "descripción clara de la recomendación",
            "material_actual": "nombre",
            "material_propuesto": "nombre",
            "justificacion_ambiental": "explicación técnica",
            "fuente_conocimiento": "Excel / Documento RAG específico",
            "impacto_esperado": "reducción estimada cualitativa",
            "reduccion_co2_estimada": valor
        }}
    ],
    "recomendaciones_procesos": [
        {{
            "prioridad": "alta/media/baja",
            "descripcion": "descripción clara de la recomendación",
            "proceso_actual": "nombre",
            "proceso_propuesto": "nombre",
            "justificacion_ambiental": "explicación técnica",
            "fuente_conocimiento": "Excel / Documento RAG específico",
            "impacto_esperado": "reducción estimada cualitativa",
            "reduccion_co2_estimada": valor
        }}
    ],
    "recomendaciones_gestion_residuos": [
        {{
            "prioridad": "alta/media/baja",
            "descripcion": "descripción clara de la recomendación",
            "tipo": "reutilizacion/reciclaje/valorizacion/diseño",
            "justificacion_ambiental": "explicación técnica",
            "fuente_conocimiento": "Documento RAG específico",
            "impacto_esperado": "reducción estimada cualitativa"
        }}
    ],
    "conclusiones": "Resumen ejecutivo con las conclusiones principales del análisis y el potencial de mejora ambiental global del proyecto"
}}

Devuelve SOLO el JSON, sin texto adicional.
"""
)

step5_chain = step5_prompt | llm | JsonOutputParser()
