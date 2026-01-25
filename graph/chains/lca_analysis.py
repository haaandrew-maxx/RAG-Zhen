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

DATASET DE MATERIALES ALTERNATIVOS (Excel/Ecoinvent):
{dataset_materiales}

DOCUMENTOS RECUPERADOS POR RAG:
{rag_documents}

INSTRUCCIONES:
Para cada material principal del proyecto:
1. Identifica alternativas más sostenibles en el dataset o documentación RAG
2. Evalúa:
   - Posibilidad de uso de material reciclado o reutilizado
   - Reducción potencial de emisiones de CO2
   - Alineación con economía circular
3. Si NO hay alternativa viable, justifica por qué
4. NO inventes alternativas que no estén en los datos

Genera un JSON con la siguiente estructura:
{{
    "analisis_materiales": [
        {{
            "material_actual": "nombre del material",
            "impacto_actual_co2": valor,
            "alternativas": [
                {{
                    "material_alternativo": "nombre",
                    "fuente": "Excel/RAG/documento específico",
                    "material_reciclado": true/false,
                    "reduccion_co2_estimada": valor,
                    "porcentaje_reduccion": valor,
                    "viabilidad": "alta/media/baja",
                    "justificacion": "explicación técnica",
                    "economia_circular": {{
                        "reutilizable": true/false,
                        "reciclable": true/false,
                        "biodegradable": true/false
                    }}
                }}
            ],
            "sin_alternativa": {{
                "razon": "justificación si no hay alternativa viable"
            }}
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
