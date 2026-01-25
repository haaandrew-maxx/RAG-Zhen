import os
import json
from typing import List, Optional, Dict, Any

import chainlit as cl

import pandas as pd

from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
# Import your LangGraph compiled app
from graph.graph import app as graph_app


# --------------------------
# 1. Helpers to load files
# --------------------------

def load_pdf_as_documents(path: str) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    return loader.load()


def load_generic_file_as_documents(path: str) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader(path)
    return loader.load()


def load_bom_table_as_documents_and_text(path: str) -> (List[Document], str):
    """
    Load a BOM table (.xlsx or .csv) and:
    - Convert it into a markdown-like table string.
    - Wrap it as a Document for retrieval.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    elif ext == ".csv":
        # Auto-detect delimiter
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()

        if ";" in first_line:
            sep = ";"
        elif "," in first_line:
            sep = ","
        else:
            sep = ","  # fallback

        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Formato no soportado como tabla BOM: {ext}")

    # Build markdown-like table
    headers = [str(h) for h in df.columns]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "|:" + "|".join(["----------------" for _ in headers]) + "|"

    rows = []
    for _, row in df.iterrows():
        row_values = [str(v) for v in row.tolist()]
        rows.append("| " + " | ".join(row_values) + " |")

    bom_text = "\n".join([header_line, sep_line] + rows)

    bom_doc = Document(
        page_content=bom_text,
        metadata={
            "source": os.path.basename(path),
            "page": 0,
            "type": "bom_table",
        },
    )

    return [bom_doc], bom_text


def load_uploaded_file(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return load_pdf_as_documents(path)
    else:
        return load_generic_file_as_documents(path)


# --------------------------
# 2. Chat start
# --------------------------

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "👋 Bienvenido/a al asistente RAG basado en LangGraph.\n\n"
            "**🔵 Modo A** (Análisis de Proyecto):\n"
            "Sube **descripción del proyecto** (PDF) + **BOM** (.xlsx/.csv) para análisis específico.\n\n"
            "**🟠 Modo B** (Consulta General):\n"
            "No subas archivos y haz preguntas directas sobre sostenibilidad y medio ambiente.\n\n"
            "Los ficheros cargados solo afectarán a esta conversación."
        )
    ).send()

    files_msg = await cl.AskFileMessage(
        content=(
            "📎 **Opcional:** Sube ficheros solo si quieres análisis de proyecto específico.\n\n"
            "**Formatos aceptados:**\n"
            "- Descripción: PDF o texto\n"
            "- BOM: Excel (.xlsx) o CSV (.csv)\n\n"
            "**Máximo 2 ficheros.**\n\n"
            "⚠️ **Importante:** Para análisis de proyecto (Modo A), necesitas subir AMBOS ficheros "
            "(descripción + BOM).\n\n"
            "✅ Pulsa **Continuar** sin subir nada para consultas generales (Modo B)."
        ),
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv",
            "application/csv",
            "application/vnd.ms-excel",
            ".csv"
        ],
        max_size_mb=20,
        max_files=2,
        timeout=600,
    ).send()

    session_docs: List[Document] = []
    bom_text: str = ""
    description_text: str = ""
    has_bom = False
    has_description = False

    if files_msg and isinstance(files_msg, list):
        for f in files_msg:
            path = f.path
            ext = os.path.splitext(path)[1].lower()

            # BOM formats
            if ext in [".xlsx", ".xls", ".csv"]:
                bom_docs, bom_text = load_bom_table_as_documents_and_text(path)
                session_docs.extend(bom_docs)
                has_bom = True

            # Description or others
            else:
                desc_docs = load_uploaded_file(path)
                session_docs.extend(desc_docs)
                has_description = True

                if not description_text:
                    description_text = (
                        f"Documentación del proyecto desde: {os.path.basename(path)}"
                    )

    # Determine mode and validate
    if has_description and has_bom:
        # Mode A: Both files present
        mode = "A"
        cl.user_session.set("mode", mode)
        await cl.Message(
            content=(
                f"✅ **Modo A activado** 🔵\n\n"
                f"Se han cargado {len(session_docs)} fragmentos de documentación.\n\n"
                "Las respuestas analizarán tu proyecto específico (descripción + BOM) "
                "combinado con la base de conocimiento de sostenibilidad."
            )
        ).send()
    elif has_description or has_bom:
        # Partial upload - warn user
        mode = "B"
        cl.user_session.set("mode", mode)
        await cl.Message(
            content=(
                "⚠️ **Advertencia:** Solo subiste uno de los dos ficheros necesarios.\n\n"
                "Para análisis de proyecto (Modo A) necesitas:\n"
                "- ✅ Descripción del proyecto (PDF)\n"
                "- ✅ BOM (Excel/CSV)\n\n"
                "**Continuando en Modo B** 🟠 (solo base de conocimiento).\n"
                "Los ficheros subidos serán ignorados."
            )
        ).send()
        # Clear partial data
        session_docs = []
        bom_text = ""
        description_text = ""
    else:
        # Mode B: No files uploaded
        mode = "B"
        cl.user_session.set("mode", mode)
        await cl.Message(
            content=(
                "✅ **Modo B activado** 🟠\n\n"
                "No se han subido ficheros.\n\n"
                "Las respuestas se basarán en la base de conocimiento de "
                "sostenibilidad y medio ambiente (60 documentos).\n\n"
                "Haz preguntas generales sobre medio ambiente, economía circular, "
                "normativas, etc."
            )
        ).send()

    cl.user_session.set("session_docs", session_docs)
    cl.user_session.set("bom_text", bom_text)
    cl.user_session.set("description_text", description_text)


# --------------------------
# 3. On each user message
# --------------------------

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()

    session_docs: List[Document] = cl.user_session.get("session_docs") or []
    bom_text: str = cl.user_session.get("bom_text") or ""
    description_text: str = cl.user_session.get("description_text") or ""
    mode: str = cl.user_session.get("mode") or "B"

    # Display current mode
    mode_emoji = "🔵" if mode == "A" else "🟠"
    mode_desc = "Análisis de Proyecto (descripción + BOM + conocimiento)" if mode == "A" else "Consulta General (solo conocimiento)"
    
    await cl.Message(
        content=f"{mode_emoji} **Modo actual:** {mode} - {mode_desc}"
    ).send()

    # Build initial graph state
    initial_state = {
        "question": question,
        "bom": bom_text,
        "description": description_text,
        "generation": "",
        "documents": [],
        "session_docs": session_docs,
        "mode": mode,
    }

    # LangGraph invoke
    try:
        final_state = graph_app.invoke(initial_state)
    except Exception as e:
        await cl.Message(content=f"❌ Error ejecutando LangGraph: {e}").send()
        return

    generation_str = final_state.get("generation", "")
    docs: List[Document] = final_state.get("documents", [])

    # Try parsing JSON
    try:
        data = json.loads(generation_str)
    except:
        await cl.Message(
            content="⚠️ La respuesta del modelo no es un JSON válido.\nSe muestra el texto sin procesar:"
        ).send()
        await cl.Message(content=generation_str).send()
        return

    # Check if this is LCA mode output or regular RAG mode output
    is_lca_mode = "recomendaciones_materiales" in data or "recomendaciones_procesos" in data
    
    if is_lca_mode:
        # New LCA Analysis output structure
        resumen_proyecto = data.get("resumen_proyecto", {})
        recomendaciones_materiales = data.get("recomendaciones_materiales", [])
        recomendaciones_procesos = data.get("recomendaciones_procesos", [])
        recomendaciones_residuos = data.get("recomendaciones_gestion_residuos", [])
        conclusiones = data.get("conclusiones", "")
    else:
        # Legacy RAG mode output structure
        answer = data.get("answer", "")
        sources_json = data.get("sources", [])
        recommendations = data.get("recommendations", [])
        comparative_analysis = data.get("comparative_analysis", {})
        material_substitution = data.get("material_substitution", {})
        circularity_considerations = data.get("circularity_considerations", {})
        highlights = data.get("highlights", [])
        limitations = data.get("limitations", [])
        notas = data.get("notas", "")

    # ---------- UI Cards ----------

    def normalize_to_list(obj):
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return [obj]
        return [str(obj)]

    def normalize_to_dict(obj):
        if isinstance(obj, dict):
            return obj
        return {}

    # ========== LCA MODE OUTPUT ==========
    if is_lca_mode:
        # Resumen del proyecto
        if resumen_proyecto:
            producto = resumen_proyecto.get("producto", "")
            impacto_total = resumen_proyecto.get("impacto_total_co2", "N/A")
            hotspots = resumen_proyecto.get("principales_hotspots", [])
            
            resumen_text = []
            if producto:
                resumen_text.append(f"**Producto/Sistema:** {producto}")
            resumen_text.append(f"**Impacto Total CO₂:** {impacto_total}")
            if hotspots:
                resumen_text.append(f"**Principales Hotspots:**\n" + "\n".join(f"  • {h}" for h in hotspots))
            
            await cl.Message(
                content="📊 **Resumen del Proyecto**",
                elements=[cl.Text(name="Resumen", content="\n".join(resumen_text))],
            ).send()

        # Recomendaciones de Materiales
        if recomendaciones_materiales:
            mat_items = []
            for i, rec in enumerate(recomendaciones_materiales, 1):
                prioridad = rec.get("prioridad", "media").upper()
                icon = "🔴" if prioridad == "ALTA" else "🟡" if prioridad == "MEDIA" else "🟢"
                
                mat_text = [f"{icon} **Recomendación {i}** [{prioridad}]"]
                mat_text.append(f"**Descripción:** {rec.get('descripcion', '')}")
                
                material_actual = rec.get('material_actual', '')
                material_propuesto = rec.get('material_propuesto', '')
                if material_actual and material_propuesto:
                    mat_text.append(f"**Cambio:** {material_actual} → {material_propuesto}")
                
                mat_text.append(f"**Justificación:** {rec.get('justificacion_ambiental', '')}")
                mat_text.append(f"**Impacto Esperado:** {rec.get('impacto_esperado', '')}")
                
                reduccion = rec.get('reduccion_co2_estimada', '')
                if reduccion:
                    mat_text.append(f"**Reducción CO₂:** {reduccion}")
                
                mat_text.append(f"**Fuente:** {rec.get('fuente_conocimiento', '')}")
                mat_items.append("\n".join(mat_text))
            
            content = "\n\n---\n\n".join(mat_items)
            await cl.Message(
                content="🧪 **Recomendaciones de Materiales**",
                elements=[cl.Text(name="Materiales", content=content)],
            ).send()

        # Recomendaciones de Procesos
        if recomendaciones_procesos:
            proc_items = []
            for i, rec in enumerate(recomendaciones_procesos, 1):
                prioridad = rec.get("prioridad", "media").upper()
                icon = "🔴" if prioridad == "ALTA" else "🟡" if prioridad == "MEDIA" else "🟢"
                
                proc_text = [f"{icon} **Recomendación {i}** [{prioridad}]"]
                proc_text.append(f"**Descripción:** {rec.get('descripcion', '')}")
                
                proceso_actual = rec.get('proceso_actual', '')
                proceso_propuesto = rec.get('proceso_propuesto', '')
                if proceso_actual and proceso_propuesto:
                    proc_text.append(f"**Cambio:** {proceso_actual} → {proceso_propuesto}")
                
                proc_text.append(f"**Justificación:** {rec.get('justificacion_ambiental', '')}")
                proc_text.append(f"**Impacto Esperado:** {rec.get('impacto_esperado', '')}")
                
                reduccion = rec.get('reduccion_co2_estimada', '')
                if reduccion:
                    proc_text.append(f"**Reducción CO₂:** {reduccion}")
                
                proc_text.append(f"**Fuente:** {rec.get('fuente_conocimiento', '')}")
                proc_items.append("\n".join(proc_text))
            
            content = "\n\n---\n\n".join(proc_items)
            await cl.Message(
                content="⚙️ **Recomendaciones de Procesos**",
                elements=[cl.Text(name="Procesos", content=content)],
            ).send()

        # Recomendaciones de Gestión de Residuos
        if recomendaciones_residuos:
            res_items = []
            for i, rec in enumerate(recomendaciones_residuos, 1):
                prioridad = rec.get("prioridad", "media").upper()
                icon = "🔴" if prioridad == "ALTA" else "🟡" if prioridad == "MEDIA" else "🟢"
                tipo = rec.get("tipo", "").upper()
                
                res_text = [f"{icon} **Recomendación {i}** [{prioridad}] - {tipo}"]
                res_text.append(f"**Descripción:** {rec.get('descripcion', '')}")
                res_text.append(f"**Justificación:** {rec.get('justificacion_ambiental', '')}")
                res_text.append(f"**Impacto Esperado:** {rec.get('impacto_esperado', '')}")
                res_text.append(f"**Fuente:** {rec.get('fuente_conocimiento', '')}")
                res_items.append("\n".join(res_text))
            
            content = "\n\n---\n\n".join(res_items)
            await cl.Message(
                content="♻️ **Recomendaciones de Gestión de Residuos**",
                elements=[cl.Text(name="Residuos", content=content)],
            ).send()

        # Conclusiones
        if conclusiones:
            await cl.Message(
                content="📝 **Conclusiones**",
                elements=[cl.Text(name="Conclusiones", content=conclusiones)],
            ).send()
    
    # ========== LEGACY RAG MODE OUTPUT ==========
    else:
        # Main Answer
        if answer:
            await cl.Message(
                content="🧾 **Respuesta principal**",
                elements=[cl.Text(name="Respuesta", content=answer)],
            ).send()
        else:
            await cl.Message(
                content="⚠️ No se generó una respuesta.",
            ).send()

        recs = normalize_to_list(recommendations)
        if recs:
            safe_items = []
            for r in recs:
                if isinstance(r, dict):
                    current = r.get("current_material", "")
                    alt = r.get("alternative", "")
                    reason = r.get("reason", "")
                    safe_items.append(f"- {current} → {alt}: {reason}")
                else:
                    safe_items.append(f"- {str(r)}")
            bullets = "\n".join(safe_items)
            await cl.Message(
                content="🛠️ **Recomendaciones técnicas**",
                elements=[cl.Text(name="Recomendaciones", content=bullets)],
            ).send()

        comp = normalize_to_dict(comparative_analysis)
        if comp:
            text = "\n".join(f"**{k}:** {v}" for k, v in comp.items())
            await cl.Message(
                content="⚖️ **Análisis comparativo**",
                elements=[cl.Text(name="Comparativa", content=text)],
            ).send()

        subs = normalize_to_dict(material_substitution)
        if subs:
            text = "\n".join(f"**{k}:** {v}" for k, v in subs.items())
            await cl.Message(
                content="🔄 **Sustitución de materiales**",
                elements=[cl.Text(name="Materiales", content=text)],
            ).send()

        circ = normalize_to_dict(circularity_considerations)
        if circ:
            text = "\n".join(f"**{k}:** {v}" for k, v in circ.items())
            await cl.Message(
                content="♻️ **Consideraciones de circularidad**",
                elements=[cl.Text(name="Circularidad", content=text)],
            ).send()

        if notas:
            await cl.Message(
                content="📝 **Notas adicionales**",
                elements=[cl.Text(name="Notas", content=str(notas))],
            ).send()

        highs = normalize_to_list(highlights)
        if highs:
            bullets = "\n".join(f"- {str(h)}" for h in highs)
            await cl.Message(
                content="⭐ **Aspectos clave**",
                elements=[cl.Text(name="Highlights", content=bullets)],
            ).send()

        lims = normalize_to_list(limitations)
        if lims:
            bullets = "\n".join(f"- {str(l)}" for l in lims)
            await cl.Message(
                content="⚠️ **Limitaciones**",
                elements=[cl.Text(name="Limitaciones", content=bullets)],
            ).send()

        # Sources
        doc_map = {}
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("page"))
            if key not in doc_map:
                doc_map[key] = d.page_content

        source_elements = []

        for i, s in enumerate(sources_json, 1):
            src = s.get("source", "desconocido")
            page = s.get("page", "N/A")
            reason = s.get("reason", "")
            key = (src, page)
            snippet = doc_map.get(key, "")

            lines = [
                f"**Documento:** {src}",
                f"**Página:** {page}",
                f"**Motivo:** {reason}",
            ]

            if snippet:
                lines.append("\n**Fragmento relevante:**\n" + snippet[:800])

            source_elements.append(
                cl.Text(name=f"Fuente {i}", content="\n".join(lines))
            )

        if source_elements:
            await cl.Message(
                content="📚 **Fuentes consultadas**",
                elements=source_elements,
            ).send()