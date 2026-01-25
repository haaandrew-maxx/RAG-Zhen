from graph.nodes.generate import generate
from graph.nodes.retrieve import retrieve
from graph.nodes.grade_documents import grade_documents
from graph.nodes.detect_mode import detect_mode
from graph.nodes.analisis_proyecto import analisis_proyecto
from graph.nodes.analisis_materiales import analisis_materiales
from graph.nodes.analisis_procesos import analisis_procesos
from graph.nodes.gestion_residuos import gestion_residuos
from graph.nodes.recomendaciones_finales import recomendaciones_finales


__all__ = [
    "generate",
    "retrieve",
    "grade_documents",
    "detect_mode",
    "analisis_proyecto",
    "analisis_materiales",
    "analisis_procesos",
    "gestion_residuos",
    "recomendaciones_finales",
]