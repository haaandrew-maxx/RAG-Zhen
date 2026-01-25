"""
Ejemplo de uso del sistema LCA con archivo Excel de materiales alternativos
"""

import json
from graph.graph import app
from excel_table_to_rag_text import procesar_excel_formato_ecoinvent, procesar_excel_materiales_json

# ============================================================================
# OPCIÓN 1: Cargar datos desde archivo Excel
# ============================================================================

def ejecutar_analisis_con_excel(excel_path: str):
    """
    Ejecuta análisis LCA cargando materiales alternativos desde Excel
    
    Args:
        excel_path: Ruta al archivo Excel con materiales alternativos
    """
    
    print("=" * 80)
    print("ANÁLISIS LCA CON DATASET EXCEL")
    print("=" * 80)
    print()
    
    # Cargar y procesar Excel
    print(f"Cargando datos desde: {excel_path}")
    try:
        dataset_materiales = procesar_excel_formato_ecoinvent(excel_path)
        print(f"✓ Excel procesado exitosamente")
        print(f"  Longitud del dataset: {len(dataset_materiales)} caracteres")
    except Exception as e:
        print(f"✗ Error al procesar Excel: {e}")
        return
    
    print()
    
    # Datos del proyecto (mismos del ejemplo anterior)
    description = """
Proyecto de fabricación de componentes metálicos para la industria naval.
El proceso incluye:
- Corte de acero
- Soldadura eléctrica
- Tratamiento superficial con pintura anticorrosiva
- Embalaje para transporte

Ubicación: Valencia, España
Producción anual: 500 unidades
Vida útil esperada: 20 años
"""

    bom = """
Material,Cantidad,Unidad,Impacto CO2 (kg)
Acero virgen,100,kg,1850
Pintura anticorrosiva,5,litros,45
Embalaje de madera,2,m3,120
Tornillos de acero,50,unidades,25
"""

    datos_proyecto = {
        "impacto_total_co2": 2040,
        "unidad": "kg CO2 eq",
        "fases": {
            "materiales": {"co2": 1850, "porcentaje": 90.7},
            "proceso_productivo": {"co2": 120, "porcentaje": 5.9},
            "transporte": {"co2": 50, "porcentaje": 2.5},
            "residuos": {"co2": 20, "porcentaje": 0.9}
        }
    }

    question = """
Analiza el impacto ambiental de este proyecto y proporciona recomendaciones 
concretas para mejorar su sostenibilidad usando los materiales alternativos 
del dataset.
"""

    # Ejecutar análisis
    initial_state = {
        "question": question,
        "description": description,
        "bom": bom,
        "datos_proyecto": datos_proyecto,
        "dataset_materiales": dataset_materiales,  # ← Datos del Excel
        "documents": [],
        "mode": "A",
        "generation": "",
    }
    
    print("Ejecutando análisis LCA...")
    print()
    
    try:
        result = app.invoke(initial_state)
        
        # Mostrar resultados
        if "generation" in result:
            resultado_final = json.loads(result["generation"])
            
            print()
            print("=" * 80)
            print("RESULTADOS DEL ANÁLISIS")
            print("=" * 80)
            print()
            
            # Guardar resultado
            output_file = "lca_result_con_excel.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(resultado_final, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Análisis completado")
            print(f"✓ Resultados guardados en: {output_file}")
            print()
            
            # Resumen
            resumen = resultado_final.get("resumen_proyecto", {})
            print(f"Producto: {resumen.get('producto', 'N/A')}")
            print(f"Impacto CO2: {resumen.get('impacto_total_co2', 'N/A')} kg")
            print(f"Recomendaciones de materiales: {len(resultado_final.get('recomendaciones_materiales', []))}")
            print(f"Recomendaciones de procesos: {len(resultado_final.get('recomendaciones_procesos', []))}")
            print()
            
            return resultado_final
        
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR EN EL ANÁLISIS")
        print("=" * 80)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EJEMPLO: Crear un Excel de ejemplo si no existe
# ============================================================================

def crear_excel_ejemplo():
    """
    Crea un archivo Excel de ejemplo con materiales alternativos
    """
    try:
        import pandas as pd
        
        # Datos de ejemplo
        materiales = pd.DataFrame({
            'Material': [
                'Acero reciclado',
                'Aluminio reciclado', 
                'Plástico reciclado (HDPE)',
                'Madera certificada FSC',
                'Hormigón bajo carbono',
                'Vidrio reciclado',
            ],
            'Categoría': [
                'Material',
                'Material',
                'Material', 
                'Material',
                'Material',
                'Material',
            ],
            'Tipo': [
                'Reciclado',
                'Reciclado',
                'Reciclado',
                'Sostenible',
                'Bajo carbono',
                'Reciclado',
            ],
            'Impacto_CO2_kg_ton': [
                550,
                800,
                1200,
                120,
                180,
                380,
            ],
            'Reduccion_Porcentaje': [
                70,
                95,
                75,
                85,
                60,
                80,
            ],
            'Disponibilidad': [
                'Alta',
                'Media',
                'Alta',
                'Alta',
                'Media',
                'Media',
            ],
            'Fuente': [
                'Ecoinvent 3.8',
                'Ecoinvent 3.8',
                'PlasticsEurope',
                'FSC Database',
                'Cement Association',
                'Glass Alliance',
            ],
            'Aplicaciones': [
                'Estructuras, componentes mecánicos',
                'Componentes ligeros, transporte',
                'Embalaje, contenedores',
                'Embalaje, construcción',
                'Fundaciones, estructuras',
                'Envases, ventanas',
            ],
        })
        
        procesos = pd.DataFrame({
            'Proceso': [
                'Soldadura con energía solar',
                'Pintura en polvo (electrostática)',
                'Corte por láser eficiente',
                'Fundición con energía renovable',
            ],
            'Categoría': [
                'Proceso',
                'Proceso',
                'Proceso',
                'Proceso',
            ],
            'Energia': [
                '150 kWh',
                '80 kWh',
                '120 kWh',
                '200 kWh',
            ],
            'Tipo_Energia': [
                'Solar renovable',
                'Eléctrica',
                'Eléctrica optimizada',
                'Renovable (eólica/solar)',
            ],
            'Reduccion_Porcentaje': [
                80,
                40,
                35,
                75,
            ],
            'Inversion': [
                'Media',
                'Baja',
                'Alta',
                'Alta',
            ],
            'Amortizacion': [
                '5-7 años',
                '3-4 años',
                '8-10 años',
                '6-8 años',
            ],
            'Fuente': [
                'Renewable Energy DB',
                'Coating Association',
                'Laser Manufacturers',
                'Green Foundry Initiative',
            ],
        })
        
        # Guardar en Excel con múltiples hojas
        excel_path = 'datos_materiales_alternativos_ejemplo.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            materiales.to_excel(writer, sheet_name='Materiales', index=False)
            procesos.to_excel(writer, sheet_name='Procesos', index=False)
        
        print(f"✓ Excel de ejemplo creado: {excel_path}")
        print(f"  - Hoja 'Materiales': {len(materiales)} registros")
        print(f"  - Hoja 'Procesos': {len(procesos)} registros")
        print()
        
        return excel_path
        
    except ImportError:
        print("✗ Error: pandas y openpyxl son necesarios")
        print("  Instalar con: pip install pandas openpyxl")
        return None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import os
    
    # Ruta a tu archivo Excel
    # Opción 1: Usar tu propio archivo
    excel_path = "tu_archivo_materiales.xlsx"
    
    # Opción 2: Crear y usar archivo de ejemplo
    if not os.path.exists(excel_path):
        print("Archivo Excel no encontrado. Creando ejemplo...")
        print()
        excel_path = crear_excel_ejemplo()
        
        if excel_path is None:
            print("No se pudo crear el archivo de ejemplo")
            exit(1)
    
    # Ejecutar análisis
    ejecutar_analisis_con_excel(excel_path)
