import pandas as pd
import os
import re
from typing import Dict, List

class MetadataGenerator:
    def __init__(self):
        self.keywords_patterns = {
            'calendario': ['calendario', 'fechas', 'cronograma', 'académico', 'año'],
            'reglamento': ['reglamento', 'normativa', 'normas', 'procedimiento'],
            'admision': ['admisión', 'ingreso', 'postulación', 'matrícula'],
            'titulacion': ['titulación', 'egreso', 'graduación', 'examen de título'],
            'academico': ['académico', 'estudios', 'asignatura', 'ramo', 'carrera'],
            'financiero': ['financiero', 'arancel', 'pago', 'obligación económica'],
            'convivencia': ['convivencia', 'conducta', 'disciplina', 'ética']
        }
    
    def extract_title_from_filename(self, filename: str) -> str:
        """Extrae un título legible del nombre del archivo"""
        # Remover extensión
        name = os.path.splitext(filename)[0]
        
        # Remover caracteres especiales y números al inicio
        name = re.sub(r'^\d+-', '', name)  # Remover prefijos numéricos
        name = re.sub(r'[-_]', ' ', name)  # Reemplazar guiones y underscores con espacios
        
        # Capitalizar palabras
        words = name.split()
        capitalized_words = [word.capitalize() for word in words]
        
        return ' '.join(capitalized_words)
    
    def generate_description(self, filename: str, title: str) -> str:
        """Genera una descripción automática basada en palabras clave del archivo"""
        filename_lower = filename.lower()
        title_lower = title.lower()
        
        description_parts = []
        
        # Detectar tipo de documento
        if any(keyword in filename_lower or keyword in title_lower 
               for keyword in self.keywords_patterns['calendario']):
            description_parts.append("Calendario oficial con fechas importantes")
        
        if any(keyword in filename_lower or keyword in title_lower 
               for keyword in self.keywords_patterns['reglamento']):
            description_parts.append("Documento normativo institucional")
        
        if any(keyword in filename_lower or keyword in title_lower 
               for keyword in self.keywords_patterns['admision']):
            description_parts.append("Procesos de admisión y matrícula")
        
        if any(keyword in filename_lower or keyword in title_lower 
               for keyword in self.keywords_patterns['titulacion']):
            description_parts.append("Procedimientos de titulación y egreso")
        
        if any(keyword in filename_lower or keyword in title_lower 
               for keyword in self.keywords_patterns['academico']):
            description_parts.append("Aspectos académicos y de estudios")
        
        # Añadir año si se detecta
        year_match = re.search(r'20\d{2}', filename)
        if year_match:
            description_parts.append(f"vigente año {year_match.group()}")
        
        return '. '.join(description_parts) + '.' if description_parts else "Documento institucional de la universidad."

def procesar_csv_automatico(archivo_csv: str, archivo_salida: str = None):
    """
    Procesa automáticamente un CSV de documentos y genera metadatos inteligentes
    
    Args:
        archivo_csv (str): Ruta al archivo CSV original
        archivo_salida (str): Ruta para guardar el CSV corregido (opcional)
    """
    
    if archivo_salida is None:
        archivo_salida = archivo_csv.replace('.csv', '_corregido.csv')
    
    # Leer el CSV original
    df = pd.read_csv(archivo_csv)
    
    # Inicializar generador de metadatos
    metadata_gen = MetadataGenerator()
    
    # Procesar cada fila
    nuevas_filas = []
    
    for _, fila in df.iterrows():
        doc_id = fila['doc_id']
        
        # Generar metadatos automáticamente
        title = metadata_gen.extract_title_from_filename(doc_id)
        description = metadata_gen.generate_description(doc_id, title)
        
        nueva_fila = {
            'doc_id': doc_id,
            'title': title,
            'filename': doc_id,  # Usar el doc_id como filename por defecto
            'description': description
        }
        
        nuevas_filas.append(nueva_fila)
    
    # Crear nuevo DataFrame
    df_corregido = pd.DataFrame(nuevas_filas)
    
    # Guardar el CSV corregido
    df_corregido.to_csv(archivo_salida, index=False)
    print(f"✅ CSV corregido guardado como: {archivo_salida}")
    print(f"📊 Total de documentos procesados: {len(df_corregido)}")
    
    return df_corregido

# Script de uso rápido
if __name__ == "__main__":
    # Ejemplo de uso
    archivo_original = "tu_archivo.csv"  # Cambia por la ruta real
    
    try:
        df_resultado = procesar_csv_automatico(archivo_original)
        
        # Mostrar preview de los resultados
        print("\n📋 Preview de los metadatos generados:")
        print(df_resultado[['doc_id', 'title', 'description']].head())
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {archivo_original}")
    except Exception as e:
        print(f"❌ Error al procesar el CSV: {e}")