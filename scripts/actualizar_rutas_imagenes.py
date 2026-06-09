"""
scripts/actualizar_rutas_imagenes.py

Actualiza las rutas de las imágenes en la BD (Imagenes_Recurso.ruta_archivo)
para que apunten a los archivos reales en data/media/.

Uso:
    python scripts/actualizar_rutas_imagenes.py
"""

import os
import sys
from pathlib import Path

# Asegura que el directorio raíz del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import get_session
from sqlalchemy import text

MEDIA_DIR = Path(__file__).resolve().parent.parent / "data" / "media"

# Mapeo: (recurso_id, ruta_antigua_en_bd) -> nombre_archivo_real
MAPPING = {
    (1, "/media/portadas/laberinto_soledad.jpg"): "el_laberinto_de_la_soledad.jpg",
    (2, "/media/portadas/cien_años_soledad.jpg"): "cien_años_de_soledad.jpg",
    (3, "/media/ilustraciones/ia_medicina_fig1.jpg"): "Inteligencia_artificial_en_medicina.jpg",
    (7, "/media/portadas/arquitectura_moderna.jpg"): "historia_arquitectura_moderna.jpg",
    (9, "/media/ilustraciones/neurociencia_fig1.jpg"): "neurociencia_cognitiva.jpg",
    (10, "/media/portadas/condicion_humana.jpg"): "la_condicion_humana.jpg",
    (11, "/media/portadas/revista_ecologia.jpg"): "revista_iberoamericana_de_ecologia.jpg",
    (12, "/media/portadas/origen_especies.jpg"): "el_origen_de_las_especies.jpg",
}

# Imágenes nuevas para recursos que no tenían ninguna
NUEVAS_IMAGENES = [
    (5, "filosofia_latinoamericana.jpg", "portada", "Portada de la Revista de Filosofía Latinoamericana."),
    (6, "cambio_climatico_evidencias_y_proyecciones.jpg", "portada", "Portada del artículo sobre cambio climático."),
]


def actualizar():
    with get_session() as session:
        # 1. Actualizar rutas existentes
        for (recurso_id, ruta_vieja), filename in MAPPING.items():
            new_path = f"data/media/{filename}"
            result = session.execute(
                text("UPDATE Imagenes_Recurso SET ruta_archivo = :new WHERE recurso_id = :rid AND ruta_archivo = :old"),
                {"new": new_path, "rid": recurso_id, "old": ruta_vieja},
            )
            if result.rowcount > 0:
                print(f"  [OK] Recurso {recurso_id}: {ruta_vieja} -> {new_path}")
            else:
                print(f"  [!] Recurso {recurso_id}: no se encontró fila con ruta '{ruta_vieja}'")

        # 2. Insertar imágenes nuevas para recursos que no tenían
        for recurso_id, filename, tipo, descripcion in NUEVAS_IMAGENES:
            filepath = f"data/media/{filename}"
            existing = session.execute(
                text("SELECT COUNT(*) FROM Imagenes_Recurso WHERE recurso_id = :rid"),
                {"rid": recurso_id},
            ).scalar()
            if existing > 0:
                print(f"  [!] Recurso {recurso_id} ya tiene imágenes, se omite inserción.")
                continue
            session.execute(
                text("""
                    INSERT INTO Imagenes_Recurso (recurso_id, ruta_archivo, tipo_imagen, descripcion)
                    VALUES (:rid, :ruta, :tipo, :desc)
                """),
                {"rid": recurso_id, "ruta": filepath, "tipo": tipo, "desc": descripcion},
            )
            print(f"  [OK] Recurso {recurso_id}: insertada nueva imagen -> {filepath}")

        print("\n[OK] Actualización completada.")


if __name__ == "__main__":
    actualizar()
