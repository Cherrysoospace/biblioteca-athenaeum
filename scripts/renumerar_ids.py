"""
scripts/renumerar_ids.py

Renumera los IDs de la tabla `consultas` secuencialmente desde 1,
actualizando todas las FK que apuntan a `consulta_id` en:
  - evaluaciones
  - embedding_consulta
  - resultado_consulta

Resetea la secuencia de PostgreSQL para que los nuevos INSERT sigan desde 72.

Uso:
    python scripts/renumerar_ids.py
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import logging

from sqlalchemy import text, select, func
from core.database import get_session

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


FK_CONSTRAINTS = [
    "evaluaciones_consulta_id_fkey",
    "embeddings_consulta_consulta_id_fkey",
    "resultados_consulta_consulta_id_fkey",
]


def main():
    with get_session() as session:
        # 1. Obtener total
        total = session.scalar(select(func.count(text("1"))).select_from(text("consultas")))
        logger.info("Total consultas: %d", total)

        if total == 0:
            logger.info("No hay consultas que renumerar.")
            return

        # 2. Bajar FK que apuntan a consultas.id
        fk_table_map = {
            "evaluaciones_consulta_id_fkey": "evaluaciones",
            "embeddings_consulta_consulta_id_fkey": "embeddings_consulta",
            "resultados_consulta_consulta_id_fkey": "resultados_consulta",
        }
        for fk_name, table in fk_table_map.items():
            session.execute(text(
                f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {fk_name}"
            ))
            logger.info("FK %s eliminada.", fk_name)

        # 3. Crear tabla temporal de mapeo old_id → new_id
        session.execute(text("""
            CREATE TEMP TABLE _id_map AS
            SELECT id AS old_id, row_number() OVER (ORDER BY id) AS new_id
            FROM consultas
        """))
        logger.info("Tabla temporal _id_map creada.")

        # 4. Actualizar FK en resultados_consulta
        session.execute(text("""
            UPDATE resultados_consulta rc
            SET consulta_id = m.new_id
            FROM _id_map m
            WHERE rc.consulta_id = m.old_id
        """))
        logger.info("resultados_consulta actualizada.")

        # 5. Actualizar FK en embeddings_consulta
        session.execute(text("""
            UPDATE embeddings_consulta ec
            SET consulta_id = m.new_id
            FROM _id_map m
            WHERE ec.consulta_id = m.old_id
        """))
        logger.info("embeddings_consulta actualizada.")

        # 6. Actualizar FK en evaluaciones
        session.execute(text("""
            UPDATE evaluaciones e
            SET consulta_id = m.new_id
            FROM _id_map m
            WHERE e.consulta_id = m.old_id
        """))
        logger.info("evaluaciones actualizada.")

        # 7. Renumerar consultas.id
        session.execute(text("""
            UPDATE consultas c
            SET id = m.new_id
            FROM _id_map m
            WHERE c.id = m.old_id
        """))
        logger.info("consultas.id renumerada.")

        # 8. Resetear la secuencia al max_id + 1
        session.execute(text("""
            SELECT setval(
                pg_get_serial_sequence('consultas', 'id'),
                COALESCE((SELECT MAX(id) + 1 FROM consultas), 1),
                FALSE
            )
        """))
        logger.info("Secuencia reseteada.")

        # 9. Re-crear FKs
        session.execute(text("""
            ALTER TABLE evaluaciones
            ADD CONSTRAINT evaluaciones_consulta_id_fkey
            FOREIGN KEY (consulta_id) REFERENCES consultas(id)
        """))
        session.execute(text("""
            ALTER TABLE embeddings_consulta
            ADD CONSTRAINT embeddings_consulta_consulta_id_fkey
            FOREIGN KEY (consulta_id) REFERENCES consultas(id)
        """))
        session.execute(text("""
            ALTER TABLE resultados_consulta
            ADD CONSTRAINT resultados_consulta_consulta_id_fkey
            FOREIGN KEY (consulta_id) REFERENCES consultas(id)
        """))
        logger.info("FKs re-creadas.")

        # 10. Limpiar
        session.execute(text("DROP TABLE IF EXISTS _id_map"))

        session.commit()
        logger.info("✅ Renumeración completada. IDs del 1 al %d.", total)


if __name__ == "__main__":
    main()
