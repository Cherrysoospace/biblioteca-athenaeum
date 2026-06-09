"""
scripts/eliminar_duplicados.py

Elimina Consultas duplicadas que tengan el mismo texto_pregunta,
conservando únicamente la más reciente (por fecha) de cada grupo.

También elimina en cascada:
  - Evaluacion
  - EmbeddingConsulta
  - ResultadoConsulta

Uso:
    python scripts/eliminar_duplicados.py
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import logging
from sqlalchemy import select, func
from core.database import get_session
from models.consultas import Consulta, Evaluacion, ResultadoConsulta
from models.embeddings_consulta import EmbeddingConsulta

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    with get_session() as session:
        # 1. Encontrar textos_pregunta que aparecen más de una vez
        dupes = (
            session.execute(
                select(Consulta.texto_pregunta, func.count(Consulta.id))
                .group_by(Consulta.texto_pregunta)
                .having(func.count(Consulta.id) > 1)
            )
            .all()
        )

        if not dupes:
            logger.info("No hay consultas duplicadas.")
            return

        logger.info("Se encontraron %d textos con duplicados.", len(dupes))

        total_eliminadas = 0
        for texto_pregunta, cnt in dupes:
            # 2. Obtener todas las consultas con ese texto, ordenadas por fecha DESC
            rows = (
                session.execute(
                    select(Consulta.id, Consulta.fecha)
                    .where(Consulta.texto_pregunta == texto_pregunta)
                    .order_by(Consulta.fecha.desc())
                )
                .all()
            )

            # Conservar la más reciente (índice 0), eliminar el resto
            keep_id = rows[0].id
            delete_ids = [r.id for r in rows[1:]]

            for cid in delete_ids:
                # Eliminar en cascada
                session.execute(
                    ResultadoConsulta.__table__.delete()
                    .where(ResultadoConsulta.consulta_id == cid)
                )
                session.execute(
                    EmbeddingConsulta.__table__.delete()
                    .where(EmbeddingConsulta.consulta_id == cid)
                )
                session.execute(
                    Evaluacion.__table__.delete()
                    .where(Evaluacion.consulta_id == cid)
                )
                session.execute(
                    Consulta.__table__.delete()
                    .where(Consulta.id == cid)
                )
                total_eliminadas += 1

            logger.info(
                "  «%s…» → conservada id=%d, eliminadas %d",
                texto_pregunta[:40], keep_id, len(delete_ids),
            )

        session.commit()
        logger.info("Total eliminadas: %d consultas duplicadas.", total_eliminadas)


if __name__ == "__main__":
    main()
