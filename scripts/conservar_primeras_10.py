"""
scripts/conservar_primeras_10.py

Conserva solo las 10 consultas más antiguas (por fecha) y elimina el resto,
junto con sus dependencias en evaluaciones, embeddings_consulta y resultados_consulta.

Uso:
    python scripts/conservar_primeras_10.py
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import logging

from sqlalchemy import text, select, func
from core.database import get_session
from models.consultas import Consulta, Evaluacion, ResultadoConsulta
from models.embeddings_consulta import EmbeddingConsulta

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    with get_session() as session:
        keep = (
            session.query(Consulta)
            .order_by(Consulta.fecha.asc())
            .limit(10)
            .all()
        )
        keep_ids = {c.id for c in keep}
        logger.info("IDs a conservar (10 más antiguas): %s", sorted(keep_ids))

        all_ids = [r[0] for r in session.execute(select(Consulta.id)).all()]
        delete_ids = [cid for cid in all_ids if cid not in keep_ids]
        logger.info("IDs a eliminar: %d consultas", len(delete_ids))

        if not delete_ids:
            logger.info("No hay consultas que eliminar.")
            return

        (session.query(ResultadoConsulta)
         .filter(ResultadoConsulta.consulta_id.in_(delete_ids))
         .delete(synchronize_session=False))
        (session.query(EmbeddingConsulta)
         .filter(EmbeddingConsulta.consulta_id.in_(delete_ids))
         .delete(synchronize_session=False))
        (session.query(Evaluacion)
         .filter(Evaluacion.consulta_id.in_(delete_ids))
         .delete(synchronize_session=False))
        (session.query(Consulta)
         .filter(Consulta.id.in_(delete_ids))
         .delete(synchronize_session=False))

        session.commit()

        remaining = session.query(Consulta).count()
        logger.info("✅ Quedan %d consultas.", remaining)


if __name__ == "__main__":
    main()
