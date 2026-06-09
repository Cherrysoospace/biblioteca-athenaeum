"""
scripts/cleanup_experimentos.py — Limpia tablas de experimentos y reinicia secuencias.

Elimina todo: evaluaciones, resultados_consulta, embeddings_consulta y consultas.
Reinicia las secuencias de PostgreSQL para que los IDs empiecen desde 1.
"""

import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from sqlalchemy import text
from core.database import get_session
from models.consultas import Consulta, ResultadoConsulta, Evaluacion
from models.embeddings_consulta import EmbeddingConsulta


def limpiar_todo():
    with get_session() as session:
        n_consulta = session.query(Consulta).count()
        n_emb = session.query(EmbeddingConsulta).count()
        n_res = session.query(ResultadoConsulta).count()
        n_eval = session.query(Evaluacion).count()

        print(f"Registros actuales:")
        print(f"  consultas:           {n_consulta}")
        print(f"  embeddings_consulta: {n_emb}")
        print(f"  resultados_consulta: {n_res}")
        print(f"  evaluaciones:        {n_eval}")

        session.execute(Evaluacion.__table__.delete())
        session.execute(ResultadoConsulta.__table__.delete())
        session.execute(EmbeddingConsulta.__table__.delete())
        session.execute(Consulta.__table__.delete())

        # Reiniciar secuencias PostgreSQL
        for seq in ["consultas_id_seq", "embeddings_consulta_id_seq",
                     "resultados_consulta_id_seq", "evaluaciones_id_seq"]:
            session.execute(text(f"ALTER SEQUENCE {seq} RESTART WITH 1"))

        session.commit()
        print("\n✅ Tablas limpiadas y secuencias reiniciadas.")


if __name__ == "__main__":
    limpiar_todo()
