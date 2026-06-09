"""
scripts/generar_embeddings_consultas.py

Genera embeddings y resultados de búsqueda para consultas existentes
que no los tienen (IDs 6, 7, 8), simulando el pipeline RAG sin crear
nuevas consultas.
"""

import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import logging
from datetime import datetime

from sqlalchemy import select
from core.database import get_session
from models.consultas import Consulta, Evaluacion, ResultadoConsulta
from models.embeddings_consulta import EmbeddingConsulta
from pipeline.embeddings_minilm import get_embedding
from pipeline.retrieval_texto import buscar_hibrido_texto
from pipeline.evaluacion import calcular_metricas_ragas
from pipeline.rag import _construir_contexto, _llamar_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    ids = [6, 7, 8]
    with get_session() as session:
        for cid in ids:
            consulta = session.get(Consulta, cid)
            if not consulta:
                logger.warning("Consulta id=%d no encontrada.", cid)
                continue

            # Verificar si ya tiene embedding
            existing_emb = session.execute(
                select(EmbeddingConsulta).where(EmbeddingConsulta.consulta_id == cid)
            ).scalar_one_or_none()
            if existing_emb:
                logger.info("Consulta %d ya tiene embedding, saltando.", cid)
                continue

            pregunta = consulta.texto_pregunta
            logger.info("Procesando consulta %d: «%s…»", cid, pregunta[:50])

            # 1. Vectorizar
            vector = get_embedding(pregunta)

            emb = EmbeddingConsulta(
                consulta_id=cid,
                vector_texto_384=vector,
            )
            session.add(emb)
            session.flush()
            logger.info("  Embedding creado.")

            # 2. Recuperar chunks
            chunks = buscar_hibrido_texto(session, vector, top_k=5)
            logger.info("  %d chunks recuperados.", len(chunks))

            for pos, chunk in enumerate(chunks, start=1):
                rc = ResultadoConsulta(
                    consulta_id=cid,
                    embedding_texto_id=chunk["id"],
                    embedding_imagen_id=None,
                    score_similitud=round(float(chunk["score"]), 4),
                    posicion=pos,
                )
                session.add(rc)
            session.flush()
            logger.info("  Resultados guardados.")

            # 3. Evaluar
            try:
                contextos = [c["chunk_texto"] for c in chunks]
                contexto_str = _construir_contexto(chunks)
                respuesta = _llamar_llm(pregunta, contexto_str)
                metricas = calcular_metricas_ragas(
                    pregunta=pregunta,
                    respuesta=respuesta,
                    contextos=contextos,
                )
                ev = Evaluacion(
                    consulta_id=cid,
                    faithfulness=round(metricas["faithfulness"], 4),
                    answer_relevancy=round(metricas["answer_relevancy"], 4),
                    context_recall=round(metricas["context_recall"], 4),
                    context_precision=round(metricas.get("context_precision", 0.0), 4),
                    answer_correctness=round(metricas.get("answer_correctness", 0.0), 4),
                    fecha=datetime.utcnow(),
                )
                session.add(ev)
                logger.info("  Evaluación creada.")
            except Exception as exc:
                logger.warning("  Error en evaluación: %s", exc)

        session.commit()
        logger.info("✅ Procesamiento completado.")


if __name__ == "__main__":
    main()
