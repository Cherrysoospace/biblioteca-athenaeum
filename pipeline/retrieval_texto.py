"""
pipeline/retrieval_texto.py — Recuperación de chunks de texto por similitud vectorial.

Soporta:
  1. Búsqueda vectorial pura: top-K chunks más similares a un vector de consulta.
  2. Búsqueda híbrida: combina filtros relacionales (idioma, tipo, calificación)
     con similitud coseno sobre Embeddings_Texto.

La tabla Embeddings_Texto usa un índice HNSW (pgvector) que acelera la búsqueda
con el operador <=> (distancia coseno).
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text, select, and_

from models.embeddings_texto import EmbeddingTexto
from models.recursos import Recurso
from core.config import TOP_K, EMBEDDING_DIM_TEXTO

logger = logging.getLogger(__name__)


# ── Búsqueda vectorial pura ───────────────────────────────────────────────────

def buscar_chunks_similares(
    session: Session,
    vector_consulta: list[float],
    top_k: int = TOP_K,
    estrategia: Optional[str] = None,
) -> list[dict]:
    """
    Recupera los `top_k` chunks de texto más similares al vector de consulta
    usando distancia coseno (pgvector operator <=>).

    Args:
        session: sesión SQLAlchemy.
        vector_consulta: embedding de la pregunta (384 dims).
        top_k: número de resultados a retornar.
        estrategia: si se especifica, filtra por estrategia_chunking.

    Returns:
        Lista de dicts con keys: id, recurso_id, chunk_id, chunk_texto,
        estrategia_chunking, score, titulo_recurso.
    """
    # Construir la consulta SQL con pgvector
    # La función 1 - (vec <=> query) convierte distancia coseno a similitud
    vec_str = _vector_to_sql(vector_consulta)

    params: dict = {"top_k": top_k}
    where_estrategia = ""
    if estrategia:
        where_estrategia = "AND et.estrategia_chunking = :estrategia"
        params["estrategia"] = estrategia

    sql = text(f"""
        SELECT
            et.id,
            et.recurso_id,
            et.chunk_id,
            et.chunk_texto,
            et.estrategia_chunking,
            1 - (et.vector_texto_384 <=> '{vec_str}'::vector) AS score,
            r.titulo AS titulo_recurso
        FROM embeddings_texto et
        JOIN recursos r ON r.id = et.recurso_id
        WHERE 1=1 {where_estrategia}
        ORDER BY et.vector_texto_384 <=> '{vec_str}'::vector
        LIMIT :top_k
    """)

    rows = session.execute(sql, params).mappings().all()
    return [dict(row) for row in rows]


# ── Búsqueda híbrida (SQL + vectorial) ───────────────────────────────────────

def buscar_hibrido_texto(
    session: Session,
    vector_consulta: list[float],
    top_k: int = TOP_K,
    filtros: Optional[dict] = None,
    estrategia: Optional[str] = None,
) -> list[dict]:
    """
    Búsqueda híbrida: aplica filtros relacionales sobre Recursos y luego
    rankea los chunks resultantes por similitud vectorial.

    Filtros soportados (en `filtros`):
        idioma          (str)   — coincidencia exacta
        tipo            (str)   — libro, articulo, revista, etc.
        calificacion_min (float) — calificación promedio mínima de reseñas
        fecha_desde     (str)   — fecha de publicación mínima (YYYY-MM-DD)
        fecha_hasta     (str)   — fecha de publicación máxima (YYYY-MM-DD)

    Returns:
        Lista de dicts (misma estructura que buscar_chunks_similares).
    """
    filtros = filtros or {}
    vec_str = _vector_to_sql(vector_consulta)

    # Condiciones SQL adicionales
    condiciones = ["1=1"]
    params: dict = {"top_k": top_k}

    if filtros.get("idioma"):
        condiciones.append("r.idioma = :idioma")
        params["idioma"] = filtros["idioma"]

    if filtros.get("tipo"):
        condiciones.append("r.tipo = :tipo")
        params["tipo"] = filtros["tipo"]

    if filtros.get("calificacion_min"):
        # Subquery: promedio de calificación de reseñas del recurso
        condiciones.append("""
            (SELECT AVG(calificacion) FROM reseñas rz WHERE rz.recurso_id = r.id)
            >= :calificacion_min
        """)
        params["calificacion_min"] = filtros["calificacion_min"]

    if filtros.get("fecha_desde"):
        condiciones.append("r.fecha_publicacion >= :fecha_desde")
        params["fecha_desde"] = filtros["fecha_desde"]

    if filtros.get("fecha_hasta"):
        condiciones.append("r.fecha_publicacion <= :fecha_hasta")
        params["fecha_hasta"] = filtros["fecha_hasta"]

    if estrategia:
        condiciones.append("et.estrategia_chunking = :estrategia")
        params["estrategia"] = estrategia

    where_clause = " AND ".join(condiciones)

    sql = text(f"""
        SELECT
            et.id,
            et.recurso_id,
            et.chunk_id,
            et.chunk_texto,
            et.estrategia_chunking,
            1 - (et.vector_texto_384 <=> '{vec_str}'::vector) AS score,
            r.titulo  AS titulo_recurso,
            r.idioma  AS idioma,
            r.tipo    AS tipo
        FROM embeddings_texto et
        JOIN recursos r ON r.id = et.recurso_id
        WHERE {where_clause}
        ORDER BY et.vector_texto_384 <=> '{vec_str}'::vector
        LIMIT :top_k
    """)

    rows = session.execute(sql, params).mappings().all()
    return [dict(row) for row in rows]


# ── Helpers internos ──────────────────────────────────────────────────────────

def _vector_to_sql(vector: list[float]) -> str:
    """Convierte una lista de floats al formato literal de pgvector: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
