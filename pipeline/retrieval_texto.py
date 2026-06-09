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
from sqlalchemy import text

from models.embeddings_texto import EmbeddingTexto
from models.recursos import Recurso
from core.config import TOP_K

logger = logging.getLogger(__name__)


class ResultadoBusqueda(list):
    """Lista de dicts que además lleva el SQL ejecutado en .sql"""
    def __new__(cls, iterable=(), sql=None):
        obj = super().__new__(cls, iterable)
        obj.sql = sql
        return obj


def _aplicar_params(sql_str: str, params: dict) -> str:
    """Reemplaza :param por sus valores literales para mostrar el SQL completo."""
    # Ordenar por longitud descendente para evitar reemplazos parciales
    result = sql_str
    for k in sorted(params.keys(), key=len, reverse=True):
        v = params[k]
        if isinstance(v, str):
            result = result.replace(f":{k}", f"'{v}'")
        elif v is None:
            result = result.replace(f":{k}", "NULL")
        else:
            result = result.replace(f":{k}", str(v))
    return result


# ── Búsqueda vectorial pura ───────────────────────────────────────────────────

def buscar_chunks_similares(
    session: Session,
    vector_consulta: list[float],
    top_k: int = TOP_K,
    estrategia: Optional[str] = None,
) -> ResultadoBusqueda:
    """
    Recupera los `top_k` chunks de texto más similares al vector de consulta
    usando distancia coseno (pgvector operator <=>).
    """
    vec_str = _vector_to_sql(vector_consulta)

    params: dict = {"top_k": top_k}
    where_estrategia = ""
    if estrategia:
        where_estrategia = "AND et.estrategia_chunking = :estrategia"
        params["estrategia"] = estrategia

    sql_str = f"""
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
    """

    sql_ejecutado = _aplicar_params(sql_str, params)
    sql = text(sql_str)
    rows = session.execute(sql, params).mappings().all()
    return ResultadoBusqueda((dict(row) for row in rows), sql=sql_ejecutado)


# ── Búsqueda híbrida (SQL + vectorial) ───────────────────────────────────────

def buscar_hibrido_texto(
    session: Session,
    vector_consulta: list[float],
    top_k: int = TOP_K,
    filtros: Optional[dict] = None,
    estrategia: Optional[str] = None,
) -> ResultadoBusqueda:
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
        ResultadoBusqueda: lista de dicts con .sql
    """
    filtros = filtros or {}
    vec_str = _vector_to_sql(vector_consulta)

    condiciones = ["1=1"]
    params: dict = {"top_k": top_k}

    if filtros.get("idioma"):
        condiciones.append("r.idioma = :idioma")
        params["idioma"] = filtros["idioma"]

    if filtros.get("tipo"):
        condiciones.append("r.tipo = :tipo")
        params["tipo"] = filtros["tipo"]

    if filtros.get("calificacion_min"):
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

    sql_str = f"""
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
    """

    sql_ejecutado = _aplicar_params(sql_str, params)
    sql = text(sql_str)
    rows = session.execute(sql, params).mappings().all()
    return ResultadoBusqueda((dict(row) for row in rows), sql=sql_ejecutado)


# ── Helpers internos ──────────────────────────────────────────────────────────

def _vector_to_sql(vector: list[float]) -> str:
    """Convierte una lista de floats al formato literal de pgvector: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
