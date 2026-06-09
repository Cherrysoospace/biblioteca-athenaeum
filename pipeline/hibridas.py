"""
pipeline/hibridas.py — Consultas híbridas que combinan SQL relacional con búsqueda vectorial.

Ejemplos de consultas implementadas (alineadas con las 10 consultas de prueba del proyecto):
  Q1  buscar_por_tema_semantico           — semántica pura sobre texto
  Q2  buscar_articulos_por_tema_y_fecha   — relacional + semántica (tipo + fecha + tema)
  Q3  buscar_por_resenas_semanticas       — semántica sobre reseñas
  Q5  buscar_libros_por_idioma_y_tema     — idioma + calificación + semántica
  Q8  buscar_imagenes_historicas          — multimodal + filtro de fecha y tipo
  Q9  buscar_autor_y_similares            — factual SQL + semántica
  Q10 buscar_revistas_por_tema            — tipo=revista + semántica

Cada función retorna una lista de dicts listos para usar en el pipeline RAG.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from core.config import TOP_K
from pipeline.retrieval_texto import (
    buscar_hibrido_texto, _vector_to_sql, ResultadoBusqueda, _aplicar_params,
)
from pipeline.retrieval_imagen import buscar_imagenes_hibrido
from pipeline.embeddings_minilm import get_embedding

logger = logging.getLogger(__name__)


# ── Q1: Búsqueda temática semántica pura ──────────────────────────────────────

def buscar_por_tema_semantico(
    session: Session,
    pregunta: str,
    top_k: int = TOP_K,
    estrategia: Optional[str] = None,
) -> list[dict]:
    """Recupera chunks de texto semánticamente relacionados con la pregunta."""
    vector = get_embedding(pregunta)
    return buscar_hibrido_texto(session, vector, top_k=top_k, estrategia=estrategia)


# ── Q2: Artículos científicos por tema y rango de fechas ─────────────────────

def buscar_articulos_por_tema_y_fecha(
    session: Session,
    pregunta: str,
    fecha_desde: Optional[str] = None,
    fecha_hasta: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Combina: tipo='articulo', rango de fechas y similitud semántica."""
    vector = get_embedding(pregunta)
    filtros = {"tipo": "articulo", "fecha_desde": fecha_desde, "fecha_hasta": fecha_hasta}
    return buscar_hibrido_texto(session, vector, top_k=top_k, filtros=filtros)


# ── Q3: Recursos con reseñas que mencionan un tema ───────────────────────────

def buscar_por_resenas_semanticas(
    session: Session,
    pregunta: str,
    top_k: int = TOP_K,
) -> ResultadoBusqueda:
    """Recupera recursos a partir de la similitud semántica sobre reseñas."""
    vector = get_embedding(pregunta)
    vec_str = _vector_to_sql(vector)

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
        ORDER BY et.vector_texto_384 <=> '{vec_str}'::vector
        LIMIT :top_k
    """
    sql_ejecutado = _aplicar_params(sql_str, {"top_k": top_k})
    sql = text(sql_str)
    rows = session.execute(sql, {"top_k": top_k}).mappings().all()
    return ResultadoBusqueda((dict(row) for row in rows), sql=sql_ejecutado)


# ── Q5: Libros en español con calificación alta y tema existencialista ────────

def buscar_libros_por_idioma_y_tema(
    session: Session,
    pregunta: str,
    idioma: str = "español",
    calificacion_min: float = 4.0,
    top_k: int = TOP_K,
) -> list[dict]:
    """Combina: tipo='libro', idioma, calificación mínima y semántica."""
    vector = get_embedding(pregunta)
    filtros = {"tipo": "libro", "idioma": idioma, "calificacion_min": calificacion_min}
    return buscar_hibrido_texto(session, vector, top_k=top_k, filtros=filtros)


# ── Q8: Imágenes históricas con filtros de fecha y tipo ─────────────────────

def buscar_imagenes_historicas(
    session: Session,
    vector_imagen: list[float],
    tipo_imagen: Optional[str] = None,
    fecha_hasta: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Búsqueda multimodal: imágenes visualmente similares filtradas."""
    filtros = {}
    if tipo_imagen:
        filtros["tipo_imagen"] = tipo_imagen
    if fecha_hasta:
        filtros["fecha_hasta"] = fecha_hasta
    return buscar_imagenes_hibrido(session, vector_imagen, top_k=top_k, filtros=filtros)


# ── Q9: Autor de un recurso + recursos similares ─────────────────────────────

def buscar_autor_y_similares(
    session: Session,
    titulo_recurso: str,
    pregunta_similitud: str,
    top_k: int = TOP_K,
) -> dict:
    """
    Consulta factual (SQL) para encontrar el autor de un recurso por título,
    seguida de búsqueda semántica para recursos similares.
    """
    # 1. Consulta SQL exacta: autor del recurso
    sql_autor = text("""
        SELECT a.nombre, a.tipo, a.biografia, ra.rol_autor, r.titulo
        FROM recursos r
        JOIN recurso_autores ra ON ra.recurso_id = r.id
        JOIN autores a           ON a.id = ra.autor_id
        WHERE LOWER(r.titulo) LIKE LOWER(:titulo)
        LIMIT 5
    """)
    autores = session.execute(
        sql_autor, {"titulo": f"%{titulo_recurso}%"}
    ).mappings().all()

    # 2. Búsqueda semántica de recursos similares
    vector = get_embedding(pregunta_similitud or titulo_recurso)
    similares = buscar_hibrido_texto(session, vector, top_k=top_k)

    return {
        "autores": [dict(a) for a in autores],
        "similares": similares,
    }


# ── Q10: Revistas científicas por tema ────────────────────────────────────────

def buscar_revistas_por_tema(
    session: Session,
    pregunta: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """Filtra tipo='revista' y recupera las más semánticamente relevantes."""
    vector = get_embedding(pregunta)
    return buscar_hibrido_texto(
        session, vector, top_k=top_k, filtros={"tipo": "revista"}
    )


# ── Consulta híbrida genérica ─────────────────────────────────────────────────

def consulta_hibrida(
    session: Session,
    pregunta: str,
    filtros: Optional[dict] = None,
    top_k: int = TOP_K,
    estrategia: Optional[str] = None,
    incluir_imagenes: bool = False,
    vector_imagen: Optional[list[float]] = None,
) -> dict:
    """
    Punto de entrada genérico para consultas híbridas.

    Returns:
        dict con keys: chunks_texto, imagenes, sql_ejecutado
    """
    vector_texto = get_embedding(pregunta)
    chunks = buscar_hibrido_texto(
        session, vector_texto,
        top_k=top_k, filtros=filtros, estrategia=estrategia
    )
    sql_ejecutado = chunks.sql if isinstance(chunks, ResultadoBusqueda) else None

    imagenes = []
    if incluir_imagenes and vector_imagen:
        filtros_img = {}
        if filtros:
            for k in ("tipo_imagen", "recurso_tipo", "idioma", "fecha_desde", "fecha_hasta", "recurso_id"):
                if k in filtros:
                    filtros_img[k] = filtros[k]
        imagenes = buscar_imagenes_hibrido(
            session, vector_imagen, top_k=top_k, filtros=filtros_img or None
        )

    return {"chunks_texto": chunks, "imagenes": imagenes, "sql_ejecutado": sql_ejecutado}
