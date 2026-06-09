"""
pipeline/retrieval_imagen.py — Recuperación de imágenes por similitud visual.

Soporta tres modalidades de consulta:
  1. imagen → imagen  : dada una imagen, encuentra imágenes visualmente similares.
  2. texto  → imagen  : dada una descripción textual (CLIP text encoder), encuentra
                        imágenes semánticamente relacionadas.
  3. híbrida imagen   : combina filtros relacionales (tipo_imagen, fecha, recurso)
                        con similitud coseno sobre Embeddings_Imagen.

El índice HNSW sobre embeddings_imagen.vector_embedding acelera las consultas.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from core.config import TOP_K, EMBEDDING_DIM_IMAGEN

logger = logging.getLogger(__name__)


def _vector_to_sql(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vector) + "]"


# ── Modalidad 1: imagen → imagen ──────────────────────────────────────────────

def buscar_imagenes_similares(
    session: Session,
    vector_imagen: list[float],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Recupera las `top_k` imágenes más similares visualmente a un embedding CLIP.

    Args:
        session: sesión SQLAlchemy.
        vector_imagen: embedding CLIP de la imagen consulta (512 dims).
        top_k: número de resultados.

    Returns:
        Lista de dicts con: id (embedding), imagen_id, recurso_id, ruta_archivo,
        tipo_imagen, descripcion, titulo_recurso, score.
    """
    vec_str = _vector_to_sql(vector_imagen)

    sql = text(f"""
        SELECT
            ei.id              AS embedding_id,
            ei.imagen_id,
            ir.recurso_id,
            ir.ruta_archivo,
            ir.tipo_imagen,
            ir.descripcion     AS descripcion_imagen,
            r.titulo           AS titulo_recurso,
            1 - (ei.vector_embedding <=> '{vec_str}'::vector) AS score
        FROM embeddings_imagen ei
        JOIN imagenes_recurso ir ON ir.id = ei.imagen_id
        JOIN recursos r           ON r.id  = ir.recurso_id
        ORDER BY ei.vector_embedding <=> '{vec_str}'::vector
        LIMIT :top_k
    """)

    rows = session.execute(sql, {"top_k": top_k}).mappings().all()
    return [dict(row) for row in rows]


# ── Modalidad 2: texto → imagen ───────────────────────────────────────────────

def buscar_imagenes_por_texto(
    session: Session,
    texto_consulta: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Recupera imágenes cuyo embedding CLIP es más cercano a la descripción textual.
    Usa el encoder de texto de CLIP (vector de 512 dims) para cruzar modalidades.

    Args:
        session: sesión SQLAlchemy.
        texto_consulta: descripción textual de la imagen buscada.
        top_k: número de resultados.

    Returns:
        Misma estructura que buscar_imagenes_similares.
    """
    from pipeline.embeddings_clip import get_embedding_texto_clip
    vector_texto = get_embedding_texto_clip(texto_consulta)
    return buscar_imagenes_similares(session, vector_texto, top_k)


# ── Modalidad 3: híbrida con filtros relacionales ─────────────────────────────

def buscar_imagenes_hibrido(
    session: Session,
    vector_imagen: list[float],
    top_k: int = TOP_K,
    filtros: Optional[dict] = None,
) -> list[dict]:
    """
    Búsqueda híbrida: combina filtros relacionales sobre imágenes y recursos
    con similitud coseno sobre Embeddings_Imagen.

    Filtros soportados (en `filtros`):
        tipo_imagen   (str)  — portada, ilustracion, mapa, fotografia, otro
        recurso_tipo  (str)  — tipo del recurso padre (libro, articulo, ...)
        idioma        (str)  — idioma del recurso padre
        fecha_desde   (str)  — fecha_publicacion >= (YYYY-MM-DD)
        fecha_hasta   (str)  — fecha_publicacion <= (YYYY-MM-DD)
        recurso_id    (int)  — limitar a imágenes de un recurso específico

    Returns:
        Lista de dicts (misma estructura que buscar_imagenes_similares).
    """
    filtros = filtros or {}
    vec_str = _vector_to_sql(vector_imagen)

    condiciones = ["1=1"]
    params: dict = {"top_k": top_k}

    if filtros.get("tipo_imagen"):
        condiciones.append("ir.tipo_imagen = :tipo_imagen")
        params["tipo_imagen"] = filtros["tipo_imagen"]

    if filtros.get("recurso_tipo"):
        condiciones.append("r.tipo = :recurso_tipo")
        params["recurso_tipo"] = filtros["recurso_tipo"]

    if filtros.get("idioma"):
        condiciones.append("r.idioma = :idioma")
        params["idioma"] = filtros["idioma"]

    if filtros.get("fecha_desde"):
        condiciones.append("r.fecha_publicacion >= :fecha_desde")
        params["fecha_desde"] = filtros["fecha_desde"]

    if filtros.get("fecha_hasta"):
        condiciones.append("r.fecha_publicacion <= :fecha_hasta")
        params["fecha_hasta"] = filtros["fecha_hasta"]

    if filtros.get("recurso_id"):
        condiciones.append("ir.recurso_id = :recurso_id")
        params["recurso_id"] = filtros["recurso_id"]

    where_clause = " AND ".join(condiciones)

    sql = text(f"""
        SELECT
            ei.id              AS embedding_id,
            ei.imagen_id,
            ir.recurso_id,
            ir.ruta_archivo,
            ir.tipo_imagen,
            ir.descripcion     AS descripcion_imagen,
            r.titulo           AS titulo_recurso,
            r.tipo             AS tipo_recurso,
            r.idioma,
            r.fecha_publicacion,
            1 - (ei.vector_embedding <=> '{vec_str}'::vector) AS score
        FROM embeddings_imagen ei
        JOIN imagenes_recurso ir ON ir.id = ei.imagen_id
        JOIN recursos r           ON r.id  = ir.recurso_id
        WHERE {where_clause}
        ORDER BY ei.vector_embedding <=> '{vec_str}'::vector
        LIMIT :top_k
    """)

    rows = session.execute(sql, params).mappings().all()
    return [dict(row) for row in rows]


# ── Imagen → texto cruzado ────────────────────────────────────────────────────

def buscar_textos_por_imagen(
    session: Session,
    vector_imagen: list[float],
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Dado el embedding CLIP de una imagen (512 dims), recupera los chunks de texto
    semánticamente relacionados en Embeddings_Texto.

    Nota: requiere que los embeddings de imagen y texto estén en el mismo espacio
    (CLIP es multimodal por diseño). El vector de imagen se proyecta sobre
    los chunks de texto truncando a 384 dims o usando una tabla bridge.

    Implementación actual: usa el texto de descripción de las imágenes más
    similares como puente hacia Embeddings_Texto.
    """
    # Paso 1: encontrar imágenes similares
    imagenes = buscar_imagenes_similares(session, vector_imagen, top_k=top_k * 2)

    if not imagenes:
        return []

    # Paso 2: para cada imagen, recuperar el recurso y sus chunks de texto
    recurso_ids = list({img["recurso_id"] for img in imagenes})

    from pipeline.retrieval_texto import _vector_to_sql as _vec2sql
    from core.config import EMBEDDING_DIM_TEXTO

    # Consulta: chunks de esos recursos ordenados por similitud con la descripción
    # de la primera imagen (puente texto→texto)
    descripcion_puente = " ".join(
        img["descripcion_imagen"] or "" for img in imagenes[:3]
    ).strip()

    if not descripcion_puente:
        return []

    from pipeline.embeddings_minilm import get_embedding
    vec_texto = get_embedding(descripcion_puente)
    vec_str = _vec2sql(vec_texto)

    ids_str = ",".join(str(rid) for rid in recurso_ids)
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
        WHERE et.recurso_id IN ({ids_str})
        ORDER BY et.vector_texto_384 <=> '{vec_str}'::vector
        LIMIT :top_k
    """)

    rows = session.execute(sql, {"top_k": top_k}).mappings().all()
    return [dict(row) for row in rows]
