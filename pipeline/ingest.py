"""
pipeline/ingest.py — Orquestador de ingesta completa de un recurso.

Flujo por recurso:
  1. Lee el recurso (titulo, descripcion) y sus campos vectorizables.
  2. Para cada estrategia de chunking solicitada:
       a. Aplica chunking sobre los campos de texto.
       b. Genera embeddings con MiniLM.
       c. Inserta en Embeddings_Texto.
  3. Para cada imagen asociada (si vectorizar_imagenes=True):
       a. Genera embedding CLIP desde la ruta de archivo.
       b. Inserta en Embeddings_Imagen (upsert por imagen_id).

Retorna un dict con estadísticas de la ingesta.
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select

from models.recursos import Recurso, Autor, ImagenRecurso
from models.embeddings_texto import EmbeddingTexto
from models.embeddings_imagen import EmbeddingImagen
from pipeline.chunking import chunk_texto
from pipeline.embeddings_minilm import get_embeddings_batch
from pipeline.embeddings_clip import get_embedding_imagen

logger = logging.getLogger(__name__)

# Campos de texto que se vectorizan para cada recurso
_CAMPOS_TEXTO = ["descripcion", "titulo"]


def _recolectar_textos(recurso: Recurso, session: Session) -> str:
    """
    Construye el texto principal a chunkear para un recurso:
    título + descripción + biografías de autores + descripciones de imágenes.
    """
    partes = []

    if recurso.titulo:
        partes.append(recurso.titulo)
    if recurso.descripcion:
        partes.append(recurso.descripcion)

    # Biografías de autores relacionados
    from models.recursos import RecursoAutor
    autor_ids = [
        row.autor_id
        for row in session.execute(
            select(RecursoAutor.autor_id).where(RecursoAutor.recurso_id == recurso.id)
        ).all()
    ]
    if autor_ids:
        autores = session.execute(
            select(Autor).where(Autor.id.in_(autor_ids))
        ).scalars().all()
        for autor in autores:
            if autor.biografia:
                partes.append(autor.biografia)

    # Descripciones de imágenes
    for img in recurso.imagenes:
        if img.descripcion:
            partes.append(img.descripcion)

    return "\n\n".join(partes)


def ingest_recurso(
    session: Session,
    recurso_id: int,
    estrategias: list[str] | None = None,
    vectorizar_imagenes: bool = True,
) -> dict:
    """
    Ingesta completa de un recurso. Idempotente: si ya existen embeddings
    con la misma (recurso_id, chunk_id, estrategia), los sobrescribe.

    Args:
        session: sesión SQLAlchemy activa.
        recurso_id: ID de la fila en Recursos.
        estrategias: lista de estrategias a aplicar. Si None, aplica las tres.
        vectorizar_imagenes: si True, genera embeddings CLIP para las imágenes.

    Returns:
        dict con keys: chunks_total, embeddings_texto, embeddings_imagen, errores.
    """
    if estrategias is None:
        estrategias = ["fixed_size", "sentence_aware", "semantic"]

    recurso: Optional[Recurso] = session.get(Recurso, recurso_id)
    if recurso is None:
        raise ValueError(f"Recurso con id={recurso_id} no encontrado.")

    logger.info("Iniciando ingesta de recurso id=%d título='%s'", recurso_id, recurso.titulo)

    stats = {"chunks_total": 0, "embeddings_texto": 0, "embeddings_imagen": 0, "errores": []}
    texto_base = _recolectar_textos(recurso, session)

    # ── 1. Embeddings de texto por estrategia ─────────────────────────────────
    for estrategia in estrategias:
        logger.info("  Chunking estrategia=%s ...", estrategia)
        chunks = chunk_texto(texto_base, estrategia=estrategia)
        stats["chunks_total"] += len(chunks)

        if not chunks:
            logger.warning("  Sin chunks para recurso_id=%d estrategia=%s", recurso_id, estrategia)
            continue

        vectores = get_embeddings_batch(chunks)

        for chunk_id, (texto, vector) in enumerate(zip(chunks, vectores), start=1):
            # Upsert: eliminar si ya existe la combinación
            existing = session.execute(
                select(EmbeddingTexto).where(
                    EmbeddingTexto.recurso_id == recurso_id,
                    EmbeddingTexto.chunk_id == chunk_id,
                    EmbeddingTexto.estrategia_chunking == estrategia,
                )
            ).scalar_one_or_none()

            if existing:
                existing.chunk_texto = texto
                existing.vector_texto_384 = vector
            else:
                emb = EmbeddingTexto(
                    recurso_id=recurso_id,
                    chunk_id=chunk_id,
                    chunk_texto=texto,
                    estrategia_chunking=estrategia,
                    vector_texto_384=vector,
                )
                session.add(emb)

            stats["embeddings_texto"] += 1

        session.flush()
        logger.info("  %d embeddings de texto insertados (estrategia=%s)", len(chunks), estrategia)

    # ── 2. Embeddings de imagen (CLIP) ─────────────────────────────────────────
    if vectorizar_imagenes:
        for imagen in recurso.imagenes:
            try:
                vector = get_embedding_imagen(imagen.ruta_archivo)
                existing = session.execute(
                    select(EmbeddingImagen).where(EmbeddingImagen.imagen_id == imagen.id)
                ).scalar_one_or_none()

                if existing:
                    existing.vector_embedding = vector
                else:
                    emb_img = EmbeddingImagen(
                        imagen_id=imagen.id,
                        vector_embedding=vector,
                    )
                    session.add(emb_img)

                stats["embeddings_imagen"] += 1
            except Exception as exc:
                msg = f"Error vectorizando imagen id={imagen.id}: {exc}"
                logger.error(msg)
                stats["errores"].append(msg)

        session.flush()

    logger.info(
        "Ingesta completada recurso_id=%d: %d chunks, %d emb_texto, %d emb_imagen, %d errores",
        recurso_id,
        stats["chunks_total"],
        stats["embeddings_texto"],
        stats["embeddings_imagen"],
        len(stats["errores"]),
    )
    return stats


def ingest_todos(
    session: Session,
    estrategias: list[str] | None = None,
    vectorizar_imagenes: bool = True,
) -> list[dict]:
    """
    Itera sobre todos los recursos en la base de datos y los ingesta.
    Útil para el seed inicial o reindexado completo.
    """
    from sqlalchemy import select as sel
    recurso_ids = session.execute(sel(Recurso.id)).scalars().all()
    resultados = []
    for rid in recurso_ids:
        try:
            r = ingest_recurso(session, rid, estrategias, vectorizar_imagenes)
            r["recurso_id"] = rid
            resultados.append(r)
        except Exception as exc:
            logger.error("Error en ingesta recurso_id=%d: %s", rid, exc)
            resultados.append({"recurso_id": rid, "error": str(exc)})
    return resultados
