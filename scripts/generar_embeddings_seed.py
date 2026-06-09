"""
scripts/generar_embeddings_seed.py

Genera embeddings reales para todas las filas placeholder en:
  - Embeddings_Texto   (all-MiniLM-L6-v2 sobre chunk_texto → vector_texto_384)
  - Embeddings_Imagen  (CLIP texto sobre descripcion de Imagenes_Recurso → vector_embedding)
  - Embeddings_Consulta (all-MiniLM-L6-v2 sobre texto_pregunta → vector_texto_384)

Ejecutar: python scripts/generar_embeddings_seed.py
"""

import logging
import sys
from pathlib import Path

from sqlalchemy import text

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from core.database import get_session
from pipeline.embeddings_minilm import get_embedding
from pipeline.embeddings_clip import get_embedding_texto_clip

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generar_embeddings_seed")


def actualizar_embeddings_texto(session) -> int:
    filas = session.execute(
        text("SELECT id, chunk_texto FROM embeddings_texto ORDER BY id")
    ).all()
    count = 0
    for row in filas:
        texto = row.chunk_texto
        if not texto or not texto.strip():
            continue
        vector = get_embedding(texto)
        session.execute(
            text(
                "UPDATE embeddings_texto SET vector_texto_384 = :vec WHERE id = :id"
            ),
            {"vec": vector, "id": row.id},
        )
        count += 1
    return count


def actualizar_embeddings_imagen(session) -> int:
    filas = session.execute(
        text(
            "SELECT ei.id, img.descripcion "
            "FROM embeddings_imagen ei "
            "JOIN imagenes_recurso img ON img.id = ei.imagen_id "
            "ORDER BY ei.id"
        )
    ).all()
    count = 0
    for row in filas:
        texto = row.descripcion or ""
        if not texto.strip():
            continue
        vector = get_embedding_texto_clip(texto)
        session.execute(
            text(
                "UPDATE embeddings_imagen SET vector_embedding = :vec WHERE id = :id"
            ),
            {"vec": vector, "id": row.id},
        )
        count += 1
    return count


def actualizar_embeddings_consulta(session) -> int:
    filas = session.execute(
        text(
            "SELECT ec.id, c.texto_pregunta "
            "FROM embeddings_consulta ec "
            "JOIN consultas c ON c.id = ec.consulta_id "
            "ORDER BY ec.id"
        )
    ).all()
    count = 0
    for row in filas:
        texto = row.texto_pregunta or ""
        if not texto.strip():
            continue
        vector = get_embedding(texto)
        session.execute(
            text(
                "UPDATE embeddings_consulta SET vector_texto_384 = :vec WHERE id = :id"
            ),
            {"vec": vector, "id": row.id},
        )
        count += 1
    return count


def main():
    with get_session() as session:
        n_texto = actualizar_embeddings_texto(session)
        n_imagen = actualizar_embeddings_imagen(session)
        n_consulta = actualizar_embeddings_consulta(session)

    logger.info("Resumen de embeddings actualizados:")
    logger.info("  Embeddings_Texto   : %d", n_texto)
    logger.info("  Embeddings_Imagen  : %d", n_imagen)
    logger.info("  Embeddings_Consulta: %d", n_consulta)
    logger.info("Total: %d", n_texto + n_imagen + n_consulta)


if __name__ == "__main__":
    main()
