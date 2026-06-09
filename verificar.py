"""Verificar estado de los embeddings después del ingest."""
from core.database import get_session
from sqlalchemy import text

with get_session() as session:
    total = session.execute(text("SELECT COUNT(*) FROM embeddings_texto")).scalar()
    print(f"Total filas: {total}")

    rows = session.execute(text(
        "SELECT id, recurso_id, chunk_id, estrategia_chunking, "
        "       (vector_texto_384::text LIKE '[0%' OR vector_texto_384::text = '{}') AS es_cero "
        "FROM embeddings_texto ORDER BY id"
    )).all()
    for r in rows:
        print(f"  id={r.id} rec={r.recurso_id} chunk={r.chunk_id} "
              f"est={r.estrategia_chunking} es_cero={r.es_cero}")
