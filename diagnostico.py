"""
Diagnóstico: verifica la estructura de la BD y el estado de los embeddings.
"""
from core.database import get_session
from sqlalchemy import text

with get_session() as session:
    # Ver columnas de embeddings_texto
    rs = session.execute(text(
        "SELECT column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_name = 'embeddings_texto' "
        "ORDER BY ordinal_position"
    )).all()
    print("Columnas en embeddings_texto:")
    for r in rs:
        print(f"  {r.column_name} ({r.data_type})")

    # Intentar leer vector_texto_384
    try:
        rs2 = session.execute(text(
            "SELECT id, recurso_id, chunk_id, estrategia_chunking, "
            "       length(vector_texto_384::text) > 10 AS tiene_vector "
            "FROM embeddings_texto "
            "LIMIT 10"
        )).all()
        print(f"\nFilas en embeddings_texto (vía vector_texto_384): {len(rs2)}")
        for r in rs2:
            print(f"  id={r.id} recurso_id={r.recurso_id} chunk_id={r.chunk_id} "
                  f"estrategia={r.estrategia_chunking} tiene_vector={r.tiene_vector}")
    except Exception as e:
        print(f"\nERROR con vector_texto_384: {e}")

    # Intentar leer vector_embedding
    try:
        rs3 = session.execute(text(
            "SELECT id, recurso_id, chunk_id, estrategia_chunking, "
            "       length(vector_embedding::text) > 10 AS tiene_vector "
            "FROM embeddings_texto "
            "LIMIT 10"
        )).all()
        print(f"\nFilas en embeddings_texto (vía vector_embedding): {len(rs3)}")
        for r in rs3:
            print(f"  id={r.id} recurso_id={r.recurso_id} chunk_id={r.chunk_id} "
                  f"estrategia={r.estrategia_chunking} tiene_vector={r.tiene_vector}")
    except Exception as e:
        print(f"\nERROR con vector_embedding: {e}")
