"""Check actual vector representation in DB."""
from core.database import get_session
from sqlalchemy import text
import ast

with get_session() as session:
    rows = session.execute(text(
        "SELECT id, recurso_id, chunk_id, estrategia_chunking, "
        "       vector_texto_384::text AS vec_str "
        "FROM embeddings_texto ORDER BY id LIMIT 5"
    )).all()
    for r in rows:
        vec_str = r.vec_str
        # Show first 100 chars of vector
        preview = vec_str[:100]
        # Parse and check sum of absolute values
        vec = ast.literal_eval(vec_str)
        total = sum(abs(v) for v in vec)
        nonzero = sum(1 for v in vec if abs(v) > 1e-10)
        print(f"id={r.id} rec={r.recurso_id} ch={r.chunk_id} {r.estrategia_chunking}")
        print(f"  preview={preview}...")
        print(f"  sum_abs={total:.6f} nonzero_dims={nonzero}/384")
