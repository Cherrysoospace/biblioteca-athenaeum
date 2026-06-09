"""Verificar si los vectores son realmente cero."""
from core.database import get_session
from sqlalchemy import text

with get_session() as session:
    rows = session.execute(text(
        "SELECT id, recurso_id, chunk_id, estrategia_chunking, "
        "       vector_texto_384::text AS vec_str "
        "FROM embeddings_texto ORDER BY id"
    )).all()
    for r in rows:
        vec_str = r.vec_str
        # Extract first value from pgvector format: [v1,v2,v3,...]
        first_val = vec_str.split(",")[0].lstrip("[")
        try:
            fv = float(first_val)
        except ValueError:
            fv = 999.0
        all_zeros = all(v.strip() == "0.0" for v in vec_str.strip("[]").split(","))
        print(f"id={r.id} rec={r.recurso_id} ch={r.chunk_id} "
              f"estr={r.estrategia_chunking} "
              f"primer_valor={fv:.6f} todo_cero={all_zeros}")
