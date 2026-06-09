from core.database import get_session
from sqlalchemy import text
with get_session() as session:
    total = session.execute(text("SELECT COUNT(*) FROM embeddings_texto WHERE estrategia_chunking = 'title'")).scalar()
    ej = session.execute(text("SELECT recurso_id, chunk_texto FROM embeddings_texto WHERE estrategia_chunking = 'title' LIMIT 5")).all()
    print(f"{total} title chunks total")
    for r in ej:
        print(f"  recurso {r.recurso_id}: {r.chunk_texto}")
