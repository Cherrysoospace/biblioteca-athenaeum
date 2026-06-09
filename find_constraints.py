from core.database import get_session
from sqlalchemy import text
with get_session() as session:
    rows = session.execute(text("""
        SELECT conname, contype, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'embeddings_texto'::regclass
    """)).all()
    for r in rows:
        print(r)
