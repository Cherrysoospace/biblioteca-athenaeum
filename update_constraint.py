from core.database import get_session
from sqlalchemy import text
with get_session() as session:
    session.execute(text("ALTER TABLE embeddings_texto DROP CONSTRAINT embeddings_texto_estrategia_chunking_check"))
    session.execute(text("""
        ALTER TABLE embeddings_texto ADD CONSTRAINT embeddings_texto_estrategia_chunking_check
        CHECK (estrategia_chunking IN ('fixed_size', 'sentence_aware', 'semantic', 'title'))
    """))
    session.commit()
    print("OK - constraint updated")
