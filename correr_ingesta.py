import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
from core.database import get_session
from pipeline.ingest import ingest_todos
with get_session() as session:
    resultados = ingest_todos(session, vectorizar_imagenes=False)
    session.commit()
    for r in resultados:
        rid = r.get("recurso_id", "?")
        emb = r.get("embeddings_texto", 0)
        err = r.get("errores", [])
        print(f"  recurso {rid}: {emb} emb_texto, errores={len(err)}")
