from core.database import get_session
from sqlalchemy import text
from collections import defaultdict

with get_session() as session:
    rows = session.execute(text("""
        SELECT e.recurso_id, r.titulo, e.chunk_id, e.estrategia_chunking, e.chunk_texto
        FROM embeddings_texto e
        JOIN recursos r ON r.id = e.recurso_id
        ORDER BY e.recurso_id, e.estrategia_chunking, e.chunk_id
    """)).all()

    # Group by resource + strategy
    by_resource = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_resource[r.recurso_id][r.estrategia_chunking].append(r)

    print("=" * 80)
    print("RESUMEN DE CHUNKS POR RECURSO Y ESTRATEGIA")
    print("=" * 80)
    for rid in sorted(by_resource.keys()):
        estrategias = by_resource[rid]
        total_chunks = sum(len(v) for v in estrategias.values())
        titulo = next(r.titulo for v in estrategias.values() for r in v)
        print(f"\n--- Recurso {rid}: {titulo} ({total_chunks} chunks) ---")
        for est in sorted(estrategias.keys()):
            chunks = estrategias[est]
            print(f"  [{est}] {len(chunks)} chunks:")
            for c in chunks:
                preview = c.chunk_texto[:120].replace('\n', ' | ')
                print(f"    chunk {c.chunk_id}: {preview}...")

    # Count total
    total = session.execute(text("SELECT COUNT(*) FROM embeddings_texto")).scalar()
    unique_est = session.execute(text("SELECT DISTINCT estrategia_chunking FROM embeddings_texto")).scalars().all()
    print(f"\n\nTOTAL: {total} chunks total")
    print(f"Estrategias: {list(unique_est)}")
