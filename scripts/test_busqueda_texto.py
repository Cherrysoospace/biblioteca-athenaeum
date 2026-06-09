"""
scripts/test_busqueda_texto.py

Prueba rápida de búsqueda semántica sobre Embeddings_Texto
usando la columna real `vector_texto_384`.

Ejecutar: python scripts/test_busqueda_texto.py
"""

import sys
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from sqlalchemy import text
from core.database import get_session
from pipeline.embeddings_minilm import get_embedding


def buscar(pregunta: str, top_k: int = 5):
    vector = get_embedding(pregunta)
    vec_str = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"

    sql = text(f"""
        SELECT
            et.id,
            et.recurso_id,
            et.chunk_id,
            et.chunk_texto,
            1 - (et.vector_texto_384 <=> '{vec_str}'::vector) AS score,
            r.titulo AS titulo_recurso
        FROM embeddings_texto et
        JOIN recursos r ON r.id = et.recurso_id
        ORDER BY et.vector_texto_384 <=> '{vec_str}'::vector
        LIMIT :top_k
    """)

    with get_session() as session:
        rows = session.execute(sql, {"top_k": top_k}).mappings().all()

    print(f"\nPregunta: {pregunta}\n")
    print(f"{'Score':>8} | Título")
    print("-" * 60)
    for r in rows:
        print(f"{r.score:>8.4f} | {r.titulo_recurso}")
        print(f"         | → {r.chunk_texto[:100]}...\n")


if __name__ == "__main__":
    pregunta = sys.argv[1] if len(sys.argv) > 1 else "¿Qué libros hablan sobre la evolución de las especies?"
    buscar(pregunta)
