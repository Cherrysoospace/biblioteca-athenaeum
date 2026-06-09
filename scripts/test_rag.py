"""
scripts/test_rag.py

Ejecuta el pipeline RAG completo (recuperación + LLM) sin depender de
los módulos rotos por el nombre de columna. Usa SQL directo con vector_texto_384.

Ejecutar: python scripts/test_rag.py "¿tu pregunta aquí?"
"""

import json
import sys
import urllib.request
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from sqlalchemy import text
from core.database import get_session
from core.config import LLM_API_KEY, LLM_API_URL, LLM_MODEL, TOP_K
from pipeline.embeddings_minilm import get_embedding

SYSTEM_PROMPT = """Eres el asistente inteligente de la Biblioteca Athenaeum. Responde
usando exclusivamente el contexto proporcionado. Sé conciso y cita el título del recurso."""


def recuperar(pregunta: str, top_k: int = TOP_K) -> list[dict]:
    vector = get_embedding(pregunta)
    vec_str = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
    sql = text(f"""
        SELECT et.id, et.chunk_texto, et.recurso_id,
               1 - (et.vector_texto_384 <=> '{vec_str}'::vector) AS score,
               r.titulo AS titulo_recurso
        FROM embeddings_texto et
        JOIN recursos r ON r.id = et.recurso_id
        ORDER BY et.vector_texto_384 <=> '{vec_str}'::vector
        LIMIT :top_k
    """)
    with get_session() as session:
        rows = session.execute(sql, {"top_k": top_k}).mappings().all()
    return [dict(r) for r in rows]


def preguntar(pregunta: str, contexto: str) -> str:
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"},
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"\nError HTTP {e.code}: {body}")
        raise
    return result["choices"][0]["message"]["content"].strip()


def main():
    pregunta = sys.argv[1] if len(sys.argv) > 1 else "¿Qué libros hablan sobre la evolución de las especies?"
    print(f"\nPregunta: {pregunta}\n")

    chunks = recuperar(pregunta)

    if not chunks:
        print("No se encontraron fragmentos relevantes.")
        return

    print(f"Se recuperaron {len(chunks)} fragmentos:\n")
    for c in chunks:
        print(f"  [{c['score']:.4f}] {c['titulo_recurso']}")
        print(f"       {c['chunk_texto'][:120]}...\n")

    contexto = "\n\n---\n\n".join(
        f"[{c['titulo_recurso']} (score: {c['score']:.3f})]\n{c['chunk_texto']}"
        for c in chunks
    )

    print("Generando respuesta con el LLM...")
    respuesta = preguntar(pregunta, contexto)
    print(f"\nRespuesta:\n{respuesta}")


if __name__ == "__main__":
    main()
