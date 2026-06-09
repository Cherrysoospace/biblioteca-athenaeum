"""
scripts/poblar_consultas.py — Ejecuta el pipeline RAG completo para las 20 preguntas
del dataset de evaluación y persiste los resultados en las tablas:
  consultas, embeddings_consulta, resultados_consulta.

Uso: python scripts/poblar_consultas.py
"""

import sys
import time
import logging
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from core.database import get_session
from data.eval_dataset import EVAL_DATASET
from pipeline.rag import run_rag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("poblar_consultas")

USUARIO_ID = 1


def main():
    total = len(EVAL_DATASET)
    ok = 0
    errors = []

    print(f"Ejecutando RAG para {total} preguntas (usuario_id={USUARIO_ID})...\n")

    for idx, item in enumerate(EVAL_DATASET):
        pregunta = item["pregunta"]
        ground_truth = item.get("ground_truth")

        print(f"[{idx + 1}/{total}] {pregunta[:60]}...")

        if idx > 0:
            time.sleep(1.5)

        try:
            with get_session() as session:
                resultado = run_rag(
                    session=session,
                    usuario_id=USUARIO_ID,
                    pregunta=pregunta,
                    evaluar=False,
                    ground_truth=ground_truth,
                )

            error = resultado.get("error")
            if error:
                print(f"  ❌ Error: {error[:120]}")
                errors.append({"pregunta": pregunta, "error": error})
            else:
                cid = resultado["consulta_id"]
                n_ctx = len(resultado.get("contexto", []))
                print(f"  ✅ consulta_id={cid} | {n_ctx} chunks recuperados")
                ok += 1

        except Exception as exc:
            print(f"  ❌ Excepción: {exc}")
            errors.append({"pregunta": pregunta, "error": str(exc)})

    print(f"\n{'='*50}")
    print(f"Completado: {ok}/{total} exitosas")
    if errors:
        print(f"Fallos: {len(errors)}")
        for e in errors:
            print(f"  - {e['pregunta'][:60]}: {e['error'][:100]}")


if __name__ == "__main__":
    main()
