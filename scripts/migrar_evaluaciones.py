"""
scripts/migrar_evaluaciones.py — Agrega columnas context_precision y answer_correctness.

Ejecutar: python scripts/migrar_evaluaciones.py
"""

import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from sqlalchemy import text
from core.database import engine


def main():
    with engine.connect() as conn:
        for col in ("context_precision", "answer_correctness"):
            conn.execute(
                text(f"ALTER TABLE evaluaciones ADD COLUMN IF NOT EXISTS {col} NUMERIC(5,4)")
            )
        conn.commit()
        print("Columnas agregadas exitosamente.")


if __name__ == "__main__":
    main()
