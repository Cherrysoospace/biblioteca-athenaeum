"""
core/config.py — Configuración centralizada vía variables de entorno.

Variables requeridas (definir en .env o entorno):
  DATABASE_URL   — PostgreSQL con pgvector, e.g.:
                   postgresql://user:pass@host:5432/athenaeum
  LLM_API_KEY    — Clave de Groq o HuggingFace Inference API
  LLM_PROVIDER   — "groq" | "huggingface"  (default: "groq")
  LLM_MODEL      — Nombre del modelo LLM (default: "llama3-8b-8192")

Variables opcionales:
  MINILM_MODEL   — Modelo sentence-transformers para texto (default: all-MiniLM-L6-v2)
  CLIP_MODEL     — Modelo CLIP para imágenes (default: openai/clip-vit-base-patch32)
  CHUNK_SIZE     — Tokens por chunk fixed-size (default: 256)
  CHUNK_OVERLAP  — Tokens de overlap fixed-size (default: 32)
  SENTENCE_MAX   — Oraciones máx. por chunk sentence-aware (default: 5)
  SEMANTIC_THR   — Umbral coseno para corte semántico (default: 0.78)
  TOP_K          — Chunks recuperados por defecto (default: 5)
  LOG_LEVEL      — DEBUG | INFO | WARNING (default: INFO)
"""

import os
import logging
from pathlib import Path

# ── Cargar .env si existe (sin requerir python-dotenv como dependencia dura) ──
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def _require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(
            f"Variable de entorno requerida no definida: {name}\n"
            f"Agrégala a .env o al entorno antes de ejecutar."
        )
    return val


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


# ── Base de datos ─────────────────────────────────────────────────────────────
DATABASE_URL: str = _require("DATABASE_URL")

# ── LLM ──────────────────────────────────────────────────────────────────────
LLM_API_KEY: str = _get("GROQ_API_KEY", "") or _require("LLM_API_KEY")
LLM_PROVIDER: str = _get("LLM_PROVIDER", "groq")          # "groq" | "huggingface"
LLM_MODEL: str = _get("LLM_MODEL", "llama-3.3-70b-versatile")

# URLs de API según proveedor
_LLM_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "huggingface": f"https://api-inference.huggingface.co/models/{_get('LLM_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')}",
}
LLM_API_URL: str = _LLM_URLS.get(LLM_PROVIDER, _LLM_URLS["groq"])

# ── Modelos de embedding ───────────────────────────────────────────────────────
MINILM_MODEL: str = _get("MINILM_MODEL", "all-MiniLM-L6-v2")
CLIP_MODEL: str = _get("CLIP_MODEL", "openai/clip-vit-base-patch32")
EMBEDDING_DIM_TEXTO: int = 384
EMBEDDING_DIM_IMAGEN: int = 512

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(_get("CHUNK_SIZE", "256"))       # tokens, fixed-size
CHUNK_OVERLAP: int = int(_get("CHUNK_OVERLAP", "32"))  # tokens, fixed-size
SENTENCE_MAX: int = int(_get("SENTENCE_MAX", "5"))     # oraciones, sentence-aware
SEMANTIC_THR: float = float(_get("SEMANTIC_THR", "0.78"))  # umbral coseno semántico

# ── Recuperación ──────────────────────────────────────────────────────────────
TOP_K: int = int(_get("TOP_K", "5"))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
