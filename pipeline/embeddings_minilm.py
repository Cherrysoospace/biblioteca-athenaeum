"""
pipeline/embeddings_minilm.py — Generación de embeddings de texto con all-MiniLM-L6-v2.

El modelo se carga una sola vez (singleton) al primer uso.
Produce vectores de 384 dimensiones normalizados en L2.

Funciones públicas:
    get_embedding(texto)           -> list[float]  (un solo texto)
    get_embeddings_batch(textos)   -> list[list[float]]  (lote)
"""

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from core.config import MINILM_MODEL, EMBEDDING_DIM_TEXTO

logger = logging.getLogger(__name__)

# ── Singleton del modelo ──────────────────────────────────────────────────────

_model = None


def _load_model():
    """Carga el modelo sentence-transformers (lazy, una sola vez)."""
    global _model
    if _model is not None:
        return _model
    try:
        logger.info("Cargando modelo MiniLM: %s (puede tomar ~30s la primera vez)...", MINILM_MODEL)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MINILM_MODEL)
        logger.info("Modelo MiniLM listo (dim=%d).", EMBEDDING_DIM_TEXTO)
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers no está instalado. "
            "Ejecuta: pip install sentence-transformers"
        ) from exc
    return _model


# ── API pública ───────────────────────────────────────────────────────────────

def get_embedding(texto: str) -> list[float]:
    """
    Genera el embedding normalizado de un único texto.

    Args:
        texto: cadena de texto a vectorizar.

    Returns:
        Lista de floats de longitud EMBEDDING_DIM_TEXTO (384).
    """
    if not texto or not texto.strip():
        # Vector cero como fallback para texto vacío
        return [0.0] * EMBEDDING_DIM_TEXTO

    model = _load_model()
    vector = model.encode(texto, normalize_embeddings=True, show_progress_bar=False)
    return vector.tolist()


def get_embeddings_batch(
    textos: list[str],
    batch_size: int = 64,
    show_progress: bool = False,
) -> list[list[float]]:
    """
    Genera embeddings para una lista de textos en lotes.

    Args:
        textos: lista de strings a vectorizar.
        batch_size: tamaño del lote de inferencia.
        show_progress: mostrar barra de progreso de sentence-transformers.

    Returns:
        Lista de listas de floats (misma longitud que `textos`).
    """
    if not textos:
        return []

    # Textos vacíos → vector cero; textos válidos → encoder
    indices_validos = [i for i, t in enumerate(textos) if t and t.strip()]
    indices_vacios = [i for i in range(len(textos)) if i not in indices_validos]

    resultado: list[Optional[list[float]]] = [None] * len(textos)

    # Rellenar vacíos con vector cero
    for i in indices_vacios:
        resultado[i] = [0.0] * EMBEDDING_DIM_TEXTO

    if indices_validos:
        textos_validos = [textos[i] for i in indices_validos]
        model = _load_model()
        vectores = model.encode(
            textos_validos,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        for idx, vec in zip(indices_validos, vectores):
            resultado[idx] = vec.tolist()

    return resultado  # type: ignore[return-value]


def similitud_coseno(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calcula la similitud coseno entre dos vectores ya normalizados.
    Si ambos están en L2, es equivalente al producto punto.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
