"""
pipeline/chunking.py — Tres estrategias de chunking de texto.

Estrategias implementadas:
  fixed_size      — fragmentos de N tokens con overlap fijo
  sentence_aware  — máx. K oraciones por chunk, overlap de 1 oración (spaCy)
  semantic        — corte por caída de similitud coseno entre oraciones

La función pública es:
    chunk_texto(texto, estrategia, **kwargs) -> list[str]

Parámetros por defecto se leen de core.config.
"""

import logging
import re
from typing import Optional

import numpy as np

from core.config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    SENTENCE_MAX, SEMANTIC_THR,
)

logger = logging.getLogger(__name__)

# ── Tokenización simple (word-based, sin requerir tiktoken) ───────────────────

def _tokenizar(texto: str) -> list[str]:
    """Tokenización por whitespace (aproximación rápida a tokens BPE)."""
    return texto.split()


def _detokenizar(tokens: list[str]) -> str:
    return " ".join(tokens)


# ── Estrategia A: Fixed-size ──────────────────────────────────────────────────

def _fixed_size(
    texto: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Divide en fragmentos de `chunk_size` tokens con `overlap` tokens de solapamiento.
    Separa únicamente por conteo de tokens; puede cortar en medio de una oración.
    """
    tokens = _tokenizar(texto)
    if not tokens:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = _detokenizar(tokens[start:end])
        chunks.append(chunk)
        start += step

    return chunks


# ── Estrategia B: Sentence-aware ─────────────────────────────────────────────

def _split_oraciones(texto: str) -> list[str]:
    """
    Divide texto en oraciones usando spaCy (es_core_news_md).
    Fallback: split por '.', '?' y '!' si spaCy no está disponible.
    """
    try:
        import spacy
        # Intentar cargar modelo español; si no está, usar el pequeño
        try:
            nlp = spacy.load("es_core_news_md", disable=["ner", "tagger", "lemmatizer"])
        except OSError:
            try:
                nlp = spacy.load("es_core_news_sm", disable=["ner", "tagger", "lemmatizer"])
            except OSError:
                nlp = spacy.blank("es")
                nlp.add_pipe("sentencizer")
        doc = nlp(texto)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except ImportError:
        logger.warning("spaCy no disponible; usando split por puntuación como fallback.")
        oraciones = re.split(r'(?<=[.!?])\s+', texto.strip())
        return [o for o in oraciones if o]


def _sentence_aware(
    texto: str,
    max_oraciones: int = SENTENCE_MAX,
    max_tokens: int = 300,
) -> list[str]:
    """
    Agrupa hasta `max_oraciones` oraciones por chunk (o `max_tokens` tokens).
    Overlap: 1 oración entre chunks consecutivos.
    """
    oraciones = _split_oraciones(texto)
    if not oraciones:
        return []

    chunks = []
    i = 0
    while i < len(oraciones):
        grupo = []
        tokens_acum = 0
        j = i
        while j < len(oraciones) and len(grupo) < max_oraciones:
            t = len(_tokenizar(oraciones[j]))
            if grupo and tokens_acum + t > max_tokens:
                break
            grupo.append(oraciones[j])
            tokens_acum += t
            j += 1

        if grupo:
            chunks.append(" ".join(grupo))

        # Overlap: retroceder 1 oración
        i = max(i + 1, j - 1)

    return chunks


# ── Estrategia C: Semantic ────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _semantic(
    texto: str,
    umbral: float = SEMANTIC_THR,
    min_tokens: int = 50,
    max_tokens: int = 400,
) -> list[str]:
    """
    Corta el texto cuando la similitud coseno entre embeddings de oraciones
    consecutivas cae por debajo de `umbral`, indicando cambio temático.

    Requiere sentence-transformers (all-MiniLM-L6-v2).
    Fallback a fixed_size si el modelo no está disponible.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from core.config import MINILM_MODEL
        model = SentenceTransformer(MINILM_MODEL)
    except ImportError:
        logger.warning("sentence-transformers no disponible; fallback a fixed_size para semantic.")
        return _fixed_size(texto)

    oraciones = _split_oraciones(texto)
    if len(oraciones) <= 1:
        return oraciones if oraciones else []

    vectores = model.encode(oraciones, show_progress_bar=False)

    chunks = []
    grupo: list[str] = [oraciones[0]]
    tokens_grupo = len(_tokenizar(oraciones[0]))

    for idx in range(1, len(oraciones)):
        sim = _cosine_similarity(vectores[idx - 1], vectores[idx])
        tokens_nueva = len(_tokenizar(oraciones[idx]))

        # Cortar si: similitud baja O chunk demasiado grande
        if (sim < umbral and tokens_grupo >= min_tokens) or (tokens_grupo + tokens_nueva > max_tokens):
            chunks.append(" ".join(grupo))
            grupo = [oraciones[idx]]
            tokens_grupo = tokens_nueva
        else:
            grupo.append(oraciones[idx])
            tokens_grupo += tokens_nueva

    if grupo:
        chunks.append(" ".join(grupo))

    return chunks


# ── API pública ───────────────────────────────────────────────────────────────

_ESTRATEGIAS = {
    "fixed_size": _fixed_size,
    "sentence_aware": _sentence_aware,
    "semantic": _semantic,
}


def chunk_texto(
    texto: str,
    estrategia: str = "sentence_aware",
    **kwargs,
) -> list[str]:
    """
    Divide `texto` en chunks según `estrategia`.

    Args:
        texto: texto plano a fragmentar.
        estrategia: "fixed_size" | "sentence_aware" | "semantic"
        **kwargs: parámetros adicionales para la estrategia específica.

    Returns:
        Lista de strings (chunks), sin vacíos.
    """
    if not texto or not texto.strip():
        return []

    fn = _ESTRATEGIAS.get(estrategia)
    if fn is None:
        raise ValueError(
            f"Estrategia desconocida: {estrategia!r}. "
            f"Opciones válidas: {list(_ESTRATEGIAS)}"
        )

    chunks = fn(texto, **kwargs)
    # Filtrar chunks vacíos o muy cortos (< 5 tokens)
    return [c for c in chunks if len(_tokenizar(c)) >= 5]
