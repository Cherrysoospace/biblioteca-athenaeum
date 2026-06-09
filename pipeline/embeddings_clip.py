"""
pipeline/embeddings_clip.py — Generación de embeddings visuales con CLIP ViT-B/32.

Produce vectores de 512 dimensiones normalizados en L2.

Soporta:
  - Imágenes locales (ruta de archivo absoluta o relativa)
  - Imágenes remotas (URL http/https)
  - Texto (para consultas imagen→texto o texto→imagen)

Funciones públicas:
    get_embedding_imagen(ruta_o_url)  -> list[float]
    get_embedding_texto_clip(texto)   -> list[float]
    similitud_clip(vec_a, vec_b)      -> float
"""

import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np

from core.config import CLIP_MODEL, EMBEDDING_DIM_IMAGEN

logger = logging.getLogger(__name__)

_clip_model = None
_clip_processor = None


def _load_clip():
    """Carga el modelo CLIP y su procesador (lazy, singleton)."""
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    try:
        logger.info("Cargando modelo CLIP: %s (puede tomar ~2min la primera vez)...", CLIP_MODEL)
        from transformers import CLIPModel, CLIPProcessor
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
        _clip_model.eval()
        logger.info("Modelo CLIP listo (dim=%d).", EMBEDDING_DIM_IMAGEN)
    except ImportError as exc:
        raise ImportError(
            "transformers y/o torch no están instalados. "
            "Ejecuta: pip install transformers torch Pillow"
        ) from exc

    return _clip_model, _clip_processor


def _cargar_imagen(ruta_o_url: str):
    """
    Carga una imagen PIL desde:
      - ruta de archivo local (absoluta o relativa)
      - URL http/https
    """
    from PIL import Image

    ruta = str(ruta_o_url).strip()

    if ruta.startswith("http://") or ruta.startswith("https://"):
        import urllib.request
        with urllib.request.urlopen(ruta, timeout=10) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")

    path = Path(ruta)
    if not path.exists():
        raise FileNotFoundError(f"Imagen no encontrada en disco: {ruta}")
    return Image.open(path).convert("RGB")


def _normalizar(vector: np.ndarray) -> list[float]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.tolist()
    return (vector / norm).tolist()


# ── API pública ───────────────────────────────────────────────────────────────

def get_embedding_imagen(ruta_o_url: str) -> list[float]:
    """
    Genera el embedding visual (512 dims) de una imagen.

    Args:
        ruta_o_url: ruta local o URL de la imagen.

    Returns:
        Lista de floats normalizada en L2.

    Raises:
        FileNotFoundError: si la ruta no existe en disco.
        ImportError: si transformers/torch no están instalados.
    """
    import torch

    model, processor = _load_clip()
    imagen = _cargar_imagen(ruta_o_url)

    inputs = processor(images=imagen, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = features[0].cpu().numpy()
    return _normalizar(vector)


def get_embedding_texto_clip(texto: str) -> list[float]:
    """
    Genera el embedding textual de CLIP (512 dims) para consultas cruzadas
    texto→imagen. Útil para recuperar imágenes a partir de una descripción.

    Args:
        texto: descripción textual de la imagen buscada.

    Returns:
        Lista de floats normalizada en L2.
    """
    import torch

    if not texto or not texto.strip():
        return [0.0] * EMBEDDING_DIM_IMAGEN

    model, processor = _load_clip()
    inputs = processor(text=[texto], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)

    vector = features[0].cpu().numpy()
    return _normalizar(vector)


def similitud_clip(vec_a: list[float], vec_b: list[float]) -> float:
    """Similitud coseno entre dos embeddings CLIP normalizados."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embeddings_imagenes_batch(
    rutas: list[str],
) -> list[list[float]]:
    """
    Genera embeddings para una lista de imágenes.
    Las imágenes que fallen se representan con vector cero.
    """
    resultados = []
    for ruta in rutas:
        try:
            resultados.append(get_embedding_imagen(ruta))
        except Exception as exc:
            logger.error("Error vectorizando imagen '%s': %s", ruta, exc)
            resultados.append([0.0] * EMBEDDING_DIM_IMAGEN)
    return resultados
