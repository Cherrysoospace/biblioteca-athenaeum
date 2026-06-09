"""ui/pages/1_Busqueda.py — Búsqueda semántica principal (texto, imagen y multimodal)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import streamlit as st
from core.database import get_session
from core.config import TOP_K
from pipeline.embeddings_minilm import get_embedding
from pipeline.retrieval_texto import buscar_chunks_similares, buscar_hibrido_texto
from pipeline.retrieval_imagen import buscar_imagenes_por_texto, buscar_imagenes_similares
from pipeline.embeddings_clip import get_embedding_texto_clip, get_embedding_imagen
from ui.components.result_card import render_result_card
from ui.utils.session import set_query, set_results

st.set_page_config(page_title="Búsqueda Semántica", layout="wide")
st.title("🔍 Búsqueda Semántica")

# ── Modo de búsqueda ──────────────────────────────────────────────────────
modo = st.radio(
    "Modo de búsqueda",
    ["Texto → Texto", "Texto → Imagen", "Imagen → Imagen", "Híbrida"],
    horizontal=True,
    key="search_mode",
)

# ── Input según modo ──────────────────────────────────────────────────────
query = ""
imagen_subida = None

if modo == "Imagen → Imagen":
    imagen_subida = st.file_uploader(
        "Sube una imagen para buscar visualmente similares",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    query = st.text_input("Buscar", placeholder="Ej: obras de Gabriel García Márquez sobre realismo mágico...")

# ── Filtros colapsables ───────────────────────────────────────────────────
with st.expander("Filtros avanzados", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        tipos_disponibles = ["", "libro", "articulo", "revista", "imagen", "mapa", "tesis"]
        filtro_tipo = st.selectbox("Tipo de recurso", tipos_disponibles)
    with col2:
        idiomas = ["", "es", "en", "pt", "fr"]
        filtro_idioma = st.selectbox("Idioma", idiomas)
    with col3:
        filtro_anio = st.number_input("Año de publicación", min_value=0, max_value=2030, value=0, step=1)

    estrategias = ["", "fixed_size", "sentence_aware", "semantic"]
    filtro_estrategia = st.selectbox("Estrategia de chunking", estrategias)

# ── Ejecución de búsqueda ─────────────────────────────────────────────────
debemos_buscar = (modo == "Imagen → Imagen" and imagen_subida is not None) or (modo != "Imagen → Imagen" and query)

if debemos_buscar:
    if query:
        set_query(query)

    filtros: dict = {}
    if filtro_tipo:
        filtros["tipo"] = filtro_tipo
    if filtro_idioma:
        filtros["idioma"] = filtro_idioma
    if filtro_anio > 0:
        filtros["fecha_desde"] = f"{filtro_anio}-01-01"
        filtros["fecha_hasta"] = f"{filtro_anio}-12-31"

    estrategia = filtro_estrategia or None

    with st.spinner("Buscando..."):
        try:
            with get_session() as session:
                if modo == "Imagen → Imagen":
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(imagen_subida.read())
                        tmp_path = tmp.name
                    vector_img = get_embedding_imagen(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    resultados_raw = buscar_imagenes_similares(session, vector_img, top_k=TOP_K)

                elif modo == "Texto → Imagen":
                    resultados_raw = buscar_imagenes_por_texto(session, query, top_k=TOP_K)

                elif modo == "Híbrida":
                    vector = get_embedding(query)
                    resultados_raw = buscar_hibrido_texto(
                        session, vector, top_k=TOP_K,
                        filtros=filtros if any(filtros.values()) else None,
                        estrategia=estrategia,
                    )
                else:
                    vector = get_embedding(query)
                    if any(filtros.values()):
                        resultados_raw = buscar_hibrido_texto(
                            session, vector, top_k=TOP_K,
                            filtros=filtros, estrategia=estrategia,
                        )
                    else:
                        resultados_raw = buscar_chunks_similares(
                            session, vector, top_k=TOP_K, estrategia=estrategia,
                        )

            # Uniformizar resultados para render_result_card
            resultados = []
            for r in resultados_raw:
                card = dict(r)
                if "ruta_archivo" in card:
                    card["ruta_imagen"] = card["ruta_archivo"]
                if "descripcion_imagen" in card and "chunk_texto" not in card:
                    card["chunk_texto"] = card["descripcion_imagen"]
                if "tipo_imagen" in card and "tipo" not in card:
                    card["tipo"] = card["tipo_imagen"]
                resultados.append(card)

            set_results(resultados)
        except Exception as exc:
            st.error(f"Error en la búsqueda: {exc}")
            resultados = []

    # ── Mostrar resultados ────────────────────────────────────────────────
    if resultados:
        st.success(f"Se encontraron {len(resultados)} resultados")
        cols_per_row = 2
        for i in range(0, len(resultados), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(resultados):
                    with col:
                        render_result_card(resultados[idx])
    else:
        st.info("No se encontraron resultados. Intenta modificar la consulta o los filtros.")
