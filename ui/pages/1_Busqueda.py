"""ui/pages/1_Busqueda.py — Búsqueda semántica principal (texto y multimodal)."""

from __future__ import annotations

import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import streamlit as st
from sqlalchemy import select
from core.database import get_session
from core.config import TOP_K
from pipeline.embeddings_minilm import get_embedding
from pipeline.retrieval_texto import buscar_chunks_similares, buscar_hibrido_texto
from pipeline.retrieval_imagen import buscar_imagenes_por_texto
from pipeline.embeddings_clip import get_embedding_texto_clip
from models.recursos import Recurso
from ui.components.result_card import render_result_card
from ui.utils.session import set_query, set_results

st.set_page_config(page_title="Búsqueda Semántica", layout="wide")
st.title("🔍 Búsqueda Semántica")

# ── Modo de búsqueda ──────────────────────────────────────────────────────
modo = st.radio(
    "Modo de búsqueda",
    ["Texto → Texto", "Texto → Imagen", "Híbrida"],
    horizontal=True,
    key="search_mode",
)

# ── Barra de búsqueda ─────────────────────────────────────────────────────
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
                if modo == "Texto → Imagen":
                    vector = get_embedding_texto_clip(query)
                    resultados = buscar_imagenes_por_texto(session, query, top_k=TOP_K)
                elif modo == "Híbrida":
                    vector = get_embedding(query)
                    resultados = buscar_hibrido_texto(
                        session, vector, top_k=TOP_K,
                        filtros=filtros if any(filtros.values()) else None,
                        estrategia=estrategia,
                    )
                else:
                    vector = get_embedding(query)
                    if any(filtros.values()):
                        resultados = buscar_hibrido_texto(
                            session, vector, top_k=TOP_K,
                            filtros=filtros, estrategia=estrategia,
                        )
                    else:
                        resultados = buscar_chunks_similares(
                            session, vector, top_k=TOP_K, estrategia=estrategia,
                        )
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
