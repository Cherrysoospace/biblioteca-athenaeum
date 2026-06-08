"""ui/components/result_card.py — Tarjeta reutilizable para mostrar resultados de búsqueda."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_result_card(result: dict[str, Any]) -> None:
    """Renderiza una tarjeta de resultado con metadatos, score y fragmento.

    Args:
        result: Diccionario con claves:
            titulo, autor, tipo, score, chunk_texto, ruta_imagen, anio.
    """
    titulo = result.get("titulo") or result.get("titulo_recurso", "Sin título")
    autor = result.get("autor", "Autor desconocido")
    tipo = result.get("tipo", "recurso")
    score = result.get("score", 0.0)
    chunk_texto = result.get("chunk_texto") or result.get("descripcion_imagen", "")
    ruta_imagen = result.get("ruta_imagen")
    anio = result.get("anio")

    with st.container(border=True):
        cols = st.columns([1, 3])

        # Columna izquierda: imagen o inicial
        with cols[0]:
            if ruta_imagen:
                try:
                    st.image(ruta_imagen, use_container_width=True)
                except Exception:
                    st.markdown(f"<div style='text-align:center;font-size:2.5rem;padding:0.5rem'>📄</div>",
                                unsafe_allow_html=True)
            else:
                icon = {"libro": "📖", "articulo": "📄", "imagen": "🖼", "revista": "📰", "mapa": "🗺"}.get(tipo, "📄")
                st.markdown(f"<div style='text-align:center;font-size:2.5rem;padding:0.5rem'>{icon}</div>",
                            unsafe_allow_html=True)

        # Columna derecha: metadatos + fragmento
        with cols[1]:
            st.markdown(f"**{titulo}**")
            st.caption(f"{autor} · {tipo}" + (f" · {anio}" if anio else ""))

            # Barra de similitud
            score_pct = min(max(score, 0.0), 1.0)
            st.progress(score_pct, text=f"Similitud: {score_pct:.1%}")

            if chunk_texto:
                max_len = 250
                texto = chunk_texto if len(chunk_texto) <= max_len else chunk_texto[:max_len] + "…"
                st.text(texto)
