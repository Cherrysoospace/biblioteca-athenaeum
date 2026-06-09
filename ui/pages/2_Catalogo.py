"""ui/pages/2_Catalogo.py — Exploración relacional del catálogo de recursos."""

from __future__ import annotations

import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pandas as pd
import streamlit as st
from sqlalchemy import select, func as sa_func
from core.database import get_session
from models.recursos import Recurso, Autor, RecursoAutor, ImagenRecurso
from models.embeddings_texto import EmbeddingTexto

st.set_page_config(page_title="Catálogo", layout="wide")
st.title("📚 Catálogo Relacional")

# ── Filtros ───────────────────────────────────────────────────────────────
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        tipos = ["", "libro", "articulo", "revista", "imagen", "mapa", "tesis"]
        f_tipo = st.selectbox("Tipo", tipos, key="cat_tipo")
    with col2:
        idiomas = ["", "Español", "English", "Français", "Português", "Deutsch", "Chinese"]
        f_idioma = st.selectbox("Idioma", idiomas, key="cat_idioma")
    with col3:
        f_anio_desde = st.number_input("Año desde", min_value=0, max_value=2030, value=0, step=1, key="cat_anio_desde")
    with col4:
        f_anio_hasta = st.number_input("Año hasta", min_value=0, max_value=2030, value=0, step=1, key="cat_anio_hasta")

# ── Estadísticas rápidas ─────────────────────────────────────────────────
with get_session() as session:
    total_recursos = session.scalar(select(sa_func.count(Recurso.id)))
    total_chunks = session.scalar(select(sa_func.count(EmbeddingTexto.id)))
    total_emb_estrategias = session.execute(
        select(EmbeddingTexto.estrategia_chunking, sa_func.count(EmbeddingTexto.id))
        .group_by(EmbeddingTexto.estrategia_chunking)
    ).all()

col_s1, col_s2, col_s3 = st.columns(3)
col_s1.metric("Total recursos", total_recursos)
col_s2.metric("Total chunks", total_chunks)
col_s3.metric("Estrategias de chunking", len(total_emb_estrategias))

if total_emb_estrategias:
    st.caption("Chunks por estrategía: " + ", ".join(f"{e}: {c}" for e, c in total_emb_estrategias))

# ── Consulta paginada ────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_recursos(
    tipo: str, idioma: str, anio_desde: int, anio_hasta: int, limit: int, offset: int
) -> tuple[list[dict], int]:
    with get_session() as session:
        q = select(Recurso)
        count_q = select(sa_func.count(Recurso.id))

        if tipo:
            q = q.where(Recurso.tipo == tipo)
            count_q = count_q.where(Recurso.tipo == tipo)
        if idioma:
            q = q.where(Recurso.idioma == idioma)
            count_q = count_q.where(Recurso.idioma == idioma)
        if anio_desde > 0:
            from datetime import date
            q = q.where(Recurso.fecha_publicacion >= date(anio_desde, 1, 1))
            count_q = count_q.where(Recurso.fecha_publicacion >= date(anio_desde, 1, 1))
        if anio_hasta > 0:
            from datetime import date
            q = q.where(Recurso.fecha_publicacion <= date(anio_hasta, 12, 31))
            count_q = count_q.where(Recurso.fecha_publicacion <= date(anio_hasta, 12, 31))

        total = session.scalar(count_q) or 0
        q = q.order_by(Recurso.id).offset(offset).limit(limit)
        rows = session.scalars(q).all()

        data = []
        for r in rows:
            autores = session.execute(
                select(Autor.nombre)
                .join(RecursoAutor, RecursoAutor.autor_id == Autor.id)
                .where(RecursoAutor.recurso_id == r.id)
            ).scalars().all()

            data.append({
                "ID": r.id,
                "Título": r.titulo,
                "Tipo": r.tipo,
                "Idioma": r.idioma or "",
                "Año": r.fecha_publicacion.year if r.fecha_publicacion else "",
                "Autores": ", ".join(autores) if autores else "",
            })
        return data, total


page_size = 20
query_params = st.query_params
page = int(query_params.get("page", 1))
offset = (page - 1) * page_size

data, total = fetch_recursos(f_tipo, f_idioma, f_anio_desde, f_anio_hasta, page_size, offset)
total_pages = max(1, (total + page_size - 1) // page_size)

if data:
    df = pd.DataFrame(data)
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Paginación manual
    col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
    with col_p1:
        if page > 1:
            if st.button("← Anterior"):
                st.query_params.page = str(page - 1)
                st.rerun()
    with col_p2:
        st.markdown(f"<div style='text-align:center'>Página {page} de {total_pages} ({total} recursos)</div>",
                    unsafe_allow_html=True)
    with col_p3:
        if page < total_pages:
            if st.button("Siguiente →"):
                st.query_params.page = str(page + 1)
                st.rerun()

    # ── Detalle de recurso seleccionado ───────────────────────────────────
    if event and event.selection and event.selection.rows:
        idx = event.selection.rows[0]
        recurso_id = data[idx]["ID"]
        with get_session() as session:
            recurso = session.get(Recurso, recurso_id)
            if recurso:
                st.markdown("---")
                st.subheader(f"📖 Detalle: {recurso.titulo}")
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown(f"**ID:** {recurso.id}")
                    st.markdown(f"**Tipo:** {recurso.tipo}")
                    st.markdown(f"**Idioma:** {recurso.idioma or '—'}")
                    st.markdown(f"**Publicación:** {recurso.fecha_publicacion or '—'}")
                    st.markdown(f"**Licencia:** {recurso.licencia or '—'}")
                with col_d2:
                    st.markdown("**Descripción:**")
                    st.markdown(recurso.descripcion or "Sin descripción")

                # Autores
                autores = session.execute(
                    select(Autor).join(RecursoAutor).where(RecursoAutor.recurso_id == recurso.id)
                ).scalars().all()
                if autores:
                    st.markdown("**Autores:**")
                    for a in autores:
                        st.markdown(f"- {a.nombre} ({a.tipo})")

                # Chunks
                chunks = session.execute(
                    select(EmbeddingTexto).where(EmbeddingTexto.recurso_id == recurso.id)
                ).scalars().all()
                if chunks:
                    st.markdown(f"**Chunks ({len(chunks)})**")
                    for ch in chunks[:10]:
                        with st.expander(f"Chunk #{ch.chunk_id} — {ch.estrategia_chunking}"):
                            st.text(ch.chunk_texto)

                # Imágenes
                imagenes = session.execute(
                    select(ImagenRecurso).where(ImagenRecurso.recurso_id == recurso.id)
                ).scalars().all()
                if imagenes:
                    st.markdown(f"**Imágenes ({len(imagenes)})**")
                    for img in imagenes:
                        st.markdown(f"- {img.ruta_archivo} ({img.tipo_imagen or 'sin tipo'})")
else:
    st.info("No se encontraron recursos con los filtros seleccionados.")
