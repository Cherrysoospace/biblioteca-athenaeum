"""ui/pages/5_Administracion.py — Ingesta de recursos, estado de embeddings y configuración."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pandas as pd
import streamlit as st
from sqlalchemy import select, func as sa_func
from core.database import get_session
from core import config as core_config
from models.recursos import Recurso, ImagenRecurso
from models.embeddings_texto import EmbeddingTexto
from models.embeddings_imagen import EmbeddingImagen
from pipeline.ingest import ingest_recurso
from pipeline.chunking import chunk_texto

st.set_page_config(page_title="Administración", layout="wide")
st.title("⚙ Administración")

# ── Tabs ──────────────────────────────────────────────────────────────────
tab_ingest, tab_estado, tab_config = st.tabs([
    "📥 Ingestar recursos",
    "📊 Estado de embeddings",
    "🔧 Configuración",
])

# ────────────────────────────────────────────────────────────────────────
# TAB 1: Ingesta
# ────────────────────────────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Nuevo recurso")

    with st.form("nuevo_recurso_form"):
        col_img, col_fields = st.columns([1, 2])

        with col_img:
            uploaded_file = st.file_uploader(
                "Imagen del recurso",
                type=["png", "jpg", "jpeg"],
                key="admin_img_upload",
            )
            if uploaded_file is not None:
                st.image(uploaded_file, width=250)

        with col_fields:
            titulo = st.text_input("Título *", placeholder="Ej: Cien años de soledad")
            tipo = st.selectbox(
                "Tipo *",
                ["libro", "articulo", "revista", "video", "mapa", "fotografia", "otro"],
                index=0,
            )
            descripcion = st.text_area(
                "Descripción",
                placeholder="Breve descripción del contenido del recurso...",
            )
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                fecha_pub = st.date_input("Fecha de publicación", value=None)
            with col_f2:
                idioma = st.text_input("Idioma", placeholder="es")
            with col_f3:
                licencia = st.text_input("Licencia", placeholder="CC BY-SA 4.0")

            autor_nombre = st.text_input(
                "Autor (opcional)",
                placeholder="Nombre del autor (se creará si no existe)",
            )

            genero_nombre = st.text_input(
                "Género (opcional)",
                placeholder="Ej: Novela, Poesía, Ciencia Ficción (se creará si no existe)",
            )

        st.markdown("---")
        col_submit, _ = st.columns([1, 3])
        with col_submit:
            submitted = st.form_submit_button("📥 Crear recurso e ingestar", type="primary")

    if submitted:
        if not titulo.strip():
            st.error("El título es obligatorio.")
        elif uploaded_file is None:
            st.error("Debes subir una imagen para el recurso.")
        else:
            with st.status("Creando recurso e ingestando...", expanded=True) as status:
                try:
                    from models.recursos import Autor, RecursoAutor, Genero, RecursoGenero

                    media_dir = Path(__file__).resolve().parent.parent.parent / "data" / "media"
                    media_dir.mkdir(parents=True, exist_ok=True)

                    # 1. Guardar imagen
                    img_ext = Path(uploaded_file.name).suffix
                    img_filename = f"recurso_{titulo.lower().replace(' ', '_')[:50]}{img_ext}"
                    img_path = media_dir / img_filename
                    with open(img_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    ruta_relativa = f"data/media/{img_filename}"
                    status.write(f"✅ Imagen guardada en {ruta_relativa}")

                    with get_session() as session:
                        # 2. Crear recurso
                        recurso = Recurso(
                            titulo=titulo.strip(),
                            tipo=tipo,
                            descripcion=descripcion.strip() if descripcion else None,
                            fecha_publicacion=fecha_pub if fecha_pub else None,
                            idioma=idioma.strip() if idioma else None,
                            licencia=licencia.strip() if licencia else None,
                        )
                        session.add(recurso)
                        session.flush()
                        status.write(f"✅ Recurso creado (ID {recurso.id})")

                        # 3. Crear ImagenRecurso
                        img_record = ImagenRecurso(
                            recurso_id=recurso.id,
                            ruta_archivo=ruta_relativa,
                            tipo_imagen="portada",
                            descripcion=f"Imagen de {titulo.strip()}",
                        )
                        session.add(img_record)
                        session.flush()
                        status.write(f"✅ Imagen asociada (ID {img_record.id})")

                        # 4. Autor opcional
                        if autor_nombre and autor_nombre.strip():
                            autor = session.execute(
                                select(Autor).where(Autor.nombre == autor_nombre.strip())
                            ).scalar_one_or_none()
                            if autor is None:
                                autor = Autor(
                                    nombre=autor_nombre.strip(),
                                    tipo="persona",
                                )
                                session.add(autor)
                                session.flush()
                                status.write(f"✅ Autor creado (ID {autor.id})")
                            else:
                                status.write(f"✅ Autor existente (ID {autor.id})")

                            vinculo = RecursoAutor(recurso_id=recurso.id, autor_id=autor.id)
                            session.add(vinculo)
                            session.flush()

                        # 5. Género opcional
                        if genero_nombre and genero_nombre.strip():
                            genero = session.execute(
                                select(Genero).where(Genero.nombre == genero_nombre.strip())
                            ).scalar_one_or_none()
                            if genero is None:
                                genero = Genero(nombre=genero_nombre.strip())
                                session.add(genero)
                                session.flush()
                                status.write(f"✅ Género creado (ID {genero.id})")
                            else:
                                status.write(f"✅ Género existente (ID {genero.id})")

                            vinculo_g = RecursoGenero(recurso_id=recurso.id, genero_id=genero.id)
                            session.add(vinculo_g)
                            session.flush()

                        # 6. Estrategias de chunking
                        estrategias = ["fixed_size", "sentence_aware", "semantic"]
                        vectorizar_imagenes = True

                        # 7. Ingestar (vectorizar)
                        stats = ingest_recurso(
                            session=session,
                            recurso_id=recurso.id,
                            estrategias=estrategias,
                            vectorizar_imagenes=vectorizar_imagenes,
                        )
                        status.write(f"✅ Chunks generados: {stats['chunks_total']}")
                        status.write(f"✅ Embeddings texto: {stats['embeddings_texto']}")
                        status.write(f"✅ Embeddings imagen: {stats['embeddings_imagen']}")
                        if stats.get("errores"):
                            for err in stats["errores"]:
                                status.write(f"⚠ {err}")

                        status.update(label="Recurso creado e ingestado", state="complete")
                        st.success(f"Recurso «{titulo.strip()}» (ID {recurso.id}) creado y vectorizado exitosamente.")
                        st.balloons()

                except Exception as exc:
                    status.update(label="Error", state="error")
                    st.error(f"Error: {exc}")
                    import traceback
                    st.code(traceback.format_exc())

    # ── Ingestar un recurso existente por ID ──────────────────────────────
    st.markdown("---")
    st.markdown("#### Re-ingestar recurso existente por ID")

    estrategias_seleccionadas = st.multiselect(
        "Estrategias de chunking",
        ["fixed_size", "sentence_aware", "semantic"],
        default=["fixed_size", "sentence_aware", "semantic"],
        key="admin_estrategias",
    )

    vectorizar_imagenes = st.checkbox("Vectorizar imágenes asociadas", value=True, key="admin_vec_img")

    col_id, col_btn = st.columns([3, 1])
    with col_id:
        recurso_id = st.number_input("ID del recurso", min_value=1, value=1, step=1, key="admin_recurso_id")
    with col_btn:
        ingest_click = st.button("▶ Re-ingestar", type="primary")

    if ingest_click:
        with st.status(f"Re-ingestando recurso ID {recurso_id}...", expanded=True) as status:
            try:
                with get_session() as session:
                    stats = ingest_recurso(
                        session=session,
                        recurso_id=recurso_id,
                        estrategias=estrategias_seleccionadas,
                        vectorizar_imagenes=vectorizar_imagenes,
                    )
                status.write(f"✅ Chunks generados: {stats['chunks_total']}")
                status.write(f"✅ Embeddings texto: {stats['embeddings_texto']}")
                status.write(f"✅ Embeddings imagen: {stats['embeddings_imagen']}")
                if stats.get("errores"):
                    for err in stats["errores"]:
                        status.write(f"⚠ {err}")
                status.update(label="Ingesta completada", state="complete")
                st.success(f"Recurso {recurso_id} re-ingestado exitosamente.")
            except Exception as exc:
                status.update(label="Error en ingesta", state="error")
                st.error(f"Error: {exc}")

    # ── Chunking de prueba ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Probar chunking")
    texto_prueba = st.text_area("Texto de prueba", height=100, placeholder="Pega aquí un texto para probar las estrategias de chunking...")
    if texto_prueba and st.button("Probar chunking", key="test_chunk"):
        for est in estrategias_seleccionadas:
            with st.expander(f"Estrategia: {est}"):
                chunks = chunk_texto(texto_prueba, estrategia=est)
                st.write(f"**{len(chunks)} chunks generados**")
                for i, ch in enumerate(chunks, 1):
                    st.text(f"[{i}] {ch[:200]}{'…' if len(ch) > 200 else ''}")

# ────────────────────────────────────────────────────────────────────────
# TAB 2: Estado de embeddings
# ────────────────────────────────────────────────────────────────────────
with tab_estado:
    st.subheader("Estado de embeddings por recurso")

    @st.cache_data(ttl=60)
    def estado_embeddings() -> pd.DataFrame:
        with get_session() as session:
            rows = session.execute(
                select(
                    Recurso.id,
                    Recurso.titulo,
                    Recurso.tipo,
                    sa_func.count(EmbeddingTexto.id).label("chunks_texto"),
                    EmbeddingTexto.estrategia_chunking,
                    sa_func.count(EmbeddingImagen.id).label("emb_imagenes"),
                )
                .outerjoin(EmbeddingTexto, EmbeddingTexto.recurso_id == Recurso.id)
                .outerjoin(ImagenRecurso, ImagenRecurso.recurso_id == Recurso.id)
                .outerjoin(EmbeddingImagen, EmbeddingImagen.imagen_id == ImagenRecurso.id)
                .group_by(Recurso.id, Recurso.titulo, Recurso.tipo, EmbeddingTexto.estrategia_chunking)
                .order_by(Recurso.id)
                .limit(200)
            ).mappings().all()

        data = [
            {
                "ID": r["id"],
                "Título": r["titulo"],
                "Tipo": r["tipo"],
                "Chunks texto": r["chunks_texto"],
                "Estrategia": r["estrategia_chunking"] or "—",
                "Emb. imágenes": r["emb_imagenes"],
            }
            for r in rows
        ]
        return pd.DataFrame(data)

    df_estado = estado_embeddings()
    if not df_estado.empty:
        st.dataframe(df_estado, use_container_width=True, hide_index=True)
    else:
        st.info("No hay recursos con embeddings generados todavía.")

    # ── Resumen general ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Resumen")
    with get_session() as session:
        total_rec = session.scalar(select(sa_func.count(Recurso.id)))
        total_emb_texto = session.scalar(select(sa_func.count(EmbeddingTexto.id)))
        total_emb_img = session.scalar(select(sa_func.count(EmbeddingImagen.id)))

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Total recursos", total_rec)
    col_r2.metric("Embeddings texto", total_emb_texto)
    col_r3.metric("Embeddings imagen", total_emb_img)

# ────────────────────────────────────────────────────────────────────────
# TAB 3: Configuración
# ────────────────────────────────────────────────────────────────────────
with tab_config:
    st.subheader("Configuración del sistema")

    config_items = {
        "Proveedor LLM": core_config.LLM_PROVIDER,
        "Modelo LLM": core_config.LLM_MODEL,
        "Modelo MiniLM": core_config.MINILM_MODEL,
        "Modelo CLIP": core_config.CLIP_MODEL,
        "Dimensión embeddings texto": str(core_config.EMBEDDING_DIM_TEXTO),
        "Dimensión embeddings imagen": str(core_config.EMBEDDING_DIM_IMAGEN),
        "Chunk size (tokens)": str(core_config.CHUNK_SIZE),
        "Chunk overlap (tokens)": str(core_config.CHUNK_OVERLAP),
        "Max oraciones (sentence-aware)": str(core_config.SENTENCE_MAX),
        "Umbral semántico": str(core_config.SEMANTIC_THR),
        "Top K por defecto": str(core_config.TOP_K),
        "Nivel de log": core_config.LOG_LEVEL,
    }

    config_df = pd.DataFrame(
        [{"Parámetro": k, "Valor": v} for k, v in config_items.items()]
    )
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    # Variables de entorno sanitizadas
    st.markdown("---")
    st.subheader("Variables de entorno")
    env_keys = ["DATABASE_URL", "LLM_API_KEY", "MINILM_MODEL", "CLIP_MODEL", "CHUNK_SIZE"]
    env_data = []
    for key in env_keys:
        raw = os.getenv(key, "")
        val = raw if key in ("MINILM_MODEL", "CLIP_MODEL", "CHUNK_SIZE") else (raw[:12] + "…" + raw[-4:] if len(raw) > 20 else "(no configurada)")
        env_data.append({"Variable": key, "Valor": val})

    st.dataframe(pd.DataFrame(env_data), use_container_width=True, hide_index=True)
    st.caption("Los valores sensibles (credenciales) se muestran truncados.")

    st.markdown("---")
    st.markdown(f"**Archivo .env:** {Path('.env').resolve()}")
    try:
        env_content = Path(".env").read_text()
        sanitized = "\n".join(
            line.split("=")[0] + "=***" if "KEY" in line or "PASSWORD" in line or "URL" in line else line
            for line in env_content.splitlines()
            if line.strip() and not line.startswith("#")
        )
        st.code(sanitized, language="bash")
    except Exception:
        st.info("No se pudo leer el archivo .env")
