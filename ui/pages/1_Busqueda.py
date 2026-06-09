"""ui/pages/1_Busqueda.py — Búsqueda semántica y Chat RAG fusionados."""

from __future__ import annotations

import re
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
from pipeline.retrieval_texto import buscar_chunks_similares, buscar_hibrido_texto, ResultadoBusqueda
from pipeline.retrieval_imagen import buscar_imagenes_por_texto, buscar_imagenes_similares, buscar_imagenes_hibrido
from pipeline.embeddings_clip import get_embedding_texto_clip, get_embedding_imagen
from pipeline.rag import run_rag
from pipeline.evaluacion import calcular_metricas_ragas
from ui.components.result_card import render_result_card
from ui.utils.session import set_query, set_results, get_usuario_id, add_message, clear_messages, set_rag_context


_RE_ANIO = r'\b(1[89]\d{2}|20[0-2]\d)\b'


def _auto_detectar_filtros(query: str, filtros_explicitos: dict) -> dict:
    """Detecta año, modo de fecha y tipo desde el texto de la consulta.

    Los filtros explícitos tienen prioridad sobre los detectados automáticamente.
    Soporta:
      - "en 2000", "de 2000"          -> rango cerrado [2000-01-01, 2000-12-31]
      - "antes de 2000", "previo a"    -> solo fecha_hasta <= 2000-12-31
      - "después de 2000", "posterior" -> solo fecha_desde >= 2000-01-01
      - "desde 2000"                   -> solo fecha_desde >= 2000-01-01
      - "hasta 2000"                   -> solo fecha_hasta <= 2000-12-31
    """
    result = dict(filtros_explicitos)

    if "fecha_desde" not in result and "fecha_hasta" not in result:
        _detectar_fecha_desde_query(query, result)

    if "tipo" not in result:
        _detectar_tipo_desde_query(query, result)

    return result


def _detectar_fecha_desde_query(query: str, result: dict) -> None:
    """Busca patrones de fecha direccional en la consulta y completa result."""
    year_match = re.search(_RE_ANIO, query)
    if not year_match:
        return

    anio = year_match.group(1)
    pre = query[: year_match.start()].lower()
    post = query[year_match.end():].lower()

    if re.search(r'(antes|pret[eé]rito|previo|anterior)\s*(de|a|al)?\s*$', pre) or \
       re.search(r'previo|anterior', pre):
        result["fecha_hasta"] = f"{anio}-12-31"
        return

    if re.search(r'(despu[eé]s|posterior|subsiguiente|luego)\s*(de|a|al)?\s*$', pre):
        result["fecha_desde"] = f"{anio}-01-01"
        return

    if re.search(r'(desde|a\s*partir\s*de)\s*$', pre):
        result["fecha_desde"] = f"{anio}-01-01"
        return

    if re.search(r'(hasta|como\s*m[aá]ximo)\s*$', pre):
        result["fecha_hasta"] = f"{anio}-12-31"
        return

    if re.search(r'^\s*(en\s*adelante|en\s*luego|para\s*adelante|hacia\s*adelante)', post):
        result["fecha_desde"] = f"{anio}-01-01"
        return

    if re.search(r'^\s*(hacia\s*atr[aá]s|para\s*atr[aá]s)', post):
        result["fecha_hasta"] = f"{anio}-12-31"
        return

    result["fecha_desde"] = f"{anio}-01-01"
    result["fecha_hasta"] = f"{anio}-12-31"


def _detectar_tipo_desde_query(query: str, result: dict) -> None:
    """Busca palabras clave de tipo de recurso en la consulta."""
    if re.search(r'\blibros?\b', query, re.IGNORECASE):
        result["tipo"] = "libro"
    elif re.search(r'\brevistas?\b', query, re.IGNORECASE):
        result["tipo"] = "revista"
    elif re.search(r'\bartículos?\b', query, re.IGNORECASE):
        result["tipo"] = "articulo"
    elif re.search(r'\bmapas?\b', query, re.IGNORECASE):
        result["tipo"] = "mapa"
    elif re.search(r'\bfotos?\b|fotografías?\b', query, re.IGNORECASE):
        result["tipo"] = "fotografia"


st.set_page_config(page_title="Búsqueda y Chat RAG", layout="wide")
st.title("🔍 Búsqueda y Chat RAG")

# ── Modo de búsqueda ──────────────────────────────────────────────────────
modo = st.radio(
    "Modo de búsqueda",
    ["Texto -> Texto", "Texto -> Imagen", "Imagen -> Imagen", "Híbrida"],
    horizontal=True,
    key="search_mode",
)

# ── Input según modo ──────────────────────────────────────────────────────
query = ""
imagen_subida = None

if modo == "Imagen -> Imagen":
    imagen_subida = st.file_uploader(
        "Sube una imagen para buscar visualmente similares",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    query = st.text_input(
        "Buscar",
        placeholder="Ej: obras de Gabriel García Márquez sobre realismo mágico...",
    )

# ── Filtros colapsables ───────────────────────────────────────────────────
with st.expander("Filtros avanzados", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        tipos_disponibles = ["", "libro", "articulo", "revista", "imagen", "mapa", "tesis"]
        filtro_tipo = st.selectbox("Tipo de recurso", tipos_disponibles)
    with col2:
        idiomas = ["", "Español", "English", "Français", "Português", "Deutsch", "Chinese"]
        filtro_idioma = st.selectbox("Idioma", idiomas)
    with col3:
        filtro_fecha_modo = st.selectbox(
            "Modo de fecha",
            ["Exacto", "Antes de", "Después de"],
            key="filtro_fecha_modo",
        )
        filtro_anio = st.number_input("Año", min_value=0, max_value=2030, value=0, step=1)

    estrategias = ["", "fixed_size", "sentence_aware", "semantic"]
    filtro_estrategia = st.selectbox("Estrategia de chunking", estrategias)

# ── Sidebar: chat controls y fuentes ─────────────────────────────────────
with st.sidebar:
    st.markdown("### Chat RAG")
    if st.button("🗑 Limpiar conversación"):
        clear_messages()
        set_rag_context([], "", None)
        st.rerun()

    st.markdown("### Fuentes usadas en última respuesta")
    rag_context = st.session_state.get("rag_context", [])
    if rag_context:
        for i, ctx in enumerate(rag_context, 1):
            with st.expander(f"Fuente #{i}"):
                st.markdown(f"**Recurso:** {ctx.get('recurso', '—')}")
                st.markdown(f"**Score:** {ctx.get('score', 0):.3f}")
                st.markdown(f"**Estrategia:** {ctx.get('estrategia', '—')}")
                st.text(ctx.get("texto", "")[:200])
    else:
        st.info("No hay fuentes aún. Haz una consulta.")

# ── Ejecución de búsqueda ─────────────────────────────────────────────────
debemos_buscar = (modo == "Imagen -> Imagen" and imagen_subida is not None) or (modo != "Imagen -> Imagen" and query)

resultados = []

if debemos_buscar:
    if query:
        set_query(query)

    filtros: dict = {}
    if filtro_tipo:
        filtros["tipo"] = filtro_tipo
    if filtro_idioma:
        filtros["idioma"] = filtro_idioma
    if filtro_anio > 0:
        anio_str = f"{filtro_anio}"
        if filtro_fecha_modo == "Antes de":
            filtros["fecha_hasta"] = f"{anio_str}-12-31"
        elif filtro_fecha_modo == "Después de":
            filtros["fecha_desde"] = f"{anio_str}-01-01"
        else:
            filtros["fecha_desde"] = f"{anio_str}-01-01"
            filtros["fecha_hasta"] = f"{anio_str}-12-31"

    filtros = _auto_detectar_filtros(query, filtros)

    estrategia = filtro_estrategia or None

    with st.spinner("Buscando..."):
        try:
            with get_session() as session:
                if modo == "Imagen -> Imagen":
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(imagen_subida.read())
                        tmp_path = tmp.name
                    vector_img = get_embedding_imagen(tmp_path)
                    Path(tmp_path).unlink(missing_ok=True)
                    resultados_raw = buscar_imagenes_similares(session, vector_img, top_k=TOP_K)

                elif modo == "Texto -> Imagen":
                    filtros_img = {k: v for k, v in filtros.items() if v}
                    if filtros_img:
                        vector_img = get_embedding_texto_clip(query)
                        resultados_raw = buscar_imagenes_hibrido(
                            session, vector_img, top_k=TOP_K,
                            filtros=filtros_img,
                        )
                    else:
                        resultados_raw = buscar_imagenes_por_texto(session, query, top_k=TOP_K)

                elif modo == "Híbrida":
                    vector = get_embedding(query)
                    textos = buscar_hibrido_texto(
                        session, vector, top_k=TOP_K,
                        filtros=filtros if any(filtros.values()) else None,
                        estrategia=estrategia,
                    )
                    filtros_img = {k: v for k, v in filtros.items() if v and k in ("tipo_imagen", "recurso_tipo", "idioma", "fecha_desde", "fecha_hasta", "recurso_id")}
                    if filtros_img:
                        vector_img = get_embedding_texto_clip(query)
                        imagenes = buscar_imagenes_hibrido(
                            session, vector_img, top_k=TOP_K,
                            filtros=filtros_img,
                        )
                        resultados_raw = textos + imagenes
                    else:
                        resultados_raw = textos
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

            # Capturar SQL inmediatamente después de la búsqueda
            sql_mostrar = getattr(resultados_raw, 'sql', None)
            if sql_mostrar:
                st.session_state["_sql_ejecutado"] = sql_mostrar

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

    # ── Mostrar SQL ejecutado ────────────────────────────────────────────
    sql_mostrar = st.session_state.get("_sql_ejecutado")
    if sql_mostrar:
        with st.expander("🗄️ ---", expanded=False):
            st.code(sql_mostrar, language="sql")

# ── SEPARADOR ─────────────────────────────────────────────────────────────
st.markdown("---")

# ── CHAT RAG ──────────────────────────────────────────────────────────────
st.subheader("💬 Consulta con IA")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta sobre la biblioteca..."):
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando la biblioteca..."):
            try:
                estrategia_val = filtro_estrategia if filtro_estrategia else None

                filtros_rag: dict = {}
                if filtro_tipo:
                    filtros_rag["tipo"] = filtro_tipo
                if filtro_idioma:
                    filtros_rag["idioma"] = filtro_idioma
                if filtro_anio > 0:
                    anio_str = f"{filtro_anio}"
                    if filtro_fecha_modo == "Antes de":
                        filtros_rag["fecha_hasta"] = f"{anio_str}-12-31"
                    elif filtro_fecha_modo == "Después de":
                        filtros_rag["fecha_desde"] = f"{anio_str}-01-01"
                    else:
                        filtros_rag["fecha_desde"] = f"{anio_str}-01-01"
                        filtros_rag["fecha_hasta"] = f"{anio_str}-12-31"

                filtros_rag = _auto_detectar_filtros(prompt, filtros_rag)

                with get_session() as session:
                    resultado = run_rag(
                        session=session,
                        usuario_id=get_usuario_id(),
                        pregunta=prompt,
                        top_k=TOP_K,
                        filtros=filtros_rag or None,
                        estrategia=estrategia_val,
                        evaluar=False,
                    )

                respuesta = resultado["respuesta"]
                contexto = resultado["contexto"]
                consulta_id = resultado["consulta_id"]

                set_rag_context(contexto, respuesta, consulta_id)
                st.markdown(respuesta)

                if contexto:
                    with st.expander(f"📚 Fuentes utilizadas ({len(contexto)} fragmentos)"):
                        for i, ctx in enumerate(contexto, 1):
                            st.markdown(
                                f"[{i}] **{ctx.get('recurso', '—')}** "
                                f"(score: {ctx.get('score', 0):.3f}, "
                                f"estrategia: {ctx.get('estrategia', '—')})"
                            )
                            st.text(ctx.get("texto", "")[:150] + ("…" if len(ctx.get("texto", "")) > 150 else ""))

                sql_rag = resultado.get("sql_ejecutado")
                if sql_rag:
                    with st.expander("🗄️ SQL ejecutado en esta consulta", expanded=False):
                        st.code(sql_rag, language="sql")

                if st.button("📊 Evaluar esta respuesta", key=f"eval_{consulta_id}"):
                    with st.spinner("Calculando métricas RAGAS..."):
                        contextos_texto = [c["texto"] for c in contexto]
                        metricas = calcular_metricas_ragas(
                            pregunta=prompt,
                            respuesta=respuesta,
                            contextos=contextos_texto,
                        )
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("Faithfulness", f"{metricas.get('faithfulness', 0):.2%}")
                    col_m2.metric("Answer Relevancy", f"{metricas.get('answer_relevancy', 0):.2%}")
                    col_m3.metric("Context Recall", f"{metricas.get('context_recall', 0):.2%}")
                    st.caption(f"Motor de evaluación: {metricas.get('motor', 'N/A')}")

                add_message("assistant", respuesta)

            except Exception as exc:
                error_msg = f"Error al generar respuesta: {exc}"
                st.error(error_msg)
                add_message("assistant", error_msg)
