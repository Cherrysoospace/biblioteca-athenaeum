"""ui/pages/3_RAG_Chat.py — Chat RAG con LLM usando contexto recuperado de la biblioteca."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import streamlit as st
from core.database import get_session
from core.config import TOP_K
from pipeline.rag import run_rag
from pipeline.evaluacion import calcular_metricas_ragas
from ui.utils.session import get_usuario_id, add_message, clear_messages, set_rag_context

_RE_ANIO = r'\b(1[89]\d{2}|20[0-2]\d)\b'


def _auto_detectar_fecha_rag(prompt: str, filtros: dict) -> None:
    """Detecta año direccional (antes/después/desde/hasta) desde el texto."""
    year_match = re.search(_RE_ANIO, prompt)
    if not year_match:
        return
    anio = year_match.group(1)
    pre = prompt[: year_match.start()].lower()

    if re.search(r'(antes|pret[eé]rito|previo|anterior)\s*(de|a|al)?\s*$', pre):
        filtros["fecha_hasta"] = f"{anio}-12-31"
    elif re.search(r'(despu[eé]s|posterior|subsiguiente|luego)\s*(de|a|al)?\s*$', pre):
        filtros["fecha_desde"] = f"{anio}-01-01"
    elif re.search(r'(desde|a\s*partir\s*de)\s*$', pre):
        filtros["fecha_desde"] = f"{anio}-01-01"
    elif re.search(r'(hasta|como\s*m[aá]ximo)\s*$', pre):
        filtros["fecha_hasta"] = f"{anio}-12-31"
    else:
        filtros["fecha_desde"] = f"{anio}-01-01"
        filtros["fecha_hasta"] = f"{anio}-12-31"


def _auto_detectar_tipo_rag(prompt: str, filtros: dict) -> None:
    """Detecta tipo de recurso desde palabras clave en el texto."""
    if re.search(r'\blibros?\b', prompt, re.IGNORECASE):
        filtros["tipo"] = "libro"
    elif re.search(r'\brevistas?\b', prompt, re.IGNORECASE):
        filtros["tipo"] = "revista"
    elif re.search(r'\bartículos?\b', prompt, re.IGNORECASE):
        filtros["tipo"] = "articulo"


st.set_page_config(page_title="Chat RAG", layout="wide")
st.title("💬 Chat RAG")

# ── Selector de estrategia ────────────────────────────────────────────────
estrategia = st.sidebar.selectbox(
    "Estrategia de chunking",
    ["", "fixed_size", "sentence_aware", "semantic"],
    index=0,
    key="rag_estrategia",
)

# ── Filtros relacionales en sidebar ──────────────────────────────────────
with st.sidebar.expander("Filtros avanzados (opcional)", expanded=False):
    rag_filtro_tipo = st.selectbox(
        "Tipo de recurso",
        ["", "libro", "articulo", "revista", "mapa", "fotografia", "tesis"],
        key="rag_filtro_tipo",
    )
    rag_filtro_fecha_modo = st.selectbox(
        "Modo de fecha",
        ["Exacto", "Antes de", "Después de"],
        key="rag_filtro_fecha_modo",
    )
    rag_filtro_anio = st.number_input(
        "Año",
        min_value=0, max_value=2030, value=0, step=1,
        key="rag_filtro_anio",
    )

# ── Limpiar conversación ─────────────────────────────────────────────────
if st.sidebar.button("🗑 Limpiar conversación"):
    clear_messages()
    set_rag_context([], "", None)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Fuentes usadas en última respuesta")

rag_context = st.session_state.get("rag_context", [])
if rag_context:
    for i, ctx in enumerate(rag_context, 1):
        with st.sidebar.expander(f"Fuente #{i}"):
            st.markdown(f"**Recurso:** {ctx.get('recurso', '—')}")
            st.markdown(f"**Score:** {ctx.get('score', 0):.3f}")
            st.markdown(f"**Estrategia:** {ctx.get('estrategia', '—')}")
            st.text(ctx.get("texto", "")[:200])
else:
    st.sidebar.info("No hay fuentes aún. Haz una consulta.")

# ── Chat ──────────────────────────────────────────────────────────────────
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
                estrategia_val = estrategia if estrategia else None

                # Construir filtros desde el sidebar
                filtros_rag: dict = {}
                if rag_filtro_tipo:
                    filtros_rag["tipo"] = rag_filtro_tipo
                if rag_filtro_anio > 0:
                    anio_str = f"{rag_filtro_anio}"
                    if rag_filtro_fecha_modo == "Antes de":
                        filtros_rag["fecha_hasta"] = f"{anio_str}-12-31"
                    elif rag_filtro_fecha_modo == "Después de":
                        filtros_rag["fecha_desde"] = f"{anio_str}-01-01"
                    else:
                        filtros_rag["fecha_desde"] = f"{anio_str}-01-01"
                        filtros_rag["fecha_hasta"] = f"{anio_str}-12-31"

                # Auto-detectar año direccional desde el texto de la consulta
                if rag_filtro_anio == 0:
                    _auto_detectar_fecha_rag(prompt, filtros_rag)

                # Auto-detectar tipo en la consulta si no se especificó manualmente
                if not rag_filtro_tipo:
                    _auto_detectar_tipo_rag(prompt, filtros_rag)

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

                # Mostrar fuentes usadas
                if contexto:
                    with st.expander(f"📚 Fuentes utilizadas ({len(contexto)} fragmentos)"):
                        for i, ctx in enumerate(contexto, 1):
                            st.markdown(
                                f"[{i}] **{ctx.get('recurso', '—')}** "
                                f"(score: {ctx.get('score', 0):.3f}, "
                                f"estrategia: {ctx.get('estrategia', '—')})"
                            )
                            st.text(ctx.get("texto", "")[:150] + ("…" if len(ctx.get("texto", "")) > 150 else ""))

                # Botón de evaluación
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
