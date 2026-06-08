"""ui/pages/4_Evaluacion.py — Dashboard de métricas RAGAS con histórico y experimentos."""

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
from models.consultas import Evaluacion, Consulta
from pipeline.evaluacion import experimento_chunking

st.set_page_config(page_title="Dashboard RAGAS", layout="wide")
st.title("📊 Dashboard de Evaluación RAGAS")

# ── Cargar histórico ─────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def cargar_historial() -> pd.DataFrame:
    with get_session() as session:
        rows = session.execute(
            select(
                Evaluacion.id,
                Evaluacion.consulta_id,
                Consulta.texto_pregunta,
                Evaluacion.faithfulness,
                Evaluacion.answer_relevancy,
                Evaluacion.context_recall,
                Evaluacion.fecha,
            )
            .join(Consulta, Consulta.id == Evaluacion.consulta_id)
            .order_by(Evaluacion.fecha.desc())
            .limit(200)
        ).mappings().all()

    data = [
        {
            "id": r["id"],
            "consulta_id": r["consulta_id"],
            "pregunta": r["texto_pregunta"][:60] + "…" if len(r["texto_pregunta"]) > 60 else r["texto_pregunta"],
            "faithfulness": float(r["faithfulness"]) if r["faithfulness"] else None,
            "answer_relevancy": float(r["answer_relevancy"]) if r["answer_relevancy"] else None,
            "context_recall": float(r["context_recall"]) if r["context_recall"] else None,
            "fecha": str(r["fecha"]) if r["fecha"] else "",
        }
        for r in rows
    ]
    return pd.DataFrame(data)


df = cargar_historial()

# ── Métricas agregadas ───────────────────────────────────────────────────
if not df.empty:
    prom_faith = df["faithfulness"].mean()
    prom_relevancy = df["answer_relevancy"].mean()
    prom_recall = df["context_recall"].mean()

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Faithfulness promedio", f"{prom_faith:.2%}" if pd.notna(prom_faith) else "N/A")
    col_m2.metric("Answer Relevancy promedio", f"{prom_relevancy:.2%}" if pd.notna(prom_relevancy) else "N/A")
    col_m3.metric("Context Recall promedio", f"{prom_recall:.2%}" if pd.notna(prom_recall) else "N/A")

    # ── Gráfico de evolución ──────────────────────────────────────────────
    st.subheader("Evolución de métricas")
    df_chart = df.dropna(subset=["faithfulness", "answer_relevancy", "context_recall"]).copy()
    if not df_chart.empty:
        df_chart["idx"] = range(len(df_chart))
        st.line_chart(
            df_chart.set_index("idx")[["faithfulness", "answer_relevancy", "context_recall"]],
        )

    # ── Tabla de histórico ────────────────────────────────────────────────
    st.subheader("Histórico de evaluaciones")
    st.dataframe(
        df[["id", "pregunta", "faithfulness", "answer_relevancy", "context_recall", "fecha"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "faithfulness": st.column_config.ProgressColumn("Faithfulness", format="%.2f", min_value=0, max_value=1),
            "answer_relevancy": st.column_config.ProgressColumn("Answer Relevancy", format="%.2f", min_value=0, max_value=1),
            "context_recall": st.column_config.ProgressColumn("Context Recall", format="%.2f", min_value=0, max_value=1),
        },
    )

    st.caption(f"Mostrando {len(df)} evaluaciones recientes")
else:
    st.info("Aún no hay evaluaciones registradas. Realiza consultas desde el Chat RAG y evalúalas.")

# ── Comparativa por estrategia ───────────────────────────────────────────
st.markdown("---")
st.subheader("Comparativa por estrategia de chunking")

st.info(
    "La comparativa por estrategia estará disponible cuando haya "
    "evaluaciones de consultas con diferentes estrategias de chunking."
)

# ── Experimento completo ─────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧪 Ejecutar experimento completo")

col_e1, col_e2 = st.columns([3, 1])
with col_e1:
    st.markdown(
        "Ejecuta las 10 consultas de prueba sobre las 3 estrategias de chunking "
        "y almacena las métricas RAGAS."
    )
with col_e2:
    ejecutar = st.button("▶ Ejecutar experimento", type="primary", disabled=st.session_state.get("experiment_running", False))

if ejecutar:
    st.session_state.experiment_running = True
    consultas_prueba = [
        {"pregunta": "¿Qué obras de Gabriel García Márquez hay en la biblioteca?"},
        {"pregunta": "¿Qué recursos hay sobre literatura latinoamericana contemporánea?"},
        {"pregunta": "¿Qué libros trata sobre la historia de Colombia en el siglo XIX?"},
        {"pregunta": "¿Hay artículos sobre inteligencia artificial en español?"},
        {"pregunta": "¿Qué recursos educativos están disponibles para ciencias naturales?"},
        {"pregunta": "¿Qué autores mexicanos están en el catálogo?"},
        {"pregunta": "¿Hay mapas históricos de América del Sur?"},
        {"pregunta": "¿Qué recursos tiene la biblioteca sobre cambio climático?"},
        {"pregunta": "¿Qué obras de ficción tienen calificación mayor a 4?"},
        {"pregunta": "¿Qué autores argentinos están disponibles en portugués?"},
    ]

    progress_bar = st.progress(0, text="Iniciando experimento...")
    status_text = st.status("Ejecutando...", expanded=False)

    try:
        with get_session() as session:
            status_text.write("Iniciando experimento de chunking...")
            resultados = experimento_chunking(
                session=session,
                usuario_id=1,
                consultas=consultas_prueba,
            )

        for i, r in enumerate(resultados):
            progress_bar.progress((i + 1) / len(resultados), text=f"Procesando consulta {i + 1}/{len(resultados)}")
            status_text.write(f"{r.get('pregunta', '—')[:40]}... [{r.get('estrategia', '—')}] → OK")

        progress_bar.progress(1.0, text="Experimento completado")
        status_text.write("✅ Experimento finalizado exitosamente.")
        st.success(f"Experimento completado: {len(resultados)} evaluaciones generadas.")
        st.cache_data.clear()
        st.rerun()

    except Exception as exc:
        st.error(f"Error durante el experimento: {exc}")
        status_text.write(f"❌ Error: {exc}")

    st.session_state.experiment_running = False
