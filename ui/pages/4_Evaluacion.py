"""ui/pages/4_Evaluacion.py — Dashboard de métricas RAGAS con histórico."""

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
                Evaluacion.context_precision,
                Evaluacion.answer_correctness,
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
            "faithfulness": float(r["faithfulness"]) if r["faithfulness"] is not None else None,
            "answer_relevancy": float(r["answer_relevancy"]) if r["answer_relevancy"] is not None else None,
            "context_recall": float(r["context_recall"]) if r["context_recall"] is not None else None,
            "context_precision": float(r["context_precision"]) if r["context_precision"] is not None else None,
            "answer_correctness": float(r["answer_correctness"]) if r["answer_correctness"] is not None else None,
            "fecha": str(r["fecha"]) if r["fecha"] else "",
        }
        for r in rows
    ]
    return pd.DataFrame(data)


df = cargar_historial()

# ── Métricas agregadas ───────────────────────────────────────────────────
if not df.empty:
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric(
        "Faithfulness",
        f"{df['faithfulness'].mean():.2%}" if pd.notna(df['faithfulness'].mean()) else "N/A",
    )
    col_m2.metric(
        "Answer Relevancy",
        f"{df['answer_relevancy'].mean():.2%}" if pd.notna(df['answer_relevancy'].mean()) else "N/A",
    )
    col_m3.metric(
        "Context Recall",
        f"{df['context_recall'].mean():.2%}" if pd.notna(df['context_recall'].mean()) else "N/A",
    )
    col_m4.metric(
        "Context Precision",
        f"{df['context_precision'].mean():.2%}" if pd.notna(df['context_precision'].mean()) else "N/A",
    )
    col_m5.metric(
        "Answer Correctness",
        f"{df['answer_correctness'].mean():.2%}" if pd.notna(df['answer_correctness'].mean()) else "N/A",
    )

    # ── Gráfico de evolución ──────────────────────────────────────────────
    st.subheader("Evolución de métricas")
    cols_chart = [
        c for c in
        ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
        if c in df.columns and df[c].notna().any()
    ]
    df_chart = df.dropna(subset=cols_chart).copy()
    if not df_chart.empty:
        df_chart["idx"] = range(len(df_chart))
        st.line_chart(df_chart.set_index("idx")[cols_chart])

    # ── Tabla de histórico ────────────────────────────────────────────────
    st.subheader("Histórico de evaluaciones")
    cols_display = [
        c for c in
        ["id", "pregunta", "faithfulness", "answer_relevancy", "context_recall",
         "context_precision", "answer_correctness", "fecha"]
        if c in df.columns
    ]
    col_config = {
        "faithfulness": st.column_config.ProgressColumn("Faithfulness", format="%.2f", min_value=0, max_value=1),
        "answer_relevancy": st.column_config.ProgressColumn("Answer Relevancy", format="%.2f", min_value=0, max_value=1),
        "context_recall": st.column_config.ProgressColumn("Context Recall", format="%.2f", min_value=0, max_value=1),
        "context_precision": st.column_config.ProgressColumn("Context Precision", format="%.2f", min_value=0, max_value=1),
        "answer_correctness": st.column_config.ProgressColumn("Answer Correctness", format="%.2f", min_value=0, max_value=1),
    }
    st.dataframe(
        df[cols_display],
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )

    st.caption(f"Mostrando {len(df)} evaluaciones recientes")
else:
    st.info("Aún no hay evaluaciones registradas. Realiza consultas desde el Chat RAG y evalúalas.")
