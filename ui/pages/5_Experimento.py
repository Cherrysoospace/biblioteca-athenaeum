"""ui/pages/5_Experimento.py — Experimento de chunking y evaluación RAGAS."""

from __future__ import annotations

import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import pandas as pd
import streamlit as st
from sqlalchemy import select, func
from core.database import get_session
from models.consultas import Evaluacion, Consulta
from pipeline.evaluacion import experimento_chunking, evaluar_dataset
from data.eval_dataset import EVAL_DATASET

st.set_page_config(page_title="Experimento Chunking", layout="wide")
st.title("🧪 Evaluación RAGAS")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_comparativa, tab_dataset, tab_ejecutar = st.tabs([
    "Comparativa por estrategia",
    "Evaluación con dataset (25 preguntas)",
    "Ejecutar experimento",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Comparativa por estrategia de chunking
# ══════════════════════════════════════════════════════════════════════════════
# ── Limpieza de evaluaciones ──────────────────────────────────────────────────

def limpiar_evaluaciones(session) -> int:
    """Elimina Evaluaciones, Resultados y Embeddings de Consulta (conserva las Consultas)."""
    from models.embeddings_consulta import EmbeddingConsulta
    from models.consultas import ResultadoConsulta, Evaluacion

    evals = session.execute(
        select(Evaluacion.id, Evaluacion.consulta_id)
    ).all()
    consulta_ids = list({r.consulta_id for r in evals if r.consulta_id})
    n_evals = len(evals)
    if not consulta_ids:
        return 0

    session.execute(
        ResultadoConsulta.__table__.delete()
        .where(ResultadoConsulta.consulta_id.in_(consulta_ids))
    )
    session.execute(
        EmbeddingConsulta.__table__.delete()
        .where(EmbeddingConsulta.consulta_id.in_(consulta_ids))
    )
    session.execute(
        Evaluacion.__table__.delete()
        .where(Evaluacion.consulta_id.in_(consulta_ids))
    )
    session.commit()

    total = n_evals + len(consulta_ids) * 2
    return total


with tab_comparativa:
    col_header, col_btn = st.columns([3, 1])
    with col_header:
        st.subheader("Comparativa por estrategia de chunking")
    with col_btn:
        with get_session() as session:
            count = session.scalar(select(func.count(Evaluacion.id)))
        if count and count > 0:
            if st.button("🗑️ Limpiar evaluaciones", type="secondary"):
                with get_session() as session:
                    n = limpiar_evaluaciones(session)
                st.success(f"{n} registros eliminados.")
                st.cache_data.clear()
                st.rerun()

    @st.cache_data(ttl=60)
    def cargar_comparativa_estrategias() -> pd.DataFrame:
        from models.embeddings_texto import EmbeddingTexto
        from models.consultas import ResultadoConsulta

        with get_session() as session:
            subq = (
                select(
                    ResultadoConsulta.consulta_id,
                    EmbeddingTexto.estrategia_chunking,
                )
                .join(EmbeddingTexto, EmbeddingTexto.id == ResultadoConsulta.embedding_texto_id)
                .group_by(ResultadoConsulta.consulta_id, EmbeddingTexto.estrategia_chunking)
                .subquery()
            )
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
                    subq.c.estrategia_chunking,
                )
                .join(Consulta, Consulta.id == Evaluacion.consulta_id)
                .join(subq, subq.c.consulta_id == Consulta.id)
                .order_by(Evaluacion.fecha.desc())
                .limit(200)
            ).mappings().all()

        data = [
            {
                "consulta_id": r["consulta_id"],
                "pregunta": r["texto_pregunta"][:60] + "…" if len(r["texto_pregunta"]) > 60 else r["texto_pregunta"],
                "estrategia": r["estrategia_chunking"],
                "faithfulness": float(r["faithfulness"]) if r["faithfulness"] is not None else None,
                "answer_relevancy": float(r["answer_relevancy"]) if r["answer_relevancy"] is not None else None,
                "context_recall": float(r["context_recall"]) if r["context_recall"] is not None else None,
                "context_precision": float(r["context_precision"]) if r["context_precision"] is not None else None,
                "answer_correctness": float(r["answer_correctness"]) if r["answer_correctness"] is not None else None,
            }
            for r in rows
        ]
        return pd.DataFrame(data)

    df_comp = cargar_comparativa_estrategias()

    st.subheader("Comparativa por estrategia de chunking")
    if not df_comp.empty:
        metric_cols = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "answer_correctness"]
        col_avg = df_comp.groupby("estrategia")[metric_cols].mean().reset_index()
        col_avg.columns = ["Estrategia"] + [f"{m} (prom)" for m in metric_cols]

        col_config_avg = {
            "Estrategia": "Estrategia",
        }
        for m in metric_cols:
            col_config_avg[f"{m} (prom)"] = st.column_config.ProgressColumn(
                f"{m} (prom)", format="%.2f", min_value=0, max_value=1,
            )
        st.dataframe(col_avg, use_container_width=True, hide_index=True, column_config=col_config_avg)

        display_cols = ["consulta_id", "pregunta", "estrategia"] + metric_cols
        col_config_det = {
            "consulta_id": "Consulta ID",
            "pregunta": "Pregunta",
            "estrategia": "Estrategia",
        }
        for m in metric_cols:
            col_config_det[m] = st.column_config.ProgressColumn(
                m, format="%.2f", min_value=0, max_value=1,
            )

        with st.expander("Ver detalle individual por consulta", expanded=False):
            st.dataframe(
                df_comp[display_cols],
                use_container_width=True,
                hide_index=True,
                column_config=col_config_det,
            )
    else:
        st.info("Aún no hay evaluaciones con estrategias registradas. Ejecuta el experimento para generar datos.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Evaluación con dataset de 25 preguntas
# ══════════════════════════════════════════════════════════════════════════════
with tab_dataset:
    st.subheader("Evaluación con dataset de 25 pares (pregunta, ground_truth)")
    st.markdown(
        "Ejecuta el pipeline RAG sobre las 25 preguntas del dataset de evaluación "
        "con ground_truth y registra todas las métricas, incluyendo "
        "**context_precision** y **answer_correctness**."
    )
    st.caption(f"Dataset: {len(EVAL_DATASET)} pares cargados desde data/eval_dataset.py")

    col_ds1, col_ds2 = st.columns([3, 1])
    with col_ds1:
        estrategia_ds = st.selectbox(
            "Estrategia de chunking",
            options=[None, "fixed_size", "sentence_aware", "semantic"],
            format_func=lambda x: "Todas (promedio)" if x is None else x.replace("_", " ").title(),
            key="ds_estrategia",
        )
    with col_ds2:
        ejecutar_ds = st.button(
            "▶ Ejecutar evaluación",
            type="primary",
            disabled=st.session_state.get("dataset_running", False),
            key="btn_dataset",
        )

    if ejecutar_ds:
        st.session_state.dataset_running = True
        st.session_state.ds_resultados = None
        ds_progress = st.progress(0, text="Iniciando evaluación del dataset...")
        ds_status = st.status("Ejecutando...", expanded=False)

        try:
            with get_session() as session:
                ds_status.write("Evaluando dataset...")
                resultados_ds = evaluar_dataset(
                    session=session,
                    usuario_id=1,
                    dataset=EVAL_DATASET,
                    estrategia=estrategia_ds,
                )

            for i, r in enumerate(resultados_ds):
                ds_progress.progress(
                    (i + 1) / len(resultados_ds),
                    text=f"Procesando {i + 1}/{len(resultados_ds)}",
                )
                ds_status.write(f"{r.get('pregunta', '—')[:40]}... → OK")

            ds_progress.progress(1.0, text="Evaluación completada")
            ds_status.write("✅ Evaluación del dataset finalizada.")
            st.success(f"Evaluación completada: {len(resultados_ds)} consultas procesadas.")
            st.session_state.ds_resultados = resultados_ds
            st.cache_data.clear()

        except Exception as exc:
            st.error(f"Error durante la evaluación: {exc}")
            ds_status.write(f"❌ Error: {exc}")

        st.session_state.dataset_running = False

    # Mostrar resultados siempre que existan en sesión
    if st.session_state.get("ds_resultados"):
        df_ds = pd.DataFrame(st.session_state.ds_resultados)
        metric_cols_ds = [
            c for c in
            ["faithfulness", "answer_relevancy", "context_recall",
             "context_precision", "answer_correctness"]
            if c in df_ds.columns
        ]

        st.subheader("Resultados promedio")
        avg_row = {m: df_ds[m].mean() for m in metric_cols_ds if df_ds[m].notna().any()}
        if avg_row:
            avg_cols = st.columns(len(avg_row))
            for col, (m, v) in zip(avg_cols, avg_row.items()):
                col.metric(m.replace("_", " ").title(), f"{v:.2%}")

        with st.expander("Ver detalle individual", expanded=False):
            cols_display = ["pregunta", "ground_truth"] + metric_cols_ds
            cols_display = [c for c in cols_display if c in df_ds.columns]
            st.dataframe(df_ds[cols_display], use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Ejecutar experimento de chunking (10 consultas × 3 estrategias)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ejecutar:
    st.subheader("Ejecutar experimento de chunking")
    col_e1, col_e2 = st.columns([3, 1])
    with col_e1:
        st.markdown(
            "Ejecuta las 10 consultas de prueba sobre las 3 estrategias de chunking "
            "y almacena las métricas RAGAS."
        )
    with col_e2:
        ejecutar = st.button(
            "▶ Ejecutar experimento",
            type="primary",
            disabled=st.session_state.get("experiment_running", False),
            key="btn_experimento",
        )

    if ejecutar:
        st.session_state.experiment_running = True
        st.session_state.exp_resultados = None
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
                progress_bar.progress(
                    (i + 1) / len(resultados),
                    text=f"Procesando consulta {i + 1}/{len(resultados)}",
                )
                status_text.write(f"{r.get('pregunta', '—')[:40]}... [{r.get('estrategia', '—')}] → OK")

            progress_bar.progress(1.0, text="Experimento completado")
            status_text.write("✅ Experimento finalizado exitosamente.")
            st.success(f"Experimento completado: {len(resultados)} evaluaciones generadas.")
            st.session_state.exp_resultados = resultados
            st.cache_data.clear()

        except Exception as exc:
            st.error(f"Error durante el experimento: {exc}")
            status_text.write(f"❌ Error: {exc}")

        st.session_state.experiment_running = False

    # Mostrar resultados del experimento siempre que existan en sesión
    if st.session_state.get("exp_resultados"):
        df_exp = pd.DataFrame(st.session_state.exp_resultados)
        n_ok = len(df_exp)
        n_err = df_exp["error"].notna().sum() if "error" in df_exp.columns else 0
        n_ok -= n_err

        if n_err:
            st.warning(f"{n_err} de {len(df_exp)} consultas fallaron.")
            with st.expander("Ver errores", expanded=True):
                st.dataframe(
                    df_exp[df_exp["error"].notna()][["pregunta", "estrategia", "error"]],
                    use_container_width=True, hide_index=True,
                )
        if n_ok:
            st.success(f"{n_ok} consultas ejecutadas correctamente.")
