"""ui/components/metrics_panel.py — Panel de métricas RAGAS con st.metric."""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_metrics(evaluacion: dict[str, Any], promedio: dict[str, float] | None = None) -> None:
    """Muestra las tres métricas RAGAS en columnas con delta respecto al promedio.

    Args:
        evaluacion: Dict con keys faithfulness, answer_relevancy, context_recall.
        promedio: Dict opcional con los mismos keys para calcular delta.
    """
    if not evaluacion:
        st.info("No hay métricas disponibles para esta evaluación.")
        return

    faith = evaluacion.get("faithness") or evaluacion.get("faithfulness", 0.0)
    relevancy = evaluacion.get("answer_relevancy", 0.0)
    recall = evaluacion.get("context_recall", 0.0)

    def _delta(valor: float, key: str) -> float | None:
        if promedio and key in promedio:
            return round(valor - promedio[key], 4)
        return None

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Faithfulness",
            f"{faith:.2%}",
            delta=_delta(faith, "faithfulness"),
            help="¿La respuesta es fiel al contexto recuperado?",
        )
    with col2:
        st.metric(
            "Answer Relevancy",
            f"{relevancy:.2%}",
            delta=_delta(relevancy, "answer_relevancy"),
            help="¿La respuesta es relevante para la pregunta?",
        )
    with col3:
        st.metric(
            "Context Recall",
            f"{recall:.2%}",
            delta=_delta(recall, "context_recall"),
            help="¿El contexto recuperado cubre la respuesta esperada?",
        )
