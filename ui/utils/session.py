"""ui/utils/session.py — Manejo centralizado de st.session_state."""

from __future__ import annotations

from typing import Any

import streamlit as st


_DEFAULT_USUARIO_ID = 1


def init_session() -> None:
    """Inicializa todas las claves de session_state con valores por defecto."""
    _defaults: dict[str, Any] = {
        "usuario_id": _DEFAULT_USUARIO_ID,
        "messages": [],
        "current_query": "",
        "current_mode": "Texto → Texto",
        "current_results": [],
        "rag_context": [],
        "rag_answer": "",
        "rag_consulta_id": None,
        "experiment_running": False,
        "_sql_ejecutado": None,
    }
    for key, value in _defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_usuario_id() -> int:
    return st.session_state.get("usuario_id", _DEFAULT_USUARIO_ID)


def set_usuario_id(uid: int) -> None:
    st.session_state.usuario_id = uid


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def clear_messages() -> None:
    st.session_state.messages = []


def set_query(query: str) -> None:
    st.session_state.current_query = query


def set_results(results: list[dict]) -> None:
    st.session_state.current_results = results


def set_rag_context(context: list[dict], answer: str, consulta_id: int | None) -> None:
    st.session_state.rag_context = context
    st.session_state.rag_answer = answer
    st.session_state.rag_consulta_id = consulta_id
