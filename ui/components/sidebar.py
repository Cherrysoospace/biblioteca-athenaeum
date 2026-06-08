"""ui/components/sidebar.py — Sidebar compartido con selector de usuario y estado de BD."""

from __future__ import annotations

import streamlit as st
from core.database import check_connection
from ui.utils.session import get_usuario_id, set_usuario_id


def render_sidebar() -> None:
    """Renderiza el sidebar común a todas las páginas."""
    with st.sidebar:
        st.markdown("## 🏛 Athenaeum")
        st.markdown("---")

        # ── Selector de usuario simulado ────────────────────────────────────
        st.markdown("### Usuario")
        uid = get_usuario_id()
        nuevo_uid = st.number_input(
            "ID de usuario",
            min_value=1,
            max_value=9999,
            value=uid,
            step=1,
            label_visibility="collapsed",
            key="sidebar_usuario_id",
        )
        if nuevo_uid != uid:
            set_usuario_id(nuevo_uid)
            st.rerun()

        # ── Estado de conexión a BD ─────────────────────────────────────────
        st.markdown("### Conexión BD")
        conectado = check_connection()
        if conectado:
            st.success("✅ Base de datos conectada")
        else:
            st.error("❌ Sin conexión a la BD")

        st.markdown("---")
        st.caption("Biblioteca Athenaeum v1.0")
