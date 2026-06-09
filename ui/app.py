"""ui/app.py — Punto de entrada de la interfaz Streamlit. Configuración global y navegación."""

import sys
from pathlib import Path

# Asegura que la raíz del proyecto esté en sys.path para imports absolutos
_proj_root = str(Path(__file__).resolve().parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Importar todos los modelos ANTES que cualquier otro módulo toque la BD
import models  # noqa: F401

import streamlit as st
from ui.utils.session import init_session
from ui.components.sidebar import render_sidebar

st.set_page_config(
    page_title="Biblioteca Athenaeum",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session()

render_sidebar()

# ── Logo / título
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("<div style='font-size:3rem;text-align:right'>🏛</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0'>Biblioteca Athenaeum</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='margin-top:-0.5rem;color:#888'>"
        "Sistema híbrido relacional-vectorial · RAG · Búsqueda semántica</p>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Navegación con pestañas
st.markdown("### Navegación")
st.page_link("pages/1_Busqueda.py", label="🔍  Búsqueda y Chat RAG", use_container_width=True)
st.page_link("pages/2_Catalogo.py", label="📚  Catálogo Relacional", use_container_width=True)
st.page_link("pages/4_Evaluacion.py", label="📊  Dashboard RAGAS", use_container_width=True)
st.page_link("pages/5_Experimento.py", label="🧪  Experimento Chunking", use_container_width=True)
st.page_link("pages/6_Administracion.py", label="⚙  Administración", use_container_width=True)
st.page_link("pages/7_Consultas_SQL.py", label="🗄  Consultas SQL", use_container_width=True)
