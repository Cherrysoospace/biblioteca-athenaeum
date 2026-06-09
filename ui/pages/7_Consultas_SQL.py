"""ui/pages/6_Consultas_SQL.py — Consola SQL para consultar la base de datos directamente."""

from __future__ import annotations

import sys
from pathlib import Path

_proj_root = str(Path(__file__).resolve().parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import streamlit as st
from sqlalchemy import text as sa_text
from core.database import get_session

st.set_page_config(page_title="Consultas SQL", layout="wide")
st.title("🗄  Consultas SQL")
st.markdown("Ejecutá consultas SQL directamente sobre la base de datos.")

# ── Sidebar: ayuda ───────────────────────────────────────────────────────
st.sidebar.markdown("### Tablas disponibles")
st.sidebar.markdown("""
- `usuarios`
- `recursos`
- `autores`
- `recurso_autores`
- `generos`
- `recurso_generos`
- `tags`
- `recurso_tags`
- `imagenes_recurso`
- `colecciones`
- `coleccion_recursos`
- `reseñas`
- `prestamos`
- `historial_lectura`
- `consultas`
- `resultados_consulta`
- `evaluaciones`
- `embeddings_texto`
- `embeddings_imagen`
- `embeddings_consulta`
""")

st.sidebar.markdown("### Ejemplos")
ejemplos = [
    "SELECT * FROM recursos LIMIT 5;",
    "SELECT id, titulo, tipo, fecha_publicacion FROM recursos WHERE tipo = 'libro' ORDER BY fecha_publicacion DESC LIMIT 10;",
    "SELECT r.titulo, AVG(rs.calificacion) AS calificacion_promedio FROM recursos r JOIN reseñas rs ON rs.recurso_id = r.id GROUP BY r.titulo ORDER BY calificacion_promedio DESC LIMIT 10;",
    "SELECT r.titulo, a.nombre AS autor FROM recursos r JOIN recurso_autores ra ON ra.recurso_id = r.id JOIN autores a ON a.id = ra.autor_id WHERE r.tipo = 'articulo' LIMIT 10;",
    "SELECT COUNT(*) AS total, tipo FROM recursos GROUP BY tipo ORDER BY total DESC;",
]
for e in ejemplos:
    if st.sidebar.button(e, use_container_width=True, key=f"ej_{e[:20]}"):
        st.session_state["sql_query"] = e

# ── Editor SQL ───────────────────────────────────────────────────────────
query = st.text_area(
    "SQL",
    value=st.session_state.get("sql_query", ""),
    height=150,
    placeholder="Escribí tu consulta SQL acá...",
    key="sql_query",
)

modo_escritura = st.checkbox("Permitir INSERT / UPDATE / DELETE", value=False)

# ── Ejecutar ─────────────────────────────────────────────────────────────
if st.button("Ejecutar", type="primary") and query.strip():
    sql_type = query.strip().split()[0].upper() if query.strip() else ""

    if sql_type in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE") and not modo_escritura:
        st.error("Operación de escritura detectada. Marcá 'Permitir INSERT / UPDATE / DELETE' para habilitarla.")
        st.stop()

    with st.spinner("Ejecutando consulta..."):
        try:
            with get_session() as session:
                result = session.execute(sa_text(query.strip()))

                if sql_type == "SELECT" or result.returns_rows:
                    rows = result.mappings().all()
                    if rows:
                        st.success(f"Consulta ejecutada — {len(rows)} fila(s) obtenida(s)")
                        st.dataframe(
                            [dict(row) for row in rows],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("Consulta ejecutada — 0 filas obtenidas.")
                else:
                    st.success(f"Consulta ejecutada — {result.rowcount} fila(s) afectada(s).")

        except Exception as exc:
            st.error(f"Error en la consulta: {exc}")

# ── Limpiar ──────────────────────────────────────────────────────────────
if st.button("Limpiar"):
    st.session_state["sql_query"] = ""
    st.rerun()
