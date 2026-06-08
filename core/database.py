"""
core/database.py — Gestión de sesiones SQLAlchemy para PostgreSQL + pgvector.

Provee:
  - engine       : instancia global del motor SQLAlchemy
  - get_session  : context manager que abre y cierra una sesión con commit/rollback
  - init_db      : crea la extensión pgvector y verifica conectividad
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

from core.config import DATABASE_URL

logger = logging.getLogger(__name__)

# ── Motor global ──────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # reconecta si la conexión está muerta
    pool_size=5,
    max_overflow=10,
    echo=False,               # poner True para ver SQL en desarrollo
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Base declarativa compartida por todos los modelos ─────────────────────────
class Base(DeclarativeBase):
    pass


# ── Context manager de sesión ─────────────────────────────────────────────────
@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Abre una sesión de base de datos, hace commit al salir sin error,
    o rollback si ocurre una excepción, y cierra siempre la sesión.

    Uso:
        with get_session() as session:
            session.add(mi_objeto)
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Inicialización ────────────────────────────────────────────────────────────
def init_db() -> None:
    """
    - Activa la extensión pgvector (idempotente).
    - Verifica que la conexión funciona.
    - No crea tablas (las tablas se crean con el script SQL del proyecto).
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
        result = conn.execute(text("SELECT version()"))
        version = result.scalar()
        logger.info("Conectado a PostgreSQL: %s", version)

    logger.info("Extensión pgvector activa. Base de datos lista.")


def check_connection() -> bool:
    """Devuelve True si la conexión a la BD es exitosa."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("Error de conexión a la BD: %s", exc)
        return False
