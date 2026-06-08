"""
models/embeddings_consulta.py — Modelo ORM para Embeddings_Consulta.

Almacena el vector semántico de cada consulta del usuario,
generado con el mismo modelo que los chunks de texto (all-MiniLM-L6-v2, 384 dims).
Relación 1:1 con Consultas (consulta_id UNIQUE).
"""

from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from core.database import Base
from core.config import EMBEDDING_DIM_TEXTO, MINILM_MODEL


class EmbeddingConsulta(Base):
    __tablename__ = "embeddings_consulta"

    id: Mapped[int] = mapped_column(primary_key=True)
    consulta_id: Mapped[int] = mapped_column(
        ForeignKey("consultas.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,   # relación 1:1
    )
    # Vector de 384 dimensiones (debe coincidir con Embeddings_Texto para comparación)
    vector_embedding: Mapped[list] = mapped_column(
        Vector(EMBEDDING_DIM_TEXTO), nullable=False
    )
    modelo: Mapped[str] = mapped_column(
        String(200), nullable=False, default=MINILM_MODEL
    )

    # Relación
    consulta: Mapped["Consulta"] = relationship(  # type: ignore[name-defined]
        "Consulta", back_populates="embedding_consulta"
    )

    def __repr__(self) -> str:
        return f"<EmbeddingConsulta id={self.id} consulta_id={self.consulta_id}>"
