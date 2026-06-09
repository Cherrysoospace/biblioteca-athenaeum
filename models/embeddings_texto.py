"""
models/embeddings_texto.py — Modelo ORM para Embeddings_Texto.

Almacena los chunks de texto vectorizados con all-MiniLM-L6-v2 (384 dims).
Cada fila corresponde a un chunk de un recurso bajo una estrategia de chunking
específica (fixed_size | sentence_aware | semantic).
"""

from typing import Optional
from sqlalchemy import String, Text, Integer, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from core.database import Base
from core.config import EMBEDDING_DIM_TEXTO, MINILM_MODEL


class EmbeddingTexto(Base):
    __tablename__ = "embeddings_texto"

    id: Mapped[int] = mapped_column(primary_key=True)
    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_id: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_texto: Mapped[str] = mapped_column(Text, nullable=False)
    estrategia_chunking: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True
    )
    # Vector de 384 dimensiones (all-MiniLM-L6-v2)
    vector_texto_384: Mapped[list] = mapped_column(
        Vector(EMBEDDING_DIM_TEXTO), nullable=False
    )
    modelo: Mapped[str] = mapped_column(
        String(200), nullable=False, default=MINILM_MODEL
    )

    # Relación hacia Recurso
    recurso: Mapped["Recurso"] = relationship(  # type: ignore[name-defined]
        "Recurso", back_populates="embeddings_texto"
    )
    # Un embedding de texto puede aparecer en resultados de consulta
    resultados: Mapped[list] = relationship(
        "ResultadoConsulta", back_populates="embedding_texto",
        foreign_keys="ResultadoConsulta.embedding_texto_id"
    )

    def __repr__(self) -> str:
        return (
            f"<EmbeddingTexto id={self.id} recurso_id={self.recurso_id} "
            f"chunk_id={self.chunk_id} estrategia={self.estrategia_chunking!r}>"
        )
