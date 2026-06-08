"""
models/consultas.py — Modelos ORM para Consultas, Resultados_Consulta y Evaluaciones.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Text, DateTime, SmallInteger, Numeric, Integer,
    ForeignKey, func, CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


class Consulta(Base):
    __tablename__ = "consultas"

    id: Mapped[int] = mapped_column(primary_key=True)
    usuario_id: Mapped[int] = mapped_column(
        ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False
    )
    texto_pregunta: Mapped[str] = mapped_column(Text, nullable=False)
    fecha: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relaciones
    usuario: Mapped["Usuario"] = relationship(  # type: ignore[name-defined]
        "Usuario", back_populates="consultas"
    )
    resultados: Mapped[list["ResultadoConsulta"]] = relationship(
        "ResultadoConsulta", back_populates="consulta", cascade="all, delete-orphan"
    )
    evaluacion: Mapped[Optional["Evaluacion"]] = relationship(
        "Evaluacion", back_populates="consulta", uselist=False,
        cascade="all, delete-orphan"
    )
    embedding_consulta: Mapped[Optional["EmbeddingConsulta"]] = relationship(  # type: ignore[name-defined]
        "EmbeddingConsulta", back_populates="consulta", uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Consulta id={self.id} usuario_id={self.usuario_id}>"


class ResultadoConsulta(Base):
    __tablename__ = "resultados_consulta"

    __table_args__ = (
        CheckConstraint(
            "(embedding_texto_id IS NOT NULL AND embedding_imagen_id IS NULL) OR "
            "(embedding_texto_id IS NULL AND embedding_imagen_id IS NOT NULL)",
            name="chk_exactamente_un_embedding",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    consulta_id: Mapped[int] = mapped_column(
        ForeignKey("consultas.id", ondelete="CASCADE"), nullable=False
    )
    embedding_texto_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("embeddings_texto.id", ondelete="CASCADE")
    )
    embedding_imagen_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("embeddings_imagen.id", ondelete="CASCADE")
    )
    score_similitud: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    posicion: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    # Relaciones
    consulta: Mapped["Consulta"] = relationship("Consulta", back_populates="resultados")
    embedding_texto: Mapped[Optional["EmbeddingTexto"]] = relationship(  # type: ignore[name-defined]
        "EmbeddingTexto", back_populates="resultados",
        foreign_keys=[embedding_texto_id]
    )
    embedding_imagen: Mapped[Optional["EmbeddingImagen"]] = relationship(  # type: ignore[name-defined]
        "EmbeddingImagen", back_populates="resultados",
        foreign_keys=[embedding_imagen_id]
    )

    def __repr__(self) -> str:
        return (
            f"<ResultadoConsulta id={self.id} consulta_id={self.consulta_id} "
            f"posicion={self.posicion} score={self.score_similitud}>"
        )


class Evaluacion(Base):
    __tablename__ = "evaluaciones"

    id: Mapped[int] = mapped_column(primary_key=True)
    consulta_id: Mapped[int] = mapped_column(
        ForeignKey("consultas.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,   # 1:1 con Consultas
    )
    faithfulness: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    answer_relevancy: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    context_recall: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    fecha: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    consulta: Mapped["Consulta"] = relationship("Consulta", back_populates="evaluacion")

    def __repr__(self) -> str:
        return (
            f"<Evaluacion id={self.id} consulta_id={self.consulta_id} "
            f"faithfulness={self.faithfulness}>"
        )
