"""
models/embeddings_imagen.py — Modelo ORM para Embeddings_Imagen.

Almacena los vectores visuales generados por CLIP ViT-B/32 (512 dims).
Relación 1:1 con Imagenes_Recurso (imagen_id UNIQUE).
"""

from sqlalchemy import String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from core.database import Base
from core.config import EMBEDDING_DIM_IMAGEN, CLIP_MODEL


class EmbeddingImagen(Base):
    __tablename__ = "embeddings_imagen"

    id: Mapped[int] = mapped_column(primary_key=True)
    imagen_id: Mapped[int] = mapped_column(
        ForeignKey("imagenes_recurso.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,   # relación 1:1
    )
    # Vector de 512 dimensiones (CLIP ViT-B/32)
    vector_embedding: Mapped[list] = mapped_column(
        Vector(EMBEDDING_DIM_IMAGEN), nullable=False
    )
    modelo: Mapped[str] = mapped_column(
        String(200), nullable=False, default=CLIP_MODEL
    )

    # Relaciones
    imagen: Mapped["ImagenRecurso"] = relationship(  # type: ignore[name-defined]
        "ImagenRecurso", back_populates="embedding_imagen"
    )
    resultados: Mapped[list] = relationship(
        "ResultadoConsulta", back_populates="embedding_imagen",
        foreign_keys="ResultadoConsulta.embedding_imagen_id"
    )

    def __repr__(self) -> str:
        return f"<EmbeddingImagen id={self.id} imagen_id={self.imagen_id}>"
