"""
models/recursos.py — Modelos ORM para:
  Recursos, Autores, Recurso_Autores, Generos, Recurso_Generos,
  Tags, Recurso_Tags, Imagenes_Recurso, Colecciones,
  Coleccion_Recursos, Reseñas, Prestamos, Historial_Lectura.
"""

from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    String, Text, Date, DateTime, Boolean, SmallInteger,
    Numeric, Integer, ForeignKey, func, UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


# ─────────────────────────────────────────────────────────────────────────────
# Tablas puente (asociaciones N:M simples)
# ─────────────────────────────────────────────────────────────────────────────

class RecursoAutor(Base):
    __tablename__ = "recurso_autores"

    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), primary_key=True
    )
    autor_id: Mapped[int] = mapped_column(
        ForeignKey("autores.id", ondelete="CASCADE"), primary_key=True
    )
    rol_autor: Mapped[Optional[str]] = mapped_column(String(100))


class RecursoGenero(Base):
    __tablename__ = "recurso_generos"

    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), primary_key=True
    )
    genero_id: Mapped[int] = mapped_column(
        ForeignKey("generos.id", ondelete="CASCADE"), primary_key=True
    )


class RecursoTag(Base):
    __tablename__ = "recurso_tags"

    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), primary_key=True
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True
    )


class ColeccionRecurso(Base):
    __tablename__ = "coleccion_recursos"

    coleccion_id: Mapped[int] = mapped_column(
        ForeignKey("colecciones.id", ondelete="CASCADE"), primary_key=True
    )
    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), primary_key=True
    )
    fecha_agregado: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entidades principales
# ─────────────────────────────────────────────────────────────────────────────

class Recurso(Base):
    __tablename__ = "recursos"

    id: Mapped[int] = mapped_column(primary_key=True)
    titulo: Mapped[str] = mapped_column(String(500), nullable=False)
    fecha_publicacion: Mapped[Optional[date]] = mapped_column(Date)
    idioma: Mapped[Optional[str]] = mapped_column(String(50))
    tipo: Mapped[str] = mapped_column(String(100), nullable=False)
    licencia: Mapped[Optional[str]] = mapped_column(String(100))
    descripcion: Mapped[Optional[str]] = mapped_column(Text)

    # Relaciones
    imagenes: Mapped[list["ImagenRecurso"]] = relationship(
        "ImagenRecurso", back_populates="recurso", cascade="all, delete-orphan"
    )
    resenas: Mapped[list["Resena"]] = relationship(
        "Resena", back_populates="recurso", cascade="all, delete-orphan"
    )
    prestamos: Mapped[list["Prestamo"]] = relationship(
        "Prestamo", back_populates="recurso", cascade="all, delete-orphan"
    )
    embeddings_texto: Mapped[list] = relationship(
        "EmbeddingTexto", back_populates="recurso", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Recurso id={self.id} titulo={self.titulo!r}>"


class Autor(Base):
    __tablename__ = "autores"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(300), nullable=False)
    tipo: Mapped[str] = mapped_column(String(50), nullable=False)  # persona | organizacion
    biografia: Mapped[Optional[str]] = mapped_column(Text)

    def __repr__(self) -> str:
        return f"<Autor id={self.id} nombre={self.nombre!r}>"


class Genero(Base):
    __tablename__ = "generos"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(150), nullable=False, unique=True)
    descripcion: Mapped[Optional[str]] = mapped_column(Text)


class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)


class ImagenRecurso(Base):
    __tablename__ = "imagenes_recurso"

    id: Mapped[int] = mapped_column(primary_key=True)
    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), nullable=False
    )
    ruta_archivo: Mapped[str] = mapped_column(String(1000), nullable=False)
    tipo_imagen: Mapped[Optional[str]] = mapped_column(String(100))
    descripcion: Mapped[Optional[str]] = mapped_column(Text)

    recurso: Mapped["Recurso"] = relationship("Recurso", back_populates="imagenes")
    embedding_imagen: Mapped[Optional["EmbeddingImagen"]] = relationship(  # type: ignore[name-defined]
        "EmbeddingImagen", back_populates="imagen", uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<ImagenRecurso id={self.id} ruta={self.ruta_archivo!r}>"


class Coleccion(Base):
    __tablename__ = "colecciones"

    id: Mapped[int] = mapped_column(primary_key=True)
    usuario_id: Mapped[int] = mapped_column(
        ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False
    )
    nombre: Mapped[str] = mapped_column(String(300), nullable=False)
    descripcion: Mapped[Optional[str]] = mapped_column(Text)
    es_publica: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fecha_creacion: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    usuario: Mapped["Usuario"] = relationship(  # type: ignore[name-defined]
        "Usuario", back_populates="colecciones"
    )


class Resena(Base):
    __tablename__ = "reseñas"

    id: Mapped[int] = mapped_column(primary_key=True)
    usuario_id: Mapped[int] = mapped_column(
        ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False
    )
    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), nullable=False
    )
    calificacion: Mapped[Optional[int]] = mapped_column(SmallInteger)
    texto: Mapped[Optional[str]] = mapped_column(Text)
    fecha: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    usuario: Mapped["Usuario"] = relationship("Usuario", back_populates="resenas")  # type: ignore
    recurso: Mapped["Recurso"] = relationship("Recurso", back_populates="resenas")


class Prestamo(Base):
    __tablename__ = "prestamos"

    id: Mapped[int] = mapped_column(primary_key=True)
    usuario_id: Mapped[int] = mapped_column(
        ForeignKey("usuarios.id", ondelete="CASCADE"), nullable=False
    )
    recurso_id: Mapped[int] = mapped_column(
        ForeignKey("recursos.id", ondelete="CASCADE"), nullable=False
    )
    fecha_inicio: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    fecha_fin_esperada: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    devuelto: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    fecha_devolucion: Mapped[Optional[datetime]] = mapped_column(DateTime)

    usuario: Mapped["Usuario"] = relationship("Usuario", back_populates="prestamos")  # type: ignore
    recurso: Mapped["Recurso"] = relationship("Recurso", back_populates="prestamos")
    historial: Mapped[list["HistorialLectura"]] = relationship(
        "HistorialLectura", back_populates="prestamo", cascade="all, delete-orphan"
    )


class HistorialLectura(Base):
    __tablename__ = "historial_lectura"

    id: Mapped[int] = mapped_column(primary_key=True)
    prestamo_id: Mapped[int] = mapped_column(
        ForeignKey("prestamos.id", ondelete="CASCADE"), nullable=False
    )
    fecha_acceso: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    pagina_hasta: Mapped[Optional[int]] = mapped_column(Integer)
    completado: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    prestamo: Mapped["Prestamo"] = relationship("Prestamo", back_populates="historial")
