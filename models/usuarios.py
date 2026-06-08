"""
models/usuarios.py — Modelo ORM para la tabla Usuarios.
"""

from datetime import datetime
from sqlalchemy import String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id: Mapped[int] = mapped_column(primary_key=True)
    nombre: Mapped[str] = mapped_column(String(200), nullable=False)
    email: Mapped[str] = mapped_column(String(300), nullable=False, unique=True)
    contrasena_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    rol: Mapped[str] = mapped_column(
        String(50), nullable=False, default="lector"
    )
    fecha_registro: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )

    # Relaciones hacia otras tablas (lazy por defecto)
    colecciones: Mapped[list] = relationship(
        "Coleccion", back_populates="usuario", cascade="all, delete-orphan"
    )
    resenas: Mapped[list] = relationship(
        "Resena", back_populates="usuario", cascade="all, delete-orphan"
    )
    prestamos: Mapped[list] = relationship(
        "Prestamo", back_populates="usuario", cascade="all, delete-orphan"
    )
    consultas: Mapped[list] = relationship(
        "Consulta", back_populates="usuario", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Usuario id={self.id} email={self.email!r} rol={self.rol!r}>"
