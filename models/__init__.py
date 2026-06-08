"""models/__init__.py — Importa todos los modelos para que SQLAlchemy los registre correctamente."""

from models.usuarios import Usuario
from models.recursos import (
    Recurso,
    Autor,
    RecursoAutor,
    Genero,
    RecursoGenero,
    Tag,
    RecursoTag,
    ImagenRecurso,
    Coleccion,
    ColeccionRecurso,
    Resena,
    Prestamo,
    HistorialLectura,
)
from models.embeddings_texto import EmbeddingTexto
from models.embeddings_imagen import EmbeddingImagen
from models.embeddings_consulta import EmbeddingConsulta
from models.consultas import Consulta, ResultadoConsulta, Evaluacion

__all__ = [
    "Usuario",
    "Recurso",
    "Autor",
    "RecursoAutor",
    "Genero",
    "RecursoGenero",
    "Tag",
    "RecursoTag",
    "ImagenRecurso",
    "Coleccion",
    "ColeccionRecurso",
    "Resena",
    "Prestamo",
    "HistorialLectura",
    "EmbeddingTexto",
    "EmbeddingImagen",
    "EmbeddingConsulta",
    "Consulta",
    "ResultadoConsulta",
    "Evaluacion",
]
