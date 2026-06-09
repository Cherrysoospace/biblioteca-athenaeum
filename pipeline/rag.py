"""
pipeline/rag.py — Pipeline RAG completo: recuperación → generación → almacenamiento.

Flujo:
  1. Vectoriza la pregunta con MiniLM.
  2. Recupera top-K chunks híbridos (SQL + vectorial).
  3. Construye el prompt de contexto.
  4. Llama al LLM (Groq o HuggingFace) con el contexto + pregunta.
  5. Persiste la consulta, embedding, resultados y evaluación en la BD.

La función principal es `run_rag(session, usuario_id, pregunta, ...)`.
"""

import json
import logging
import urllib.request
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from core.config import (
    LLM_API_KEY, LLM_API_URL, LLM_MODEL, LLM_PROVIDER, TOP_K
)
from models.consultas import Consulta, ResultadoConsulta, Evaluacion
from models.embeddings_consulta import EmbeddingConsulta
from pipeline.embeddings_minilm import get_embedding
from pipeline.retrieval_texto import buscar_hibrido_texto
from pipeline.evaluacion import calcular_metricas_ragas

logger = logging.getLogger(__name__)

# ── Construcción del prompt ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """Eres el asistente inteligente de la Biblioteca Athenaeum, una \
biblioteca digital de acceso abierto. Tu objetivo es responder preguntas de los usuarios \
utilizando exclusivamente la información de los fragmentos de documentos proporcionados \
como contexto. 

Reglas:
- Responde siempre en el idioma de la pregunta.
- Si la respuesta no se puede inferir del contexto, dilo honestamente.
- Cita el título del recurso cuando sea relevante.
- Sé conciso pero completo."""


def _construir_contexto(chunks: list[dict]) -> str:
    """Formatea los chunks recuperados como bloque de contexto para el prompt."""
    partes = []
    for i, chunk in enumerate(chunks, 1):
        titulo = chunk.get("titulo_recurso", "Recurso desconocido")
        texto = chunk.get("chunk_texto", "")
        score = chunk.get("score", 0)
        partes.append(f"[Fragmento {i} — {titulo} (relevancia: {score:.3f})]\n{texto}")
    return "\n\n---\n\n".join(partes)


def _llamar_llm_groq(pregunta: str, contexto: str) -> str:
    """Llama a la API de Groq (compatible con OpenAI)."""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Contexto:\n{contexto}\n\nPregunta: {pregunta}",
            },
        ],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"].strip()


def _llamar_llm_huggingface(pregunta: str, contexto: str) -> str:
    """Llama a HuggingFace Inference API."""
    prompt = (
        f"{_SYSTEM_PROMPT}\n\nContexto:\n{contexto}\n\nPregunta: {pregunta}\nRespuesta:"
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 800, "temperature": 0.3}}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
    if isinstance(result, list):
        return result[0].get("generated_text", "").replace(prompt, "").strip()
    return result.get("generated_text", str(result))


def _llamar_llm(pregunta: str, contexto: str) -> str:
    """Despacha la llamada al proveedor configurado."""
    try:
        if LLM_PROVIDER == "groq":
            return _llamar_llm_groq(pregunta, contexto)
        else:
            return _llamar_llm_huggingface(pregunta, contexto)
    except Exception as exc:
        logger.error("Error llamando al LLM (%s): %s", LLM_PROVIDER, exc)
        raise


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_rag(
    session: Session,
    usuario_id: int,
    pregunta: str,
    top_k: int = TOP_K,
    filtros: Optional[dict] = None,
    estrategia: Optional[str] = None,
    evaluar: bool = True,
    ground_truth: Optional[str] = None,
) -> dict:
    """
    Ejecuta el pipeline RAG completo.

    Args:
        session: sesión SQLAlchemy.
        usuario_id: ID del usuario que realiza la consulta.
        pregunta: texto de la pregunta en lenguaje natural.
        top_k: número de chunks a recuperar.
        filtros: dict de filtros relacionales (idioma, tipo, calificacion_min, ...).
        estrategia: filtrar por estrategia de chunking específica.
        evaluar: si True, calcula y persiste métricas RAGAS.
        ground_truth: respuesta de referencia para context_recall (opcional).

    Returns:
        dict con keys: consulta_id, pregunta, respuesta, contexto, metricas.
    """
    logger.info("RAG — Iniciando consulta usuario_id=%d", usuario_id)

    # 1. Vectorizar la pregunta
    vector_pregunta = get_embedding(pregunta)

    # 2. Recuperar chunks relevantes
    chunks = buscar_hibrido_texto(
        session, vector_pregunta,
        top_k=top_k, filtros=filtros, estrategia=estrategia
    )
    logger.info("RAG — Recuperados %d chunks", len(chunks))

    # 3. Generar respuesta con el LLM
    if not chunks:
        contexto_str = "No se encontraron documentos relevantes en la biblioteca."
        respuesta = (
            "No encontré fragmentos relevantes en la biblioteca para responder tu pregunta. "
            "Intenta reformularla o ampliar los criterios de búsqueda."
        )
    else:
        contexto_str = _construir_contexto(chunks)
        respuesta = _llamar_llm(pregunta, contexto_str)

    logger.info("RAG — Respuesta generada (%d chars)", len(respuesta))

    # 4. Persistir en la BD: Consulta + EmbeddingConsulta + ResultadoConsulta
    consulta = Consulta(
        usuario_id=usuario_id,
        texto_pregunta=pregunta,
        fecha=datetime.utcnow(),
    )
    session.add(consulta)
    session.flush()  # obtener consulta.id

    emb_consulta = EmbeddingConsulta(
        consulta_id=consulta.id,
        vector_texto_384=vector_pregunta,
    )
    session.add(emb_consulta)

    for posicion, chunk in enumerate(chunks, start=1):
        resultado = ResultadoConsulta(
            consulta_id=consulta.id,
            embedding_texto_id=chunk["id"],
            embedding_imagen_id=None,
            score_similitud=round(float(chunk["score"]), 4),
            posicion=posicion,
        )
        session.add(resultado)

    session.flush()

    # 5. Evaluación RAGAS (opcional)
    metricas = None
    if evaluar:
        try:
            contextos_recuperados = [c["chunk_texto"] for c in chunks]
            metricas = calcular_metricas_ragas(
                pregunta=pregunta,
                respuesta=respuesta,
                contextos=contextos_recuperados,
                ground_truth=ground_truth,
            )
            evaluacion = Evaluacion(
                consulta_id=consulta.id,
                faithfulness=round(metricas["faithfulness"], 4),
                answer_relevancy=round(metricas["answer_relevancy"], 4),
                context_recall=round(metricas.get("context_recall", 0.0), 4),
                fecha=datetime.utcnow(),
            )
            session.add(evaluacion)
            session.flush()
        except Exception as exc:
            logger.warning("Error en evaluación RAGAS: %s", exc)

    return {
        "consulta_id": consulta.id,
        "pregunta": pregunta,
        "respuesta": respuesta,
        "contexto": [
            {
                "texto": c["chunk_texto"],
                "score": float(c["score"]),
                "recurso": c.get("titulo_recurso"),
                "estrategia": c.get("estrategia_chunking"),
            }
            for c in chunks
        ],
        "metricas": metricas,
    }
