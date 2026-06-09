"""
pipeline/evaluacion.py — Evaluación de calidad del pipeline RAG con métricas RAGAS.

Métricas implementadas:
  faithfulness       — ¿La respuesta es fiel al contexto recuperado?
  answer_relevancy   — ¿La respuesta es relevante para la pregunta?
  context_recall     — ¿El contexto recuperado cubre la respuesta esperada?
                       (requiere ground_truth)
  context_precision  — ¿Cuánto del contexto recuperado es realmente útil?
                       (requiere ground_truth)
  answer_correctness — Compara semánticamente la respuesta con el ground truth.
                       (requiere ground_truth)

Estrategia:
  1. Si el paquete `ragas` está instalado, se usa directamente.
  2. Si no está disponible, se usan aproximaciones locales con MiniLM.

Funciones principales:
    calcular_metricas_ragas(pregunta, respuesta, contextos, ground_truth)
        -> dict con todas las métricas

    evaluar_dataset(session, usuario_id, dataset, estrategia)
        -> list[dict] con resultados de cada item del dataset
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from pipeline.embeddings_minilm import get_embedding, similitud_coseno

logger = logging.getLogger(__name__)


# ── Aproximaciones locales (sin ragas) ───────────────────────────────────────

def _faithfulness_local(respuesta: str, contextos: list[str]) -> float:
    if not respuesta or not contextos:
        return 0.0
    vec_respuesta = get_embedding(respuesta)
    scores = []
    for ctx in contextos:
        if ctx:
            vec_ctx = get_embedding(ctx)
            scores.append(similitud_coseno(vec_respuesta, vec_ctx))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _answer_relevancy_local(pregunta: str, respuesta: str) -> float:
    if not pregunta or not respuesta:
        return 0.0
    vec_p = get_embedding(pregunta)
    vec_r = get_embedding(respuesta)
    return float(similitud_coseno(vec_p, vec_r))


def _context_recall_local(contextos: list[str], ground_truth: str) -> float:
    if not contextos or not ground_truth:
        return 0.0
    vec_gt = get_embedding(ground_truth)
    scores = []
    for ctx in contextos:
        if ctx:
            vec_ctx = get_embedding(ctx)
            scores.append(similitud_coseno(vec_gt, vec_ctx))
    return float(max(scores)) if scores else 0.0


def _context_precision_local(
    contextos: list[str],
    ground_truth: str,
    threshold: float = 0.5,
) -> float:
    """
    Proxy de context_precision: evalúa si los chunks mejor rankeados son
    relevantes respecto al ground_truth.

    Para cada chunk se calcula la similitud coseno contra el ground_truth.
    Si supera el umbral, se considera relevante.
    Fórmula: Σ(k=1..K) (Precision@k × rel(k)) / total_relevantes
    """
    if not contextos or not ground_truth:
        return 0.0
    vec_gt = get_embedding(ground_truth)
    relevants = []
    for ctx in contextos:
        if ctx:
            vec_ctx = get_embedding(ctx)
            rel = 1 if similitud_coseno(vec_gt, vec_ctx) >= threshold else 0
        else:
            rel = 0
        relevants.append(rel)

    total_relevant = sum(relevants)
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    relevant_so_far = 0
    for k, rel in enumerate(relevants, start=1):
        if rel:
            relevant_so_far += 1
            precision_at_k = relevant_so_far / k
            precision_sum += precision_at_k

    return float(precision_sum / total_relevant)


def _answer_correctness_local(respuesta: str, ground_truth: str) -> float:
    """
    Proxy de answer_correctness: similitud coseno entre la respuesta
    generada y el ground_truth.
    """
    if not respuesta or not ground_truth:
        return 0.0
    vec_r = get_embedding(respuesta)
    vec_gt = get_embedding(ground_truth)
    return float(similitud_coseno(vec_r, vec_gt))


# ── Evaluación con ragas (si disponible) ─────────────────────────────────────

def _evaluar_con_ragas(
    pregunta: str,
    respuesta: str,
    contextos: list[str],
    ground_truth: Optional[str],
) -> dict:
    """
    Usa el framework RAGAS para calcular métricas.
    Requiere: pip install ragas datasets
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    )

    data = {
        "question": [pregunta],
        "answer": [respuesta],
        "contexts": [contextos],
        "ground_truth": [ground_truth or ""],
    }
    dataset = Dataset.from_dict(data)

    metricas_elegidas = [faithfulness, answer_relevancy]
    if ground_truth:
        metricas_elegidas += [context_recall, context_precision, answer_correctness]

    result = evaluate(dataset=dataset, metrics=metricas_elegidas)
    scores = result.to_pandas().iloc[0].to_dict()

    return {
        "faithfulness": float(scores.get("faithfulness", 0.0)),
        "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
        "context_recall": float(scores.get("context_recall", 0.0)),
        "context_precision": float(scores.get("context_precision", 0.0)),
        "answer_correctness": float(scores.get("answer_correctness", 0.0)),
        "motor": "ragas",
    }


# ── API pública ───────────────────────────────────────────────────────────────

def calcular_metricas_ragas(
    pregunta: str,
    respuesta: str,
    contextos: list[str],
    ground_truth: Optional[str] = None,
) -> dict:
    """
    Calcula las métricas RAGAS de una respuesta generada.

    Intenta usar el paquete `ragas`; si no está disponible, usa
    aproximaciones locales con MiniLM.

    Args:
        pregunta: pregunta original del usuario.
        respuesta: respuesta generada por el LLM.
        contextos: lista de chunks recuperados que sirvieron de contexto.
        ground_truth: respuesta esperada (para métricas que la requieren).

    Returns:
        dict con: faithfulness, answer_relevancy, context_recall,
                  context_precision, answer_correctness, motor.
    """
    try:
        import ragas  # noqa: F401
        import datasets  # noqa: F401
        logger.debug("Usando framework RAGAS para evaluación.")
        return _evaluar_con_ragas(pregunta, respuesta, contextos, ground_truth)
    except ImportError:
        logger.info("ragas/datasets no disponible; usando aproximaciones locales.")

    faith = _faithfulness_local(respuesta, contextos)
    relevancy = _answer_relevancy_local(pregunta, respuesta)

    if ground_truth:
        recall = _context_recall_local(contextos, ground_truth)
        precision = _context_precision_local(contextos, ground_truth)
        correctness = _answer_correctness_local(respuesta, ground_truth)
    else:
        recall = 0.0
        precision = 0.0
        correctness = 0.0

    return {
        "faithfulness": round(faith, 4),
        "answer_relevancy": round(relevancy, 4),
        "context_recall": round(recall, 4),
        "context_precision": round(precision, 4),
        "answer_correctness": round(correctness, 4),
        "motor": "local_minilm",
    }


def evaluar_consulta(session: Session, consulta_id: int) -> dict:
    """
    Recupera una consulta ya almacenada y calcula/actualiza sus métricas RAGAS.
    Útil para re-evaluar consultas históricas.
    """
    from models.consultas import Consulta, ResultadoConsulta, Evaluacion
    from models.embeddings_texto import EmbeddingTexto
    from sqlalchemy import select

    consulta = session.get(Consulta, consulta_id)
    if consulta is None:
        raise ValueError(f"Consulta id={consulta_id} no encontrada.")

    resultados = session.execute(
        select(ResultadoConsulta)
        .where(ResultadoConsulta.consulta_id == consulta_id)
        .order_by(ResultadoConsulta.posicion)
    ).scalars().all()

    contextos = []
    for r in resultados:
        if r.embedding_texto_id:
            emb = session.get(EmbeddingTexto, r.embedding_texto_id)
            if emb:
                contextos.append(emb.chunk_texto)

    metricas = calcular_metricas_ragas(
        pregunta=consulta.texto_pregunta,
        respuesta=" ".join(contextos[:2]),
        contextos=contextos,
        ground_truth=None,
    )

    evaluacion = session.execute(
        select(Evaluacion).where(Evaluacion.consulta_id == consulta_id)
    ).scalar_one_or_none()

    if evaluacion:
        evaluacion.faithfulness = round(metricas["faithfulness"], 4)
        evaluacion.answer_relevancy = round(metricas["answer_relevancy"], 4)
        evaluacion.context_recall = round(metricas["context_recall"], 4)
    else:
        evaluacion = Evaluacion(
            consulta_id=consulta_id,
            faithfulness=round(metricas["faithfulness"], 4),
            answer_relevancy=round(metricas["answer_relevancy"], 4),
            context_recall=round(metricas["context_recall"], 4),
        )
        session.add(evaluacion)

    session.flush()
    logger.info(
        "Evaluación consulta_id=%d: faith=%.4f rel=%.4f recall=%.4f",
        consulta_id,
        metricas["faithfulness"],
        metricas["answer_relevancy"],
        metricas["context_recall"],
    )
    return metricas


# ── Evaluación de dataset completo ────────────────────────────────────────────

def evaluar_dataset(
    session: Session,
    usuario_id: int,
    dataset: list[dict],
    estrategia: Optional[str] = None,
) -> list[dict]:
    """
    Evalúa un dataset de pares (pregunta, ground_truth) ejecutando el pipeline
    RAG completo para cada uno y registrando las métricas.

    Args:
        session: sesión SQLAlchemy.
        usuario_id: ID de usuario para registrar las consultas.
        dataset: lista de dicts con keys: pregunta, ground_truth.
        estrategia: estrategia de chunking a usar (opcional).

    Returns:
        Lista de dicts con pregunta, ground_truth y todas las métricas.
    """
    import time

    from pipeline.rag import run_rag

    resultados = []
    for idx, item in enumerate(dataset):
        pregunta = item["pregunta"]
        ground_truth = item.get("ground_truth")

        if idx > 0:
            time.sleep(1.5)

        try:
            resultado = run_rag(
                session=session,
                usuario_id=usuario_id,
                pregunta=pregunta,
                estrategia=estrategia,
                evaluar=True,
                ground_truth=ground_truth,
            )
            m = resultado.get("metricas") or {}
            resultados.append({
                "pregunta": pregunta,
                "ground_truth": ground_truth,
                "faithfulness": m.get("faithfulness", 0.0),
                "answer_relevancy": m.get("answer_relevancy", 0.0),
                "context_recall": m.get("context_recall", 0.0),
                "context_precision": m.get("context_precision", 0.0),
                "answer_correctness": m.get("answer_correctness", 0.0),
                "consulta_id": resultado["consulta_id"],
            })
        except Exception as exc:
            session.rollback()
            logger.error("Error evaluando '%s...': %s", pregunta[:40], exc)
            resultados.append({
                "pregunta": pregunta,
                "ground_truth": ground_truth,
                "error": str(exc),
            })

    return resultados


# ── Experimento de chunking: evaluar 10 consultas × 3 estrategias ─────────────

def experimento_chunking(
    session: Session,
    usuario_id: int,
    consultas: list[dict],
    estrategias: Optional[list[str]] = None,
) -> list[dict]:
    """
    Ejecuta el experimento comparativo de las 10 consultas de prueba del proyecto
    sobre las 3 estrategias de chunking y registra las métricas RAGAS.
    """
    import time

    from pipeline.rag import run_rag

    if estrategias is None:
        estrategias = ["fixed_size", "sentence_aware", "semantic"]

    resultados = []
    total = len(consultas) * len(estrategias)
    for idx, consulta_cfg in enumerate(consultas):
        pregunta = consulta_cfg["pregunta"]
        ground_truth = consulta_cfg.get("ground_truth")
        filtros = consulta_cfg.get("filtros")

        for est_idx, estrategia in enumerate(estrategias):
            call_num = idx * len(estrategias) + est_idx
            if call_num > 0:
                time.sleep(1.5)

            logger.info(
                "Experimento — pregunta='%s...' estrategia=%s",
                pregunta[:40], estrategia,
            )
            try:
                resultado = run_rag(
                    session=session,
                    usuario_id=usuario_id,
                    pregunta=pregunta,
                    filtros=filtros,
                    estrategia=estrategia,
                    evaluar=True,
                    ground_truth=ground_truth,
                )
                m = resultado.get("metricas") or {}
                resultados.append({
                    "pregunta": pregunta,
                    "estrategia": estrategia,
                    "faithfulness": m.get("faithfulness", 0.0),
                    "answer_relevancy": m.get("answer_relevancy", 0.0),
                    "context_recall": m.get("context_recall", 0.0),
                    "context_precision": m.get("context_precision", 0.0),
                    "answer_correctness": m.get("answer_correctness", 0.0),
                    "consulta_id": resultado["consulta_id"],
                })
            except Exception as exc:
                session.rollback()
                logger.error(
                    "Error en experimento pregunta='%s' estrategia=%s: %s",
                    pregunta[:40], estrategia, exc,
                )
                resultados.append({
                    "pregunta": pregunta,
                    "estrategia": estrategia,
                    "error": str(exc),
                })

    return resultados
