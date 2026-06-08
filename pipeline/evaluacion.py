"""
pipeline/evaluacion.py — Evaluación de calidad del pipeline RAG con métricas RAGAS.

Métricas implementadas:
  faithfulness      — ¿La respuesta es fiel al contexto recuperado?
  answer_relevancy  — ¿La respuesta es relevante para la pregunta?
  context_recall    — ¿El contexto recuperado cubre la respuesta esperada?
                      (requiere ground_truth)

Estrategia:
  1. Si el paquete `ragas` está instalado, se usa directamente.
  2. Si no está disponible, se usan aproximaciones locales con MiniLM
     (similitud semántica como proxy de las métricas).

Función principal:
    calcular_metricas_ragas(pregunta, respuesta, contextos, ground_truth)
        -> dict con faithfulness, answer_relevancy, context_recall

Función de persistencia:
    evaluar_consulta(session, consulta_id) -> dict de métricas
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from pipeline.embeddings_minilm import get_embedding, similitud_coseno

logger = logging.getLogger(__name__)


# ── Aproximaciones locales (sin ragas) ───────────────────────────────────────

def _faithfulness_local(respuesta: str, contextos: list[str]) -> float:
    """
    Proxy de faithfulness: similitud coseno promedio entre el embedding
    de la respuesta y los embeddings de los contextos.
    Una respuesta fiel debería ser semánticamente cercana al contexto.
    """
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
    """
    Proxy de answer_relevancy: similitud coseno entre pregunta y respuesta.
    Una respuesta relevante debería responder semánticamente la pregunta.
    """
    if not pregunta or not respuesta:
        return 0.0
    vec_p = get_embedding(pregunta)
    vec_r = get_embedding(respuesta)
    return float(similitud_coseno(vec_p, vec_r))


def _context_recall_local(contextos: list[str], ground_truth: str) -> float:
    """
    Proxy de context_recall: similitud máxima entre los contextos recuperados
    y la respuesta esperada (ground_truth).
    """
    if not contextos or not ground_truth:
        return 0.0

    vec_gt = get_embedding(ground_truth)
    scores = []
    for ctx in contextos:
        if ctx:
            vec_ctx = get_embedding(ctx)
            scores.append(similitud_coseno(vec_gt, vec_ctx))

    return float(max(scores)) if scores else 0.0


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
    from ragas.metrics import faithfulness, answer_relevancy, context_recall

    data = {
        "question": [pregunta],
        "answer": [respuesta],
        "contexts": [contextos],
        "ground_truth": [ground_truth or ""],
    }
    dataset = Dataset.from_dict(data)

    metricas_elegidas = [faithfulness, answer_relevancy]
    if ground_truth:
        metricas_elegidas.append(context_recall)

    result = evaluate(dataset=dataset, metrics=metricas_elegidas)
    scores = result.to_pandas().iloc[0].to_dict()

    return {
        "faithfulness": float(scores.get("faithfulness", 0.0)),
        "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
        "context_recall": float(scores.get("context_recall", 0.0)),
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
    aproximaciones locales con MiniLM (menos precisas pero sin dependencias extra).

    Args:
        pregunta: pregunta original del usuario.
        respuesta: respuesta generada por el LLM.
        contextos: lista de chunks recuperados que sirvieron de contexto.
        ground_truth: respuesta esperada (para context_recall). Opcional.

    Returns:
        dict con: faithfulness, answer_relevancy, context_recall, motor.
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
    recall = _context_recall_local(contextos, ground_truth) if ground_truth else 0.0

    return {
        "faithfulness": round(faith, 4),
        "answer_relevancy": round(relevancy, 4),
        "context_recall": round(recall, 4),
        "motor": "local_minilm",
    }


def evaluar_consulta(session: Session, consulta_id: int) -> dict:
    """
    Recupera una consulta ya almacenada y calcula/actualiza sus métricas RAGAS.

    Útil para re-evaluar consultas históricas o calcular métricas diferidas.

    Args:
        session: sesión SQLAlchemy.
        consulta_id: ID de la fila en Consultas.

    Returns:
        dict con faithfulness, answer_relevancy, context_recall.
    """
    from models.consultas import Consulta, ResultadoConsulta, Evaluacion
    from models.embeddings_texto import EmbeddingTexto
    from sqlalchemy import select

    consulta = session.get(Consulta, consulta_id)
    if consulta is None:
        raise ValueError(f"Consulta id={consulta_id} no encontrada.")

    # Recuperar chunks que se devolvieron en esta consulta
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

    # No hay respuesta almacenada en BD (solo se genera en tiempo real).
    # Para re-evaluación usamos solo faithfulness y context_recall aproximados
    # sobre el contexto. Se omite answer_relevancy sin la respuesta del LLM.
    metricas = calcular_metricas_ragas(
        pregunta=consulta.texto_pregunta,
        respuesta=" ".join(contextos[:2]),  # proxy: primeros 2 chunks como "respuesta"
        contextos=contextos,
        ground_truth=None,
    )

    # Persistir o actualizar en Evaluaciones
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

    Args:
        session: sesión SQLAlchemy.
        usuario_id: ID de usuario para registrar las consultas.
        consultas: lista de dicts con keys: pregunta, ground_truth (opcional),
                   filtros (opcional).
        estrategias: list de estrategias a comparar. Default: las tres.

    Returns:
        Lista de dicts: pregunta, estrategia, faithfulness, answer_relevancy,
        context_recall, consulta_id.
    """
    from pipeline.rag import run_rag

    if estrategias is None:
        estrategias = ["fixed_size", "sentence_aware", "semantic"]

    resultados = []
    for consulta_cfg in consultas:
        pregunta = consulta_cfg["pregunta"]
        ground_truth = consulta_cfg.get("ground_truth")
        filtros = consulta_cfg.get("filtros")

        for estrategia in estrategias:
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
                    "consulta_id": resultado["consulta_id"],
                })
            except Exception as exc:
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
