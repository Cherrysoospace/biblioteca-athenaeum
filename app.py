"""
Biblioteca Athenaeum — Punto de entrada principal
Sistema híbrido relacional-vectorial (RAG)
"""

import argparse
import sys
from core.database import init_db, get_session
from pipeline.ingest import ingest_recurso
from pipeline.rag import run_rag
from pipeline.evaluacion import evaluar_consulta


def cmd_ingest(args):
    """Ingesta completa de un recurso: chunking + embeddings + almacenamiento."""
    with get_session() as session:
        result = ingest_recurso(
            session=session,
            recurso_id=args.recurso_id,
            estrategias=args.estrategias or ["fixed_size", "sentence_aware", "semantic"],
            vectorizar_imagenes=not args.sin_imagenes,
        )
    print(f"[ingest] Recurso {args.recurso_id} procesado:")
    print(f"  Chunks generados : {result['chunks_total']}")
    print(f"  Embeddings texto : {result['embeddings_texto']}")
    print(f"  Embeddings imagen: {result['embeddings_imagen']}")


def cmd_query(args):
    """Ejecuta una consulta RAG y muestra la respuesta."""
    with get_session() as session:
        resultado = run_rag(
            session=session,
            usuario_id=args.usuario_id,
            pregunta=args.pregunta,
            top_k=args.top_k,
            filtros={
                "idioma": args.idioma,
                "tipo": args.tipo,
                "calificacion_min": args.calificacion_min,
            },
        )
    print("\n=== RESPUESTA RAG ===")
    print(resultado["respuesta"])
    print("\n--- Contexto recuperado ---")
    for i, chunk in enumerate(resultado["contexto"], 1):
        print(f"[{i}] (score={chunk['score']:.4f}) {chunk['texto'][:120]}...")
    if resultado.get("metricas"):
        m = resultado["metricas"]
        print(f"\nMétricas RAGAS — faithfulness={m['faithfulness']:.4f}  "
              f"answer_relevancy={m['answer_relevancy']:.4f}  "
              f"context_recall={m['context_recall']:.4f}")


def cmd_evaluar(args):
    """Evalúa una consulta ya registrada y guarda métricas en Evaluaciones."""
    with get_session() as session:
        metricas = evaluar_consulta(session=session, consulta_id=args.consulta_id)
    print(f"Evaluación consulta {args.consulta_id}:")
    print(f"  faithfulness     = {metricas['faithfulness']:.4f}")
    print(f"  answer_relevancy = {metricas['answer_relevancy']:.4f}")
    print(f"  context_recall   = {metricas['context_recall']:.4f}")


def cmd_initdb(args):
    """Inicializa tablas y extensión pgvector (idempotente)."""
    init_db()
    print("[initdb] Base de datos inicializada correctamente.")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="athenaeum",
        description="Biblioteca Athenaeum — CLI del sistema RAG híbrido",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    # initdb
    sub.add_parser("initdb", help="Inicializa la base de datos")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingesta un recurso completo")
    p_ingest.add_argument("recurso_id", type=int, help="ID del recurso en Recursos")
    p_ingest.add_argument(
        "--estrategias", nargs="+",
        choices=["fixed_size", "sentence_aware", "semantic"],
        help="Estrategias de chunking a aplicar (default: las tres)",
    )
    p_ingest.add_argument(
        "--sin-imagenes", action="store_true",
        help="Omite la vectorización de imágenes asociadas",
    )

    # query
    p_query = sub.add_parser("query", help="Lanza una consulta RAG")
    p_query.add_argument("pregunta", type=str)
    p_query.add_argument("--usuario-id", type=int, default=1)
    p_query.add_argument("--top-k", type=int, default=5)
    p_query.add_argument("--idioma", type=str, default=None)
    p_query.add_argument("--tipo", type=str, default=None)
    p_query.add_argument("--calificacion-min", type=float, default=None)

    # evaluar
    p_eval = sub.add_parser("evaluar", help="Evalúa una consulta con RAGAS")
    p_eval.add_argument("consulta_id", type=int)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "initdb": cmd_initdb,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "evaluar": cmd_evaluar,
    }
    handlers[args.comando](args)
