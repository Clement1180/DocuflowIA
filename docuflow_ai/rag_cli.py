"""Interface CLI pour le RAG DocuFlow AI.

Usage :
    python -m docuflow_ai.rag_cli index <fichier_ou_dossier>
    python -m docuflow_ai.rag_cli query "Quelle est la date de la facture ?"
    python -m docuflow_ai.rag_cli status
    python -m docuflow_ai.rag_cli clear
"""

import argparse
import sys
from pathlib import Path

from docuflow_ai.modules.ocr_engine import process_document
from docuflow_ai.modules.rag_engine import RAGEngine, DocumentStore
from docuflow_ai.modules.entity_extractor import extract_entities


def cmd_index(args):
    engine = RAGEngine()
    path = Path(args.path)

    files = []
    if path.is_dir():
        for ext in ("*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tiff", "*.bmp"):
            files.extend(path.glob(ext))
    elif path.is_file():
        files = [path]
    else:
        print(f"Erreur : '{path}' introuvable.")
        sys.exit(1)

    print(f"Indexation de {len(files)} document(s)...")

    for f in files:
        print(f"  → OCR de {f.name}...", end=" ", flush=True)
        try:
            result = process_document(str(f))
            doc_id = engine.index_document(
                text=result["text"],
                source=f.name,
                metadata={
                    "confidence": result["confidence"],
                    "word_count": result["word_count"],
                    "path": str(f),
                },
            )
            print(f"OK (id={doc_id[:8]}, {result['word_count']} mots, conf={result['confidence']}%)")
        except Exception as e:
            print(f"ERREUR : {e}")

    print(f"\nTotal : {engine.store.get_document_count()} document(s) indexé(s).")


def cmd_query(args):
    engine = RAGEngine()

    if engine.store.get_document_count() == 0:
        print("Aucun document indexé. Utilisez 'index' d'abord.")
        sys.exit(1)

    result = engine.query(args.question, top_k=args.top_k)

    print(f"Question : {args.question}")
    print(f"\nRéponse :\n{result['answer']}")
    print(f"\nSources ({result['chunks_used']} passages) : {', '.join(result['sources'])}")


def cmd_status(args):
    store = DocumentStore()
    count = store.get_document_count()
    sources = store.get_all_sources()

    print(f"Documents indexés : {count}")
    if sources:
        print("Sources :")
        for s in sources:
            print(f"  - {s}")


def cmd_clear(args):
    store = DocumentStore()
    store.clear()
    print("Index vidé.")


def main():
    parser = argparse.ArgumentParser(description="DocuFlow AI - RAG CLI")
    sub = parser.add_subparsers(dest="command")

    p_index = sub.add_parser("index", help="Indexer un fichier ou dossier")
    p_index.add_argument("path", help="Chemin du fichier ou dossier à indexer")

    p_query = sub.add_parser("query", help="Poser une question sur les documents")
    p_query.add_argument("question", help="La question en langage naturel")
    p_query.add_argument("--top-k", type=int, default=5, help="Nombre de passages à récupérer")

    sub.add_parser("status", help="Afficher l'état de l'index")
    sub.add_parser("clear", help="Vider l'index")

    args = parser.parse_args()

    commands = {"index": cmd_index, "query": cmd_query, "status": cmd_status, "clear": cmd_clear}
    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
