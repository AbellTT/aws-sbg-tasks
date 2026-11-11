"""
AASTU Student Handbook RAG Assistant - Interactive CLI & Pipeline Entry Point
Week 4: Generative AI & AI Applications
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from retrieval import get_chroma_client, get_or_create_collection, index_chunks
from rag_engine import ask_handbook, get_api_key
from evaluate import run_evaluation


def print_banner():
    banner = """
================================================================================
   AASTU STUDENT HANDBOOK - PRODUCTION RAG ASSISTANT (CLI)
   Powered by ChromaDB Vector Search + Google Gemini LLM
================================================================================
 Type your question about academic policies, registration, grading, or probation.
 Type 'exit' or 'quit' to close the assistant.
================================================================================
"""
    print(banner)


def display_answer(result: dict, show_sources: bool = True):
    print("\n" + "-" * 75)
    print("ASSISTANT RESPONSE:")
    print("-" * 75)
    print(result["answer"])
    print("-" * 75)

    print(f"Citations:     {', '.join(result['citations']) if result['citations'] else 'None (Out of Scope)'}")
    print(f"Confidence:    {result['confidence']:.2f}")
    print(f"Grounded:      {'Yes' if result['can_answer'] else 'No (Refusal)'}")
    print(f"Model:         {result.get('model', 'gemini-3.1-flash-lite')}")

    if show_sources and result.get("retrieved_hits"):
        print("\nRetrieved Source Chunks:")
        for idx, hit in enumerate(result["retrieved_hits"], 1):
            pages = hit.get("citation_label", f"Pages {hit.get('page_start')}-{hit.get('page_end')}")
            preview = hit.get("text", "")[:120].replace("\n", " ")
            print(f"  [{idx}] {pages} (Distance: {hit.get('distance', 0):.3f})")
            print(f"      \"{preview}...\"")
    print("-" * 75 + "\n")


def interactive_loop(top_k: int = 3, show_sources: bool = True):
    print_banner()
    api_key = get_api_key()

    while True:
        try:
            query = input("Student Query > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting AASTU Assistant. Goodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Exiting AASTU Assistant. Goodbye!")
            break

        print("\n[*] Searching handbook vector index and generating grounded response...")
        try:
            result = ask_handbook(query, top_k=top_k, api_key=api_key)
            display_answer(result, show_sources=show_sources)
        except Exception as e:
            print(f"[!] Error processing request: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="AASTU Student Handbook RAG Assistant - Grounded QA & Evaluation"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query through the RAG engine and exit."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run the Phase 4 Golden Regression Evaluation benchmark."
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=3,
        help="Number of continuous chunks to retrieve (default: 3)."
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force re-indexing of handbook chunks into ChromaDB."
    )
    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Omit retrieved chunk previews from output."
    )

    args = parser.parse_args()

    # Verify or build vector store index
    chroma_client = get_chroma_client()
    collection = get_or_create_collection(chroma_client)
    index_chunks(collection, force_reindex=args.reindex)

    if args.eval:
        run_evaluation(verbose=True)
        return

    if args.query:
        api_key = get_api_key()
        result = ask_handbook(args.query, top_k=args.top_k, api_key=api_key)
        display_answer(result, show_sources=not args.hide_sources)
        return

    # Default: Interactive console mode
    interactive_loop(top_k=args.top_k, show_sources=not args.hide_sources)


if __name__ == "__main__":
    main()
