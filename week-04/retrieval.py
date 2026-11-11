"""
AASTU Student Handbook RAG Assistant - Vector Store & Semantic Retrieval Module
Week 4: Generative AI & AI Applications
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
CHROMA_DB_PATH = DATA_DIR / "chroma_db"
COLLECTION_NAME = "aastu_handbook"


def get_chroma_client(db_path: Path = CHROMA_DB_PATH) -> chromadb.PersistentClient:
    """
    Initializes a persistent ChromaDB client saving to disk.
    """
    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def get_or_create_collection(
    client: Optional[chromadb.PersistentClient] = None,
    collection_name: str = COLLECTION_NAME
):
    """
    Retrieves or creates the handbook collection configured with cosine distance.
    """
    if client is None:
        client = get_chroma_client()

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )


def index_chunks(
    collection,
    chunks_path: Path = CHUNKS_PATH,
    batch_size: int = 64,
    force_reindex: bool = False
) -> int:
    """
    Indexes the continuous document chunks into ChromaDB with rich metadata.
    Avoids re-indexing if chunks are already present unless force_reindex is True.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found at: {chunks_path}. Please run ingestion.py first."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    current_count = collection.count()
    if current_count == len(chunks) and not force_reindex:
        print(f"[*] Vector store already indexed with all {current_count} chunks. Ready for retrieval.")
        return current_count

    if force_reindex:
        print("[*] Force reindex requested: Clearing existing collection...")
        # Delete and recreate to ensure clean state
        client = get_chroma_client()
        client.delete_collection(collection.name)
        collection = get_or_create_collection(client, collection.name)

    print(f"[*] Indexing {len(chunks)} continuous chunks into ChromaDB...")

    # Index in batches of 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        
        ids = [c["chunk_id"] for c in batch]
        documents = [c["text"] for c in batch]
        
        # ChromaDB metadata must only contain primitive types (str, int, float, bool)
        metadatas = [
            {
                "citation_label": c["citation_label"],
                "page_start": int(c["page_start"]),
                "page_end": int(c["page_end"]),
                "pages_str": ", ".join(map(str, c["pages"])),
                "source": c["source"],
                "word_count": int(c["word_count"])
            }
            for c in batch
        ]

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"    Indexed chunks {i+1} -> {min(i + batch_size, len(chunks))} of {len(chunks)}...")

    print(f"[*] Successfully indexed {collection.count()} chunks in ChromaDB!")
    return collection.count()


def retrieve_context(
    query: str,
    top_k: int = 3,
    collection = None
) -> List[Dict[str, Any]]:
    """
    Executes semantic vector search against the AASTU handbook collection.
    Returns the top_k most relevant chunks with similarity metrics and page citations.
    """
    if collection is None:
        collection = get_or_create_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    if results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for idx, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
            # With cosine space: distance is in [0, 2], where 0 is identical
            # Convert to approximate similarity percentage
            similarity = max(0.0, min(100.0, (1.0 - dist) * 100))
            
            hits.append({
                "rank": idx + 1,
                "chunk_id": meta.get("chunk_id", ""),
                "citation_label": meta.get("citation_label", f"Page {meta.get('page_start', '?')}"),
                "page_start": meta.get("page_start", 0),
                "page_end": meta.get("page_end", 0),
                "pages_str": meta.get("pages_str", ""),
                "source": meta.get("source", "AASTU Student Handbook 2023/2024"),
                "distance": round(float(dist), 4),
                "similarity_pct": round(similarity, 1),
                "text": doc
            })

    return hits


def format_context_for_prompt(hits: List[Dict[str, Any]]) -> str:
    """
    Formats retrieved chunks into a clean, delimited XML block for the RAG prompt.
    Includes explicit source IDs and page citation headers for LLM grounding.
    """
    if not hits:
        return "No relevant excerpts found in the handbook."

    formatted_parts = []
    for hit in hits:
        header = f"[EXCERPT {hit['rank']} | {hit['citation_label']} | Source ID: {hit['chunk_id']}]"
        formatted_parts.append(f"{header}\n{hit['text']}")

    return "\n\n" + ("\n" + "-"*60 + "\n").join(formatted_parts) + "\n"


if __name__ == "__main__":
    print("=" * 75)
    print("  AASTU STUDENT HANDBOOK: VECTOR STORE & SEMANTIC RETRIEVAL (PHASE 2)")
    print("=" * 75)

    client = get_chroma_client()
    collection = get_or_create_collection(client)
    index_chunks(collection)

    # Demonstration queries covering common student handbook inquiries
    test_queries = [
        "What are the rules and criteria for academic probation or dismissal?",
        "How can a student add or drop a course after registration?",
        "What are the disciplinary penalties for cheating or exam malpractice?"
    ]

    print("\n" + "=" * 75)
    print("  SEMANTIC RETRIEVAL TESTBED (TOP-2 HITS PER QUERY)")
    print("=" * 75)

    for q_idx, query in enumerate(test_queries, 1):
        print(f"\n[Query #{q_idx}]: \"{query}\"")
        hits = retrieve_context(query, top_k=2, collection=collection)

        for hit in hits:
            print(f"  Hit #{hit['rank']} -> {hit['citation_label']} (Cosine Distance: {hit['distance']} | Sim: {hit['similarity_pct']}%)")
            preview = hit['text'].replace('\n', ' ')[:160]
            print(f"         Preview: \"{preview}...\"")

    print("\n" + "=" * 75)
    print("  PHASE 2 READY: Vector store indexed and validated.")
    print("=" * 75)

