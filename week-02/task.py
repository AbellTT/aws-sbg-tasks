import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def load_dataset(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        df = pd.read_json(file_path)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported format '{ext}'. Use .json or .csv")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    for col in ["question", "answer", "category", "department"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(["nan", "None", ""], "Unspecified")

    if "combined_text" not in df.columns:
        df["combined_text"] = df["question"] + " " + df["answer"]
    return df


def clean_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = re.sub(r"[^\w\s\-\?\.]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def chunk_text(text: str, size: int = 80, overlap: int = 20) -> List[str]:
    words = text.split()
    if len(words) <= size:
        return [text]

    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if i + size >= len(words):
            break
    return chunks


def prepare_documents(df: pd.DataFrame, chunk_size: int = 80, overlap: int = 20) -> List[Dict]:
    documents = []
    for idx, row in df.iterrows():
        raw_text = f"{row['question']} {row['answer']}"
        chunks = chunk_text(raw_text, size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{row.get('id', f'faq_{idx}')}_c{chunk_idx}"
            documents.append({
                "id": doc_id,
                "text": chunk,
                "question": row["question"],
                "answer": row["answer"],
                "category": row.get("category", "General"),
                "department": row.get("department", "General"),
                "chunk_index": chunk_idx,
            })
    return documents


class FallbackTFIDFEmbedding:
    def __init__(self):
        self.vocabulary: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self.doc_vectors: np.ndarray = np.array([])
        self.docs: List[Dict] = []

    def fit_transform(self, documents: List[Dict]) -> np.ndarray:
        self.docs = documents
        tokenized_docs = [re.findall(r"\b\w+\b", d["text"].lower()) for d in documents]

        unique_tokens = sorted(set(t for doc in tokenized_docs for t in doc))
        self.vocabulary = {token: idx for idx, token in enumerate(unique_tokens)}
        n_docs = len(documents)
        vocab_size = len(unique_tokens)

        doc_freq = np.zeros(vocab_size)
        tf_matrix = np.zeros((n_docs, vocab_size))

        for i, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            for t in tokens:
                if t in self.vocabulary:
                    tf_matrix[i, self.vocabulary[t]] += 1
            unique_in_doc = set(tokens)
            for t in unique_in_doc:
                if t in self.vocabulary:
                    doc_freq[self.vocabulary[t]] += 1

        self.idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1.0
        tfidf = tf_matrix * self.idf

        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.doc_vectors = tfidf / norms
        return self.doc_vectors

    def transform_query(self, query: str) -> np.ndarray:
        tokens = re.findall(r"\b\w+\b", query.lower())
        q_vec = np.zeros(len(self.vocabulary))
        for t in tokens:
            if t in self.vocabulary:
                q_vec[self.vocabulary[t]] += 1
        q_tfidf = q_vec * self.idf
        norm = np.linalg.norm(q_tfidf)
        if norm > 0:
            q_tfidf /= norm
        return q_tfidf

    def query(self, query_text: str, top_k: int = 3, category: Optional[str] = None) -> List[Dict]:
        q_vec = self.transform_query(query_text)
        if np.all(q_vec == 0) or len(self.doc_vectors) == 0:
            return []

        scores = self.doc_vectors @ q_vec
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for idx in ranked_indices:
            doc = self.docs[idx]
            if category and doc.get("category", "").lower() != category.lower():
                continue
            score = float(scores[idx])
            results.append({
                "id": doc["id"],
                "question": doc["question"],
                "answer": doc["answer"],
                "category": doc["category"],
                "department": doc["department"],
                "snippet": doc["text"],
                "score": round(score, 4),
            })
            if len(results) >= top_k:
                break
        return results


class SemanticSearchEngine:
    def __init__(self, use_chroma: bool = True, persist_dir: Optional[str] = None):
        self.use_chroma = use_chroma and CHROMA_AVAILABLE
        self.persist_dir = persist_dir
        self.fallback_engine = FallbackTFIDFEmbedding()
        self.collection = None

        if self.use_chroma:
            if persist_dir:
                self.client = chromadb.PersistentClient(path=persist_dir)
            else:
                self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(
                name="aastu_faqs",
                metadata={"hnsw:space": "cosine"}
            )

    def index_documents(self, documents: List[Dict]):
        if self.use_chroma and self.collection is not None:
            ids = [d["id"] for d in documents]
            texts = [d["text"] for d in documents]
            metadatas = [
                {
                    "question": d["question"][:500],
                    "answer": d["answer"][:500],
                    "category": d["category"],
                    "department": d["department"],
                }
                for d in documents
            ]
            self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        else:
            self.fallback_engine.fit_transform(documents)

    def search(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[Dict]:
        if self.use_chroma and self.collection is not None:
            where_filter = {"category": category} if category else None
            res = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
            matches = []
            if res and res.get("documents") and res["documents"][0]:
                docs = res["documents"][0]
                metas = res["metadatas"][0]
                distances = res["distances"][0] if res.get("distances") else [0.0] * len(docs)
                ids = res["ids"][0]
                for doc_id, snippet, meta, dist in zip(ids, docs, metas, distances):
                    similarity = round(max(0.0, 1.0 - dist), 4)
                    matches.append({
                        "id": doc_id,
                        "question": meta.get("question", ""),
                        "answer": meta.get("answer", ""),
                        "category": meta.get("category", ""),
                        "department": meta.get("department", ""),
                        "snippet": snippet,
                        "score": similarity,
                    })
            return matches
        else:
            return self.fallback_engine.query(query, top_k=top_k, category=category)


def display_results(query: str, results: List[Dict]):
    print("\n" + "=" * 70)
    print(f"QUERY: \"{query}\"")
    print("=" * 70)
    if not results:
        print("No matching FAQ found.")
        return

    for rank, res in enumerate(results, start=1):
        print(f"\n[Match {rank}] Relevance Score: {res['score']}")
        print(f"  Category:   {res['category']} | Department: {res['department']}")
        print(f"  Question:   {res['question']}")
        print(f"  Answer:     {res['answer']}")
        print(f"  Snippet:    {res['snippet'][:120]}...")
    print("\n" + "-" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AASTU Mini Semantic Search Engine")
    parser.add_argument("--data", default="week-02/data/faqs.json", help="Path to FAQ dataset (JSON or CSV)")
    parser.add_argument("--query", type=str, help="Search query string")
    parser.add_argument("--top-k", type=int, default=3, help="Number of nearest matches to retrieve")
    parser.add_argument("--category", type=str, default=None, help="Optional category filter")
    parser.add_argument("--chunk-size", type=int, default=80, help="Words per text chunk")
    parser.add_argument("--overlap", type=int, default=20, help="Word overlap between chunks")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        alt_paths = ["data/faqs.json", "week-02/data/faqs.json", "data/faqs.csv", "week-02/data/faqs.csv"]
        for p in alt_paths:
            if os.path.exists(p):
                data_path = p
                break

    print(f"[*] Loading dataset: {data_path}")
    raw_df = load_dataset(data_path)
    print(f"[*] Raw records: {len(raw_df)}")

    cleaned_df = clean_dataframe(raw_df)
    print(f"[*] Clean records: {len(cleaned_df)}")

    documents = prepare_documents(cleaned_df, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"[*] Generated chunks: {len(documents)}")

    engine_type = "ChromaDB" if CHROMA_AVAILABLE else "NumPy Cosine Engine"
    print(f"[*] Initializing Vector Engine: {engine_type}")
    engine = SemanticSearchEngine(use_chroma=CHROMA_AVAILABLE)
    engine.index_documents(documents)
    print("[*] Indexing complete!")

    if args.query:
        results = engine.search(args.query, top_k=args.top_k, category=args.category)
        display_results(args.query, results)
    elif args.interactive:
        print("\nEntering interactive search mode (type 'exit' to quit):")
        while True:
            try:
                user_q = input("\nSearch FAQ > ").strip()
                if not user_q or user_q.lower() in ["exit", "quit", "q"]:
                    break
                res = engine.search(user_q, top_k=args.top_k)
                display_results(user_q, res)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        sample_queries = [
            "How do I connect to campus wifi?",
            "What happens if my CGPA is below 2.0?",
            "I want to borrow books for the exam",
        ]
        print("\nRunning demonstration queries:")
        for sq in sample_queries:
            results = engine.search(sq, top_k=args.top_k)
            display_results(sq, results)


if __name__ == "__main__":
    main()
