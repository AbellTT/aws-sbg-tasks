"""
AASTU Student Handbook RAG Assistant - Document Ingestion & Chunking Module
Week 4: Generative AI & AI Applications
"""

import os
import re
import json
from typing import List, Dict, Any
from pathlib import Path
import pypdf

# Default paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "aastu_handbook.pdf"
CHUNKS_PATH = DATA_DIR / "chunks.json"

HEADER_PATTERN = re.compile(r"AASTU\s+Student\s+Handbook\s+2023/2024", re.IGNORECASE)


def clean_text(raw_text: str) -> str:
    """
    Cleans raw PDF text by removing repetitive headers, page numbers,
    and normalizing whitespace while preserving paragraph breaks.
    """
    if not raw_text:
        return ""

    # Remove standard running header
    text = HEADER_PATTERN.sub("", raw_text)

    # Replace multiple horizontal spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines (keep max 2 for paragraph breaks)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from the PDF using pypdf.
    Returns a list of dictionaries with page number and cleaned text.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at expected path: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    extracted_pages = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        raw_text = reader.pages[page_idx].extract_text() or ""
        cleaned = clean_text(raw_text)

        # Skip completely empty or trivial pages (e.g. blank separator pages)
        if len(cleaned) > 40:
            extracted_pages.append({
                "page": page_num,
                "text": cleaned,
                "char_count": len(cleaned)
            })

    return extracted_pages


def chunk_text(
    pages: List[Dict[str, Any]],
    chunk_size_words: int = 350,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Splits page text into overlapping semantic chunks.
    Ensures that chunks carry exact page references for accurate RAG citations.
    """
    chunks = []
    chunk_counter = 0

    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        words = text.split()

        if not words:
            continue

        # If page is short, keep as single chunk
        if len(words) <= chunk_size_words:
            chunk_id = f"aastu_p{page_num:03d}_c00"
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "source": "AASTU Student Handbook 2023/2024",
                "text": text,
                "word_count": len(words),
                "char_count": len(text)
            })
            chunk_counter += 1
            continue

        # Sliding window chunking with overlap
        start = 0
        local_chunk_idx = 0
        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)

            chunk_id = f"aastu_p{page_num:03d}_c{local_chunk_idx:02d}"
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "source": "AASTU Student Handbook 2023/2024",
                "text": chunk_text_str,
                "word_count": len(chunk_words),
                "char_count": len(chunk_text_str)
            })

            local_chunk_idx += 1
            chunk_counter += 1

            if end == len(words):
                break
            start += (chunk_size_words - overlap_words)

    return chunks


def run_ingestion(pdf_path: Path = PDF_PATH, output_json: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """
    Executes the full ingestion pipeline: extracts, cleans, chunks, and caches to JSON.
    """
    print(f"[*] Ingesting PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"[*] Extracted {len(pages)} non-empty pages.")

    print(f"[*] Chunking document (size=350 words, overlap=50 words)...")
    chunks = chunk_text(pages, chunk_size_words=350, overlap_words=50)
    print(f"[*] Generated {len(chunks)} semantic chunks.")

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"[*] Cached chunks to: {output_json}")
    return chunks


def load_or_ingest_chunks(pdf_path: Path = PDF_PATH, cache_path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    """
    Loads chunks from cached JSON if available, otherwise runs ingestion.
    """
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return run_ingestion(pdf_path, cache_path)


if __name__ == "__main__":
    print("=" * 70)
    print("  AASTU STUDENT HANDBOOK: DOCUMENT INGESTION & CHUNKING (PHASE 1)")
    print("=" * 70)

    chunks = run_ingestion()

    total_words = sum(c["word_count"] for c in chunks)
    avg_words = total_words / len(chunks) if chunks else 0

    print("-" * 70)
    print("  INGESTION SUMMARY")
    print("-" * 70)
    print(f"  Total Chunks Created:   {len(chunks):,}")
    print(f"  Total Words Indexed:    {total_words:,}")
    print(f"  Average Chunk Size:     {avg_words:.1f} words")
    print(f"  Page Range Covered:     Page {chunks[0]['page']} -> Page {chunks[-1]['page']}")
    print("-" * 70)
    print("  SAMPLE CHUNK PREVIEW")
    print("-" * 70)
    sample = chunks[5]
    print(f"  Chunk ID:   {sample['chunk_id']}")
    print(f"  Page:       {sample['page']}")
    print(f"  Word Count: {sample['word_count']}")
    print(f"  Preview:    {sample['text'][:280]}...")
    print("=" * 70)
