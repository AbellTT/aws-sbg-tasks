"""
AASTU Student Handbook RAG Assistant - Document Ingestion & Chunking Module
Week 4: Generative AI & AI Applications
"""

import os
import re
import json
from typing import List, Dict, Any, Tuple
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


def chunk_document_continuous(
    pages: List[Dict[str, Any]],
    chunk_size_words: int = 350,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Document-based continuous chunker with cross-page overlap.
    
    Instead of isolating pages, this combines all text into one continuous
    stream of words while tagging each word with its originating page number.
    This guarantees:
      1. No sentences or rules are abruptly severed at page boundaries.
      2. True cross-page overlap (the end of Page N blends into Page N+1).
      3. Precise multi-page citation tracking (e.g., 'Pages 45–46').
    """
    # 1. Build a global token stream tagged with originating page numbers
    tagged_words: List[Tuple[str, int]] = []
    for page_data in pages:
        p_num = page_data["page"]
        p_words = page_data["text"].split()
        for w in p_words:
            tagged_words.append((w, p_num))

    if not tagged_words:
        return []

    chunks = []
    start = 0
    step = chunk_size_words - overlap_words
    chunk_idx = 0

    # 2. Slide window continuously across document boundaries
    while start < len(tagged_words):
        end = min(start + chunk_size_words, len(tagged_words))
        window = tagged_words[start:end]

        chunk_words = [w for w, _ in window]
        chunk_text_str = " ".join(chunk_words)

        # Identify all pages spanning this chunk
        pages_in_chunk = sorted(list(set(p for _, p in window)))
        page_start = pages_in_chunk[0]
        page_end = pages_in_chunk[-1]

        if page_start == page_end:
            citation_label = f"Page {page_start}"
        else:
            citation_label = f"Pages {page_start}–{page_end}"

        chunk_id = f"aastu_c{chunk_idx:04d}_p{page_start:03d}"

        chunks.append({
            "chunk_id": chunk_id,
            "page_start": page_start,
            "page_end": page_end,
            "pages": pages_in_chunk,
            "citation_label": citation_label,
            "source": "AASTU Student Handbook 2023/2024",
            "text": chunk_text_str,
            "word_count": len(chunk_words),
            "char_count": len(chunk_text_str)
        })

        chunk_idx += 1

        if end == len(tagged_words):
            break

        start += step

    return chunks


def run_ingestion(
    pdf_path: Path = PDF_PATH,
    output_json: Path = CHUNKS_PATH,
    chunk_size_words: int = 350,
    overlap_words: int = 50
) -> List[Dict[str, Any]]:
    """
    Executes the full continuous ingestion pipeline:
    extracts, cleans, continuously chunks across page boundaries, and caches to JSON.
    """
    print(f"[*] Ingesting PDF: {pdf_path}")
    pages = extract_pages(pdf_path)
    print(f"[*] Extracted {len(pages)} non-empty pages.")

    print(f"[*] Applying continuous document chunking (size={chunk_size_words} words, overlap={overlap_words} words)...")
    chunks = chunk_document_continuous(pages, chunk_size_words=chunk_size_words, overlap_words=overlap_words)
    print(f"[*] Generated {len(chunks)} continuous semantic chunks.")

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
    print("  AASTU STUDENT HANDBOOK: CONTINUOUS DOCUMENT CHUNKING (FIX)")
    print("=" * 70)

    chunks = run_ingestion()

    total_words = sum(c["word_count"] for c in chunks)
    avg_words = total_words / len(chunks) if chunks else 0
    multi_page_chunks = [c for c in chunks if c["page_start"] != c["page_end"]]

    print("-" * 70)
    print("  INGESTION SUMMARY")
    print("-" * 70)
    print(f"  Total Chunks Created:      {len(chunks):,}")
    print(f"  Total Words Indexed:       {total_words:,}")
    print(f"  Average Chunk Size:        {avg_words:.1f} words")
    print(f"  Page Range Covered:        Page {chunks[0]['page_start']} -> Page {chunks[-1]['page_end']}")
    print(f"  Cross-Page Spanning Chunks:{len(multi_page_chunks):,} ({len(multi_page_chunks)/len(chunks)*100:.1f}%)")
    print("-" * 70)
    print("  SAMPLE CROSS-PAGE CHUNK PREVIEW")
    print("-" * 70)
    # Pick a sample that spans two pages to demonstrate the fix!
    sample = multi_page_chunks[0] if multi_page_chunks else chunks[0]
    print(f"  Chunk ID:       {sample['chunk_id']}")
    print(f"  Citation Label: {sample['citation_label']} (Spans: {sample['pages']})")
    print(f"  Word Count:     {sample['word_count']}")
    print(f"  Preview:        {sample['text'][:280]}...")
    print("=" * 70)
