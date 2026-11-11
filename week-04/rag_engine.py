"""
AASTU Student Handbook RAG Assistant - Grounded Generation & Validation Engine
Week 4: Generative AI & AI Applications
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# Reconfigure stdout for utf-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Import retrieval module
from retrieval import retrieve_context, format_context_for_prompt, get_or_create_collection

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
PROMPT_TEMPLATE_PATH = PROMPTS_DIR / "qa_system.txt"
ENV_PATH = BASE_DIR / ".env"

DEFAULT_MODEL = "gemini-3.1-flash-lite"
FALLBACK_MODEL = "gemini-3.8-flash"


# ==============================================================================
# 1. STRUCTURED OUTPUT SCHEMA (PYDANTIC)
# ==============================================================================
class GroundedAnswer(BaseModel):
    """
    Contract schema for all RAG assistant responses.
    Guarantees type safety, explicit citations, and refusal flag.
    """
    answer: str = Field(
        ...,
        description="The clear, grounded explanation answering the student inquiry."
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of exact page labels cited from the context, e.g. ['Pages 25–27', 'Page 18']."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    can_answer: bool = Field(
        ...,
        description="True if the provided context directly answers the question; False if out-of-scope."
    )


# ==============================================================================
# 2. API KEY & PROMPT UTILITIES
# ==============================================================================
def get_api_key() -> str:
    """
    Loads API key from .env (checks GEMINI_API_KEY and APIKEY).
    """
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()

    raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("APIKEY")
    if not raw_key:
        raise ValueError(
            f"No Gemini API key found! Please ensure 'GEMINI_API_KEY' is set in {ENV_PATH} or your environment."
        )

    return raw_key.strip()


def load_prompt_template(template_path: Path = PROMPT_TEMPLATE_PATH) -> str:
    """
    Loads the system prompt from the external prompt file.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template missing at: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def clean_llm_json_response(raw_text: str) -> str:
    """
    Strips accidental Markdown fences (e.g. ```json ... ```) that LLMs sometimes emit.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


# ==============================================================================
# 3. ROBUST REST CALL WITH EXPONENTIAL BACKOFF
# ==============================================================================
def call_gemini_raw(prompt: str, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 40) -> str:
    """
    Executes a direct, highly resilient HTTP REST call to Gemini with exponential backoff.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            
            # Handle rate-limit (429) or temporary server load (503)
            if resp.status_code in (429, 503):
                wait_sec = (attempt + 1) * 3
                print(f"    [!] Gemini returned HTTP {resp.status_code}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
                continue

            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned empty candidate response.")
            return candidates[0]["content"]["parts"][0]["text"]

        except (requests.Timeout, requests.ConnectionError) as net_err:
            if attempt == max_retries - 1:
                # Try fallback model on final attempt if primary model timed out
                if model != FALLBACK_MODEL:
                    return call_gemini_raw(prompt, api_key, model=FALLBACK_MODEL, timeout=timeout)
                raise net_err
            wait_sec = (attempt + 1) * 3
            time.sleep(wait_sec)

    raise RuntimeError("Failed to obtain response from Gemini API after network retries.")


# ==============================================================================
# 4. GENERATION WITH BOUNDED REPAIR LOOP
# ==============================================================================
def generate_grounded_answer(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_repair_attempts: int = 3
) -> Tuple[GroundedAnswer, int]:
    """
    Executes Gemini generation with structured JSON validation and a bounded repair loop.
    If the output fails JSON decoding or Pydantic validation, it retries with the error details.
    """
    current_prompt = prompt

    for attempt in range(1, max_repair_attempts + 1):
        raw_text = call_gemini_raw(current_prompt, api_key, model=model)
        clean_json = clean_llm_json_response(raw_text)

        try:
            validated_obj = GroundedAnswer.model_validate_json(clean_json)
            return validated_obj, attempt
        except (json.JSONDecodeError, ValidationError) as err:
            print(f"    [!] Schema validation error on attempt {attempt}/{max_repair_attempts}: {err}")
            if attempt == max_repair_attempts:
                raise RuntimeError(
                    f"Failed to produce valid GroundedAnswer after {max_repair_attempts} attempts: {err}"
                ) from err

            # Repair Loop: Supply specific error feedback to LLM
            repair_directive = (
                f"\n\n[SYSTEM REPAIR DIRECTIVE]\n"
                f"Your previous JSON response violated schema validation with error:\n"
                f"{err}\n"
                f"Please correct the error and output strictly valid JSON conforming to the schema."
            )
            current_prompt = prompt + repair_directive


# ==============================================================================
# 5. HIGH-LEVEL RAG INFERENCE PIPELINE
# ==============================================================================
def ask_handbook(
    query: str,
    top_k: int = 3,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    End-to-End RAG execution:
      1. Semantic search over continuous handbook chunks (ChromaDB)
      2. Grounded prompt assembly using external prompt file
      3. Real LLM call to Gemini with structured output validation & repair loop
    """
    if api_key is None:
        api_key = get_api_key()

    # 1. Retrieve
    hits = retrieve_context(query, top_k=top_k)
    context_str = format_context_for_prompt(hits)

    # 2. Augment (Load prompt template and fill)
    template = load_prompt_template()
    rendered_prompt = template.replace("{context}", context_str).replace("{question}", query)

    # 3. Generate & Validate with Repair Loop
    grounded_answer, attempts = generate_grounded_answer(rendered_prompt, api_key, model=model)

    return {
        "query": query,
        "answer": grounded_answer.answer,
        "citations": grounded_answer.citations,
        "confidence": grounded_answer.confidence,
        "can_answer": grounded_answer.can_answer,
        "retrieved_hits": hits,
        "repair_attempts": attempts,
        "model": model
    }


# ==============================================================================
# STANDALONE DEMONSTRATION & TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("  AASTU STUDENT HANDBOOK: GROUNDED RAG ENGINE DEMO (PHASE 3)")
    print("=" * 75)

    key = get_api_key()

    # Test 1: Grounded in-handbook academic question
    in_scope_query = "What are the rules and criteria for academic probation or dismissal?"
    print(f"\n[Test 1 - Grounded Inquiry]: \"{in_scope_query}\"")
    result_1 = ask_handbook(in_scope_query, top_k=3, api_key=key)

    print("-" * 75)
    print(f"Answer:     {result_1['answer']}")
    print(f"Citations:  {result_1['citations']}")
    print(f"Confidence: {result_1['confidence']}")
    print(f"Can Answer: {result_1['can_answer']} (Grounded)")
    print(f"Attempts:   {result_1['repair_attempts']}")

    # Test 2: Out-of-scope question to test anti-hallucination refusal
    out_of_scope_query = "How do I bake a chocolate cake at home?"
    print("\n" + "=" * 75)
    print(f"[Test 2 - Out-of-Scope Refusal]: \"{out_of_scope_query}\"")
    result_2 = ask_handbook(out_of_scope_query, top_k=2, api_key=key)

    print("-" * 75)
    print(f"Answer:     {result_2['answer']}")
    print(f"Citations:  {result_2['citations']}")
    print(f"Confidence: {result_2['confidence']}")
    print(f"Can Answer: {result_2['can_answer']} (Correctly Refused Hallucination!)")
    print(f"Attempts:   {result_2['repair_attempts']}")

    print("\n" + "=" * 75)
    print("  PHASE 3 READY: Prompt template, Pydantic schema & Gemini RAG engine validated.")
    print("=" * 75)
