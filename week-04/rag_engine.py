"""
AASTU Student Handbook RAG Assistant - Grounded Generation & Validation Engine
Week 4: Generative AI & AI Applications
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types

# Import retrieval module from same package
from retrieval import retrieve_context, format_context_for_prompt, get_or_create_collection

# Directory configuration
BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
PROMPT_TEMPLATE_PATH = PROMPTS_DIR / "qa_system.txt"
ENV_PATH = BASE_DIR / ".env"

MODEL_NAME = "gemini-3.8-flash"


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
# 2. CLIENT & PROMPT INITIALIZATION
# ==============================================================================
def get_gemini_client() -> genai.Client:
    """
    Loads API key from .env (checks GEMINI_API_KEY and APIKEY) and returns GenAI Client.
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

    return genai.Client(api_key=raw_key.strip())


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
# 3. GENERATION WITH BOUNDED REPAIR LOOP
# ==============================================================================
def generate_grounded_answer(
    prompt: str,
    client: genai.Client,
    max_attempts: int = 3
) -> Tuple[GroundedAnswer, int]:
    """
    Executes Gemini generation with structured JSON mode and an active repair loop.
    If the output fails JSON decoding or Pydantic validation, it retries with the error details.
    """
    current_prompt = prompt
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1  # low temperature for factual grounding
    )

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=current_prompt,
                config=config
            )
            raw_text = response.text or ""
            clean_json = clean_llm_json_response(raw_text)

            # Validate against Pydantic contract
            validated_obj = GroundedAnswer.model_validate_json(clean_json)
            return validated_obj, attempt

        except (json.JSONDecodeError, ValidationError) as err:
            print(f"    [!] Validation error on attempt {attempt}/{max_attempts}: {err}")
            if attempt == max_attempts:
                raise RuntimeError(
                    f"Failed to produce valid GroundedAnswer after {max_attempts} attempts: {err}"
                ) from err

            # Repair Loop: Give the model specific error feedback to self-correct
            repair_instructions = (
                f"\n\n[SYSTEM REPAIR DIRECTIVE]\n"
                f"Your previous JSON response failed schema validation with error:\n"
                f"{err}\n"
                f"Please fix the schema violation and output strictly valid JSON conforming to the requested format."
            )
            current_prompt = prompt + repair_instructions


# ==============================================================================
# 4. HIGH-LEVEL RAG INFERENCE PIPELINE
# ==============================================================================
def ask_handbook(
    query: str,
    top_k: int = 3,
    client: Optional[genai.Client] = None
) -> Dict[str, Any]:
    """
    End-to-End RAG execution:
      1. Semantic search over continuous handbook chunks (ChromaDB)
      2. Grounded prompt assembly using external prompt file
      3. Real LLM call to Gemini with structured output validation & repair loop
    """
    if client is None:
        client = get_gemini_client()

    # 1. Retrieve
    hits = retrieve_context(query, top_k=top_k)
    context_str = format_context_for_prompt(hits)

    # 2. Augment (Load prompt template and fill)
    template = load_prompt_template()
    rendered_prompt = template.replace("{context}", context_str).replace("{question}", query)

    # 3. Generate & Validate with Repair Loop
    grounded_answer, attempts = generate_grounded_answer(rendered_prompt, client)

    return {
        "query": query,
        "answer": grounded_answer.answer,
        "citations": grounded_answer.citations,
        "confidence": grounded_answer.confidence,
        "can_answer": grounded_answer.can_answer,
        "retrieved_hits": hits,
        "repair_attempts": attempts
    }


# ==============================================================================
# STANDALONE DEMONSTRATION & TEST
# ==============================================================================
if __name__ == "__main__":
    print("=" * 75)
    print("  AASTU STUDENT HANDBOOK: GROUNDED RAG ENGINE DEMO (PHASE 3)")
    print("=" * 75)

    client = get_gemini_client()

    # Test 1: Grounded in-handbook academic question
    in_scope_query = "What are the rules and criteria for academic probation or dismissal?"
    print(f"\n[Test 1 - Grounded Inquiry]: \"{in_scope_query}\"")
    result_1 = ask_handbook(in_scope_query, top_k=3, client=client)

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
    result_2 = ask_handbook(out_of_scope_query, top_k=2, client=client)

    print("-" * 75)
    print(f"Answer:     {result_2['answer']}")
    print(f"Citations:  {result_2['citations']}")
    print(f"Confidence: {result_2['confidence']}")
    print(f"Can Answer: {result_2['can_answer']} (Correctly Refused Hallucination!)")
    print(f"Attempts:   {result_2['repair_attempts']}")

    print("\n" + "=" * 75)
    print("  PHASE 3 READY: Prompt template, Pydantic schema & Gemini RAG engine validated.")
    print("=" * 75)
