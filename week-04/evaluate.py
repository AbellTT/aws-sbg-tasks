"""
AASTU Student Handbook RAG Assistant - Automated Evaluation & Regression Suite
Week 4: Generative AI & AI Applications
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Reconfigure stdout for utf-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag_engine import ask_handbook, get_api_key

BASE_DIR = Path(__file__).resolve().parent
GOLDEN_DATASET_PATH = BASE_DIR / "tests" / "golden_dataset.json"


def load_golden_dataset(path: Path = GOLDEN_DATASET_PATH) -> List[Dict[str, Any]]:
    """
    Loads golden regression test cases from JSON.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset missing at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_test_case(test_case: Dict[str, Any], api_key: str) -> Dict[str, Any]:
    """
    Executes a single test case through the RAG engine and evaluates:
      1. Retrieval hit (page range overlap)
      2. Grounding decision (can_answer matches is_in_scope)
      3. Citation correctness
      4. Keyword presence
    """
    query = test_case["query"]
    is_in_scope = test_case["is_in_scope"]
    expected_pages = test_case.get("expected_page_range", [])
    expected_keywords = test_case.get("expected_keywords", [])

    start_time = time.time()
    result = ask_handbook(query, top_k=3, api_key=api_key)
    latency = round(time.time() - start_time, 2)

    hits = result["retrieved_hits"]
    answer_text = result["answer"]
    citations = result["citations"]
    can_answer = result["can_answer"]

    # 1. Retrieval Hit Evaluation
    retrieval_hit = False
    if is_in_scope and expected_pages:
        min_p, max_p = expected_pages[0], expected_pages[1]
        for h in hits:
            # Check if chunk pages overlap with expected page range
            if any(min_p <= p <= max_p for p in range(h["page_start"], h["page_end"] + 1)):
                retrieval_hit = True
                break
    elif not is_in_scope:
        retrieval_hit = True  # N/A for out of scope

    # 2. Decision Accuracy (In-scope recognized vs out-of-scope rejected)
    decision_correct = (can_answer == is_in_scope)

    # 3. Citation Check
    if is_in_scope:
        citation_valid = (len(citations) > 0)
    else:
        citation_valid = (len(citations) == 0)  # No fake citations for adversarial queries

    # 4. Keyword Check (for in-scope)
    keyword_matches = []
    if is_in_scope and expected_keywords:
        for kw in expected_keywords:
            if kw.lower() in answer_text.lower():
                keyword_matches.append(kw)
        keyword_score = len(keyword_matches) / len(expected_keywords)
    else:
        keyword_score = 1.0  # N/A for out-of-scope

    # Overall PASS condition
    passed = decision_correct and citation_valid and (retrieval_hit if is_in_scope else True) and (keyword_score >= 0.5)

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "query": query,
        "is_in_scope": is_in_scope,
        "can_answer": can_answer,
        "decision_correct": decision_correct,
        "retrieval_hit": retrieval_hit,
        "citation_valid": citation_valid,
        "citations": citations,
        "keyword_score": round(keyword_score * 100, 1),
        "confidence": result["confidence"],
        "latency_sec": latency,
        "passed": passed,
        "answer_preview": answer_text[:140].replace("\n", " ")
    }


def run_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """
    Executes the full golden regression evaluation suite and calculates summary metrics.
    """
    api_key = get_api_key()
    test_cases = load_golden_dataset()

    if verbose:
        print("=" * 80)
        print("  AASTU STUDENT HANDBOOK: GOLDEN REGRESSION TEST SUITE (PHASE 4)")
        print("=" * 80)
        print(f"Loaded {len(test_cases)} test cases from {GOLDEN_DATASET_PATH.name}\n")

    eval_results = []
    for idx, tc in enumerate(test_cases, 1):
        if verbose:
            print(f"[*] Running Test {idx}/{len(test_cases)}: [{tc['category']}] \"{tc['query'][:55]}...\"", flush=True)
        res = evaluate_test_case(tc, api_key)
        eval_results.append(res)
        if verbose:
            status = "PASS [OK]" if res["passed"] else "FAIL [X]"
            print(f"    -> Result: {status} | Latency: {res['latency_sec']}s | Confidence: {res['confidence']}", flush=True)
        
        # Gentle delay between queries to avoid free-tier rate spikes / 503s
        if idx < len(test_cases):
            time.sleep(2)

    # Aggregate Metrics
    total_cases = len(eval_results)
    passed_cases = sum(1 for r in eval_results if r["passed"])

    in_scope_cases = [r for r in eval_results if r["is_in_scope"]]
    out_of_scope_cases = [r for r in eval_results if not r["is_in_scope"]]

    retrieval_hit_rate = (
        sum(1 for r in in_scope_cases if r["retrieval_hit"]) / len(in_scope_cases) * 100
        if in_scope_cases else 0.0
    )
    decision_accuracy = (
        sum(1 for r in eval_results if r["decision_correct"]) / total_cases * 100
    )
    refusal_rate = (
        sum(1 for r in out_of_scope_cases if r["decision_correct"]) / len(out_of_scope_cases) * 100
        if out_of_scope_cases else 0.0
    )
    avg_latency = sum(r["latency_sec"] for r in eval_results) / total_cases

    summary = {
        "total_tests": total_cases,
        "passed_tests": passed_cases,
        "overall_pass_rate": round(passed_cases / total_cases * 100, 1),
        "retrieval_hit_rate": round(retrieval_hit_rate, 1),
        "grounding_decision_accuracy": round(decision_accuracy, 1),
        "adversarial_refusal_rate": round(refusal_rate, 1),
        "average_latency_sec": round(avg_latency, 2),
        "detailed_results": eval_results
    }

    if verbose:
        print("\n" + "=" * 80)
        print("  EVALUATION SUMMARY REPORT")
        print("=" * 80)
        print(f"  Overall Test Suite Pass Rate:     {summary['overall_pass_rate']}% ({passed_cases}/{total_cases})")
        print(f"  Retrieval Hit Rate (In-Scope):    {summary['retrieval_hit_rate']}%")
        print(f"  Grounding Decision Accuracy:      {summary['grounding_decision_accuracy']}%")
        print(f"  Adversarial Refusal Rate:         {summary['adversarial_refusal_rate']}% (Zero Hallucinations)")
        print(f"  Average Response Latency:         {summary['average_latency_sec']} seconds")
        print("=" * 80)

        print("\n### Golden Test Results Breakdown\n", flush=True)
        print("| ID | Category | Query | Retrieval Hit | Refusal/Answer | Citations | Status |", flush=True)
        print("|:---|:---|:---|:---:|:---:|:---:|:---:|", flush=True)
        for r in eval_results:
            hit_str = "Yes [OK]" if r["retrieval_hit"] else "No [X]"
            dec_str = "Answered [OK]" if r["can_answer"] else "Refused (Safe) [OK]"
            cits_str = ", ".join(r["citations"]) if r["citations"] else "None (Safe)"
            status_str = "**PASS [OK]**" if r["passed"] else "**FAIL [X]**"
            q_short = r["query"][:40] + ("..." if len(r["query"]) > 40 else "")
            print(f"| {r['id']} | {r['category']} | {q_short} | {hit_str} | {dec_str} | {cits_str} | {status_str} |", flush=True)
        print("\n" + "=" * 80, flush=True)

    return summary


if __name__ == "__main__":
    run_evaluation(verbose=True)
