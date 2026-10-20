"""
Data cleaning and validation module for student performance records.
"""
import csv
from typing import List, Dict, Any, Optional


def load_raw_data(filepath: str) -> List[Dict[str, str]]:
    """Loads a CSV file and returns rows as dictionaries."""
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def parse_score(val: Any) -> Optional[int]:
    """Safely converts a score string to an integer between 0 and 100.
    Returns None if missing, non-numeric, or outside valid range.
    """
    try:
        score = int(float(str(val).strip()))
        if 0 <= score <= 100:
            return score
        return None
    except (ValueError, TypeError):
        return None


def clean_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Cleans, normalizes, and validates a single student record."""
    normalized_row = {k.strip().lower(): v.strip() for k, v in row.items()}

    math = parse_score(normalized_row.get("math score"))
    reading = parse_score(normalized_row.get("reading score"))
    writing = parse_score(normalized_row.get("writing score"))

    if math is None or reading is None or writing is None:
        return None

    return {
        "gender": normalized_row.get("gender", "unknown").lower(),
        "race_ethnicity": normalized_row.get("race/ethnicity", "unknown").title(),
        "parental_education": normalized_row.get("parental level of education", "unknown"),
        "lunch": normalized_row.get("lunch", "unknown").lower(),
        "test_prep_completed": normalized_row.get("test preparation course", "").lower() == "completed",
        "math_score": math,
        "reading_score": reading,
        "writing_score": writing,
    }


def clean_dataset(records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Applies cleaning to all records and returns a list of valid records."""
    cleaned = []
    dropped_count = 0
    for row in records:
        cleaned_row = clean_row(row)
        if cleaned_row is not None:
            cleaned.append(cleaned_row)
        else:
            dropped_count += 1
    return cleaned
