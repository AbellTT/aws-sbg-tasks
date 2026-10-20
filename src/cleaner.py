import csv
from typing import List, Dict, Any, Optional


def load_raw_data(filepath: str) -> List[Dict[str, str]]:
    with open(filepath, mode="r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_score(val: Any) -> Optional[int]:
    try:
        score = int(float(str(val).strip()))
        return score if 0 <= score <= 100 else None
    except (ValueError, TypeError):
        return None


def clean_row(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
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
    cleaned = []
    for row in records:
        cleaned_row = clean_row(row)
        if cleaned_row is not None:
            cleaned.append(cleaned_row)
    return cleaned
