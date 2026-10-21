import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

SUBJECTS = ["Math", "Reading", "Writing"]


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
    normalized = {k.strip().lower(): v.strip() for k, v in row.items()}

    math = parse_score(normalized.get("math score"))
    reading = parse_score(normalized.get("reading score"))
    writing = parse_score(normalized.get("writing score"))

    if math is None or reading is None or writing is None:
        return None

    return {
        "gender": normalized.get("gender", "unknown").lower(),
        "race_ethnicity": normalized.get("race/ethnicity", "unknown").title(),
        "parental_education": normalized.get("parental level of education", "unknown"),
        "lunch": normalized.get("lunch", "unknown").lower(),
        "test_prep_completed": normalized.get("test preparation course", "").lower() == "completed",
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


def extract_scores_matrix(cleaned_data: List[Dict[str, Any]]) -> np.ndarray:
    return np.array(
        [[r["math_score"], r["reading_score"], r["writing_score"]] for r in cleaned_data],
        dtype=np.float64,
    )


def compute_student_aggregates(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    totals = np.sum(scores, axis=1)
    averages = np.mean(scores, axis=1)
    return totals, averages


def compute_subject_statistics(scores: np.ndarray) -> Dict[str, Dict[str, float]]:
    stats = {}
    for idx, subject in enumerate(SUBJECTS):
        col = scores[:, idx]
        stats[subject] = {
            "mean": float(np.mean(col)),
            "median": float(np.median(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p25": float(np.percentile(col, 25)),
            "p75": float(np.percentile(col, 75)),
        }
    return stats


def filter_passing_students(scores: np.ndarray, passing_threshold: float = 60.0) -> Tuple[np.ndarray, float]:
    passing_mask = np.all(scores >= passing_threshold, axis=1)
    pass_rate = float(np.mean(passing_mask) * 100)
    return passing_mask, pass_rate


def filter_top_performers(avg_scores: np.ndarray, percentile: float = 90.0) -> Tuple[np.ndarray, float]:
    threshold = float(np.percentile(avg_scores, percentile))
    top_mask = avg_scores >= threshold
    return top_mask, threshold


def analyze_test_prep_impact(cleaned_data: List[Dict[str, Any]], avg_scores: np.ndarray) -> Dict[str, Any]:
    prep_mask = np.array([r["test_prep_completed"] for r in cleaned_data], dtype=bool)
    completed_avg = float(np.mean(avg_scores[prep_mask]))
    none_avg = float(np.mean(avg_scores[~prep_mask]))
    return {
        "completed_count": int(np.sum(prep_mask)),
        "completed_mean": completed_avg,
        "none_count": int(np.sum(~prep_mask)),
        "none_mean": none_avg,
        "difference": completed_avg - none_avg,
    }


def compute_z_scores(scores: np.ndarray) -> np.ndarray:
    means = np.mean(scores, axis=0)
    stds = np.std(scores, axis=0)
    return (scores - means) / stds


def resolve_default_data_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "data", "StudentsPerformance.csv"),
        os.path.join(script_dir, "..", "week-01", "data", "StudentsPerformance.csv"),
        "week-01/data/StudentsPerformance.csv",
        "data/StudentsPerformance.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return candidates[0]


def run_pipeline(data_path: str):
    print("=" * 60)
    print("  WEEK-01: STUDENT PERFORMANCE DATA PROCESSING APP")
    print("=" * 60)

    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        sys.exit(1)

    print(f"\n[1/3] Loading raw data from: {data_path}")
    raw_data = load_raw_data(data_path)
    print(f"      Loaded {len(raw_data)} raw records.")

    print("\n[2/3] Cleaning and validating records...")
    cleaned_data = clean_dataset(raw_data)
    dropped = len(raw_data) - len(cleaned_data)
    print(f"      Valid records: {len(cleaned_data)} | Dropped: {dropped}")

    print("\n[3/3] Processing with NumPy...")
    scores = extract_scores_matrix(cleaned_data)
    _, averages = compute_student_aggregates(scores)

    stats = compute_subject_statistics(scores)
    print("\n--- Subject Score Statistics ---")
    header = f"{'Subject':<10} | {'Mean':>7} | {'Median':>7} | {'Std Dev':>7} | {'Min':>5} | {'Max':>5} | {'IQR':>7}"
    print(header)
    print("-" * len(header))
    for subj in SUBJECTS:
        s = stats[subj]
        iqr = s["p75"] - s["p25"]
        print(f"{subj:<10} | {s['mean']:>7.2f} | {s['median']:>7.2f} | {s['std']:>7.2f} | {s['min']:>5.0f} | {s['max']:>5.0f} | {iqr:>7.2f}")

    passing_mask, pass_rate = filter_passing_students(scores, passing_threshold=60.0)
    passing_count = int(passing_mask.sum())
    print("\n--- Student Success Metrics ---")
    print(f"Passed All Subjects (>= 60) : {passing_count}/{len(cleaned_data)} ({pass_rate:.1f}%)")

    top_mask, p90_threshold = filter_top_performers(averages, percentile=90.0)
    print(f"Top 10% Cutoff (90th Pct)   : {p90_threshold:.2f} avg score ({int(top_mask.sum())} students)")

    impact = analyze_test_prep_impact(cleaned_data, averages)
    print("\n--- Impact of Test Preparation ---")
    print(f"Completed Course ({impact['completed_count']} students) : {impact['completed_mean']:.2f} avg")
    print(f"No Prep Course   ({impact['none_count']} students) : {impact['none_mean']:.2f} avg")
    print(f"Performance Difference         : +{impact['difference']:.2f} points higher with prep")

    z_scores = compute_z_scores(scores)
    print("\n--- Sample Standardized Z-Scores (First 3 Students) ---")
    for i in range(min(3, len(cleaned_data))):
        z_str = ", ".join([f"{SUBJECTS[j]}: {z_scores[i, j]:+.2f}" for j in range(3)])
        print(f"Student {i + 1:2d} -> {z_str}")

    print("\n" + "=" * 60)
    print("  Processing finished successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Week-01: Data Processing App")
    parser.add_argument(
        "--data",
        type=str,
        default=resolve_default_data_path(),
        help="Path to raw CSV file",
    )
    args = parser.parse_args()
    run_pipeline(args.data)


if __name__ == "__main__":
    main()
