from typing import List, Dict, Any, Tuple
import numpy as np

SUBJECTS = ["Math", "Reading", "Writing"]


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
