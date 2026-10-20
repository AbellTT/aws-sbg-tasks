"""
CLI entry point for Student Exam Performance Data-Processing App.
Usage:
    python main.py
    python main.py --data path/to/dataset.csv
"""
import argparse
import os
import sys

from src.cleaner import load_raw_data, clean_dataset
from src.processor import (
    extract_scores_matrix,
    compute_student_aggregates,
    compute_subject_statistics,
    filter_passing_students,
    filter_top_performers,
    analyze_test_prep_impact,
    compute_z_scores,
    SUBJECTS,
)


def run_pipeline(data_path: str):
    print("=" * 60)
    print("  STUDENT PERFORMANCE DATA PROCESSING APP (CLI)")
    print("=" * 60)

    # 1. Verification & Data Loading
    if not os.path.exists(data_path):
        print(f"[Error] Dataset not found at: {data_path}")
        sys.exit(1)

    print(f"\n[1/3] Loading raw data from: {data_path}")
    raw_data = load_raw_data(data_path)
    print(f"      Loaded {len(raw_data)} raw records.")

    # 2. Data Cleaning
    print("\n[2/3] Cleaning and validating records...")
    cleaned_data = clean_dataset(raw_data)
    dropped = len(raw_data) - len(cleaned_data)
    print(f"      Valid records: {len(cleaned_data)} | Dropped: {dropped}")

    # 3. NumPy Processing
    print("\n[3/3] Processing with NumPy...")
    scores = extract_scores_matrix(cleaned_data)
    totals, averages = compute_student_aggregates(scores)

    # Subject Statistics
    stats = compute_subject_statistics(scores)
    print("\n--- Subject Score Statistics ---")
    header = f"{'Subject':<10} | {'Mean':>7} | {'Median':>7} | {'Std Dev':>7} | {'Min':>5} | {'Max':>5} | {'IQR':>7}"
    print(header)
    print("-" * len(header))
    for subj in SUBJECTS:
        s = stats[subj]
        iqr = s["p75"] - s["p25"]
        print(f"{subj:<10} | {s['mean']:>7.2f} | {s['median']:>7.2f} | {s['std']:>7.2f} | {s['min']:>5.0f} | {s['max']:>5.0f} | {iqr:>7.2f}")

    # Passing filter (scores >= 60 in all subjects)
    passing_mask, pass_rate = filter_passing_students(scores, passing_threshold=60.0)
    passing_count = int(passing_mask.sum())
    print("\n--- Student Success Metrics ---")
    print(f"Passed All Subjects (>= 60) : {passing_count}/{len(cleaned_data)} ({pass_rate:.1f}%)")

    # Top Performers (90th percentile)
    top_mask, p90_threshold = filter_top_performers(averages, percentile=90.0)
    print(f"Top 10% Cutoff (90th Pct)   : {p90_threshold:.2f} avg score ({int(top_mask.sum())} students)")

    # Impact of Test Preparation
    impact = analyze_test_prep_impact(cleaned_data, averages)
    print("\n--- Impact of Test Preparation ---")
    print(f"Completed Course ({impact['completed_count']} students) : {impact['completed_mean']:.2f} avg")
    print(f"No Prep Course   ({impact['none_count']} students) : {impact['none_mean']:.2f} avg")
    print(f"Performance Difference         : +{impact['difference']:.2f} points higher with prep")

    # Z-scores sample
    z_scores = compute_z_scores(scores)
    print("\n--- Sample Standardized Z-Scores (First 3 Students) ---")
    for i in range(min(3, len(cleaned_data))):
        z_str = ", ".join([f"{SUBJECTS[j]}: {z_scores[i, j]:+.2f}" for j in range(3)])
        print(f"Student {i + 1:2d} -> {z_str}")

    print("\n" + "=" * 60)
    print("  Processing finished successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Student Exam Performance Data Processor")
    parser.add_argument(
        "--data",
        type=str,
        default="data/StudentsPerformance.csv",
        help="Path to raw CSV file (default: data/StudentsPerformance.csv)",
    )
    args = parser.parse_args()
    run_pipeline(args.data)


if __name__ == "__main__":
    main()
