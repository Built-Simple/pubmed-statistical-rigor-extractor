#!/usr/bin/env python3
"""
Calculate rigor score statistics from the extracted PubMed data
"""

import json
import os
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
import sys

def analyze_rigor_scores(extraction_dir):
    """Analyze rigor scores from all JSONL files"""

    # Initialize statistics
    all_scores = []
    score_distribution = Counter()
    scores_by_year = defaultdict(list)
    scores_by_journal = defaultdict(list)
    articles_with_stats = 0
    articles_without_stats = 0

    # Categories of statistical reporting
    category_counts = {
        'p_values': 0,
        'sample_sizes': 0,
        'confidence_intervals': 0,
        'effect_sizes': 0,
        'statistical_tests': 0,
        'power_analysis': 0,
        'corrections': 0
    }

    # Get all JSONL files
    jsonl_files = list(Path(extraction_dir).glob('*.jsonl'))
    total_files = len(jsonl_files)

    print(f"Processing {total_files:,} files...")

    # Process each file
    for i, file_path in enumerate(jsonl_files):
        if i % 100 == 0:
            print(f"  Processing file {i+1}/{total_files} ({(i+1)/total_files*100:.1f}%)")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)

                    # Skip error records
                    if 'error' in data:
                        continue

                    # Get rigor score
                    rigor_score = data['stats'].get('rigor_score', 0)
                    all_scores.append(rigor_score)
                    score_distribution[rigor_score] += 1

                    # Track if article has any statistics
                    if rigor_score > 0:
                        articles_with_stats += 1
                    else:
                        articles_without_stats += 1

                    # Track scores by year if metadata available
                    if 'metadata' in data and data['metadata'].get('pub_year'):
                        year = data['metadata']['pub_year']
                        if 1900 <= year <= 2025:  # Sanity check
                            scores_by_year[year].append(rigor_score)

                    # Track scores by journal if metadata available
                    if 'metadata' in data and data['metadata'].get('journal'):
                        journal = data['metadata']['journal']
                        scores_by_journal[journal].append(rigor_score)

                    # Count categories that have data
                    for category in category_counts:
                        if data['stats'].get(category):
                            category_counts[category] += 1

                except Exception as e:
                    # Skip malformed lines
                    continue

    # Calculate statistics
    scores_array = np.array(all_scores)

    print("\n" + "="*60)
    print("RIGOR SCORE ANALYSIS COMPLETE")
    print("="*60)

    # Basic statistics
    print(f"\nTotal articles analyzed: {len(all_scores):,}")
    print(f"Articles with statistics: {articles_with_stats:,} ({articles_with_stats/len(all_scores)*100:.1f}%)")
    print(f"Articles without statistics: {articles_without_stats:,} ({articles_without_stats/len(all_scores)*100:.1f}%)")

    print("\n--- Rigor Score Statistics ---")
    print(f"Mean rigor score: {np.mean(scores_array):.2f}")
    print(f"Median rigor score: {np.median(scores_array):.2f}")
    print(f"Standard deviation: {np.std(scores_array):.2f}")
    print(f"Min score: {np.min(scores_array):.0f}")
    print(f"Max score: {np.max(scores_array):.0f}")

    # Percentiles
    print("\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {np.percentile(scores_array, p):.0f}")

    # Score distribution
    print("\n--- Score Distribution ---")
    score_ranges = {
        "0 (no statistics)": 0,
        "1-25 (minimal)": 0,
        "26-50 (basic)": 0,
        "51-75 (moderate)": 0,
        "76-99 (good)": 0,
        "100 (comprehensive)": 0
    }

    for score, count in score_distribution.items():
        if score == 0:
            score_ranges["0 (no statistics)"] += count
        elif score <= 25:
            score_ranges["1-25 (minimal)"] += count
        elif score <= 50:
            score_ranges["26-50 (basic)"] += count
        elif score <= 75:
            score_ranges["51-75 (moderate)"] += count
        elif score < 100:
            score_ranges["76-99 (good)"] += count
        else:
            score_ranges["100 (comprehensive)"] += count

    for range_name, count in score_ranges.items():
        pct = count / len(all_scores) * 100 if all_scores else 0
        print(f"  {range_name}: {count:,} ({pct:.1f}%)")

    # Statistical categories
    print("\n--- Statistical Categories Reported ---")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(all_scores) * 100 if all_scores else 0
        print(f"  {category}: {count:,} articles ({pct:.1f}%)")

    # Trends by year (last 10 years)
    print("\n--- Rigor Scores by Year (2014-2024) ---")
    for year in sorted(scores_by_year.keys())[-10:]:
        year_scores = scores_by_year[year]
        if year_scores:
            mean_score = np.mean(year_scores)
            print(f"  {year}: mean={mean_score:.2f}, n={len(year_scores):,}")

    # Top journals by average rigor score (with at least 1000 articles)
    print("\n--- Top 10 Journals by Average Rigor Score ---")
    print("(Minimum 1000 articles)")

    journal_avg_scores = []
    for journal, scores in scores_by_journal.items():
        if len(scores) >= 1000:
            journal_avg_scores.append((journal, np.mean(scores), len(scores)))

    journal_avg_scores.sort(key=lambda x: x[1], reverse=True)

    for i, (journal, avg_score, n) in enumerate(journal_avg_scores[:10]):
        journal_display = journal[:50] + "..." if len(journal) > 50 else journal
        print(f"  {i+1}. {journal_display}: {avg_score:.2f} (n={n:,})")

    # Save summary to JSON
    summary = {
        'total_articles': len(all_scores),
        'articles_with_stats': articles_with_stats,
        'articles_without_stats': articles_without_stats,
        'mean_rigor_score': float(np.mean(scores_array)),
        'median_rigor_score': float(np.median(scores_array)),
        'std_rigor_score': float(np.std(scores_array)),
        'score_distribution': dict(score_ranges),
        'category_counts': category_counts
    }

    summary_file = os.path.join(extraction_dir, 'rigor_analysis_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    return summary

if __name__ == "__main__":
    # Use the most recent extraction directory
    extraction_dir = r"D:\PubMed_Statistical_Analysis\extraction_20250923_210605"

    if not os.path.exists(extraction_dir):
        print(f"Error: Directory not found: {extraction_dir}")
        sys.exit(1)

    analyze_rigor_scores(extraction_dir)