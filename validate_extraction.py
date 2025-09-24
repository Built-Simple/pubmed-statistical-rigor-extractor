#!/usr/bin/env python3
"""
Validate extraction by sampling random articles and comparing extracted stats
with the original full text
"""

import json
import random
import psycopg2
import re
from pathlib import Path
from datetime import datetime

# Database configuration
db_config = {
    'host': 'localhost',
    'database': 'pmc_fulltext',
    'user': 'postgres',
    'password': 'Tapane2001!',
    'port': 5432
}

def get_random_extracted_articles(extraction_dir, n=20):
    """Get n random articles from extraction results"""

    # Get all JSONL files
    jsonl_files = list(Path(extraction_dir).glob('*.jsonl'))

    # Collect all articles with statistics
    articles_with_stats = []

    for file_path in random.sample(jsonl_files, min(100, len(jsonl_files))):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'error' not in data and data['stats']['rigor_score'] > 50:
                        articles_with_stats.append(data)
                except:
                    continue

    # Sample n articles
    return random.sample(articles_with_stats, min(n, len(articles_with_stats)))

def get_full_text_from_db(article_id):
    """Retrieve full text from database"""
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, pmc_id, pmid, title, journal, pub_year, full_text
        FROM articles
        WHERE id = %s
    """, (article_id,))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        return {
            'id': result[0],
            'pmc_id': result[1],
            'pmid': result[2],
            'title': result[3],
            'journal': result[4],
            'pub_year': result[5],
            'full_text': result[6]
        }
    return None

def manually_extract_stats(text):
    """Manually extract statistics for validation"""

    # Simplified patterns for validation
    patterns = {
        'p_values': re.compile(
            r'[Pp]\s*[<>=≤≥]\s*0?\.?\d+|[Pp][-\s]?value[s]?\s*[<>=≤≥]\s*0?\.?\d+',
            re.IGNORECASE
        ),
        'sample_sizes': re.compile(
            r'[nN]\s*=\s*[\d,]+|sample\s+size\s*:?\s*[\d,]+',
            re.IGNORECASE
        ),
        'confidence_intervals': re.compile(
            r'\d+%?\s*CI[\s:]+\[?[\d.-]+\s*[,;]\s*[\d.-]+\]?',
            re.IGNORECASE
        ),
        'effect_sizes': re.compile(
            r"Cohen'?s?\s*d\s*[=:]\s*[\d.-]+|OR\s*[=:]\s*[\d.-]+|RR\s*[=:]\s*[\d.-]+",
            re.IGNORECASE
        ),
        't_tests': re.compile(
            r't\s*\(\s*\d+\s*\)\s*[=]\s*[\d.-]+|t[-\s]?test',
            re.IGNORECASE
        )
    }

    results = {}
    for category, pattern in patterns.items():
        matches = pattern.findall(text)
        results[category] = matches[:10]  # Limit to first 10 for display

    return results

def validate_extraction():
    """Main validation function"""

    extraction_dir = r"D:\PubMed_Statistical_Analysis\extraction_20250923_210605"

    print("="*80)
    print("EXTRACTION VALIDATION - SAMPLING 20 RANDOM ARTICLES")
    print("="*80)

    # Get random extracted articles
    print("\nStep 1: Sampling extracted articles with rigor score > 50...")
    sample_articles = get_random_extracted_articles(extraction_dir, 20)
    print(f"Sampled {len(sample_articles)} articles")

    validation_results = []

    for i, extracted_data in enumerate(sample_articles, 1):
        article_id = extracted_data['id']

        print(f"\n{'='*70}")
        print(f"ARTICLE {i}/20 - ID: {article_id}")
        print(f"{'='*70}")

        # Get original full text
        original = get_full_text_from_db(article_id)

        if not original:
            print(f"Could not retrieve original text for article {article_id}")
            continue

        # Display article info
        print(f"Title: {original['title'][:100]}...")
        print(f"Journal: {original['journal']}")
        print(f"Year: {original['pub_year']}")
        print(f"PMC ID: {original['pmc_id']}")
        print(f"Text length: {len(original['full_text']):,} characters")

        # Show extracted statistics
        print(f"\n--- EXTRACTED STATISTICS ---")
        print(f"Rigor Score: {extracted_data['stats']['rigor_score']}/100")

        for category in ['p_values', 'sample_sizes', 'confidence_intervals',
                        'effect_sizes', 'statistical_tests']:
            extracted = extracted_data['stats'].get(category, [])
            if extracted:
                print(f"\n{category.upper()} ({len(extracted)} found):")
                for j, item in enumerate(extracted[:5], 1):
                    print(f"  {j}. {item}")
                if len(extracted) > 5:
                    print(f"  ... and {len(extracted)-5} more")

        # Manual re-extraction for validation
        print(f"\n--- VALIDATION CHECK ---")
        manual_stats = manually_extract_stats(original['full_text'][:50000])  # Check first 50k chars

        # Compare counts
        validation_result = {
            'article_id': article_id,
            'title': original['title'][:100],
            'matches': {}
        }

        for category in ['p_values', 'sample_sizes', 'confidence_intervals']:
            extracted_count = len(extracted_data['stats'].get(category, []))
            manual_count = len(manual_stats.get(category, []))

            match_status = "[MATCH]" if abs(extracted_count - manual_count) <= 2 else "[MISMATCH]"

            print(f"{category}: Extracted={extracted_count}, Manual Check={manual_count} {match_status}")

            validation_result['matches'][category] = {
                'extracted': extracted_count,
                'manual': manual_count,
                'status': match_status
            }

            # Show examples from manual extraction
            if manual_stats.get(category) and manual_count > 0:
                print(f"  Manual examples: {manual_stats[category][:3]}")

        validation_results.append(validation_result)

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    total_matches = 0
    total_checks = 0

    for category in ['p_values', 'sample_sizes', 'confidence_intervals']:
        matches = sum(1 for r in validation_results
                     if r['matches'].get(category, {}).get('status') == "[MATCH]")
        total_matches += matches
        total_checks += len(validation_results)
        accuracy = matches / len(validation_results) * 100 if validation_results else 0
        print(f"{category}: {matches}/{len(validation_results)} matched ({accuracy:.1f}% accuracy)")

    overall_accuracy = total_matches / total_checks * 100 if total_checks > 0 else 0
    print(f"\nOverall Validation Accuracy: {overall_accuracy:.1f}%")

    # Save validation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(validation_results),
        'overall_accuracy': overall_accuracy,
        'validation_results': validation_results
    }

    report_file = Path(extraction_dir) / 'validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nValidation report saved to: {report_file}")

if __name__ == "__main__":
    validate_extraction()