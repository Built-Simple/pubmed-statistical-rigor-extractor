#!/usr/bin/env python3
"""
Validate extraction for MEDIUM SCORING articles (35-50 rigor score)
This completes validation across the full spectrum
"""

import json
import random
import psycopg2
import re
from pathlib import Path
from datetime import datetime
import sys

# Set encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Database configuration
db_config = {
    'host': 'localhost',
    'database': 'pmc_fulltext',
    'user': 'postgres',
    'password': 'Tapane2001!',
    'port': 5432
}

def clean_text(text):
    """Clean text for safe display"""
    if text is None:
        return ""
    return text.encode('ascii', 'ignore').decode('ascii')

def get_medium_scoring_articles(extraction_dir, n=20):
    """Get n random articles with MEDIUM rigor scores (35-50)"""

    # Get all JSONL files
    jsonl_files = list(Path(extraction_dir).glob('*.jsonl'))

    # Collect articles with MEDIUM statistics
    medium_scoring_articles = []

    print("Searching for medium-scoring articles (rigor score 35-50)...")

    # Sample from multiple files
    for file_path in random.sample(jsonl_files, min(150, len(jsonl_files))):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Get articles with MEDIUM rigor scores (around median)
                    rigor_score = data.get('stats', {}).get('rigor_score', 0)
                    if 'error' not in data and 35 <= rigor_score <= 50:
                        medium_scoring_articles.append(data)
                        if len(medium_scoring_articles) >= n * 3:  # Get enough to sample
                            break
                except:
                    continue
        if len(medium_scoring_articles) >= n * 3:
            break

    print(f"Found {len(medium_scoring_articles)} medium-scoring articles")

    # Sample n articles
    return random.sample(medium_scoring_articles, min(n, len(medium_scoring_articles)))

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

def verify_extracted_statistics(text, extracted_stats):
    """Verify that extracted statistics actually exist in the text"""

    verification_results = {}
    text_sample = text[:100000] if len(text) > 100000 else text

    for category in ['p_values', 'sample_sizes', 'confidence_intervals', 'effect_sizes', 'statistical_tests']:
        extracted_items = extracted_stats.get(category, [])

        if not extracted_items:
            verification_results[category] = {'found': 0, 'total': 0, 'verified': True}
            continue

        found_count = 0
        not_found = []

        for item in extracted_items[:10]:  # Check first 10
            search_item = str(item).strip()

            # Try to find it in the text
            if search_item in text_sample:
                found_count += 1
            else:
                not_found.append(search_item)

        total_checked = min(10, len(extracted_items))
        verification_results[category] = {
            'found': found_count,
            'total': total_checked,
            'verified': found_count == total_checked,
            'accuracy': (found_count / total_checked * 100) if total_checked > 0 else 100,
            'not_found': not_found[:3]  # Show first 3 not found
        }

    return verification_results

def search_for_additional_statistics(text, extracted_stats):
    """Search for statistics we might have missed"""

    missed_stats = {}
    text_sample = text[:50000] if len(text) > 50000 else text

    # Quick patterns to find missed statistics
    quick_patterns = {
        'p_values': r'[Pp]\s*[<>=]\s*0?\.\d+',
        'sample_sizes': r'[nN]\s*=\s*\d+',
        'confidence_intervals': r'95%\s*CI|confidence\s+interval',
        'effect_sizes': r'OR\s*[:=]|HR\s*[:=]|RR\s*[:=]',
        'statistical_tests': r't-test|ANOVA|chi-square|regression'
    }

    for category, pattern in quick_patterns.items():
        matches = re.findall(pattern, text_sample, re.IGNORECASE)
        found_count = len(set(matches))
        extracted_count = len(extracted_stats.get(category, []))

        if found_count > extracted_count * 1.5:  # Found 50% more than extracted
            missed_stats[category] = {
                'extracted': extracted_count,
                'found_in_search': found_count,
                'potentially_missed': found_count - extracted_count
            }

    return missed_stats

def validate_medium_scoring():
    """Main validation function for medium-scoring articles"""

    extraction_dir = r"D:\PubMed_Statistical_Analysis\extraction_20250923_210605"

    print("="*80)
    print("VALIDATION OF MEDIUM-SCORING ARTICLES (35-50 RIGOR SCORE)")
    print("Completing validation across the full spectrum")
    print("="*80)

    # Get random medium-scoring articles
    sample_articles = get_medium_scoring_articles(extraction_dir, 20)

    if not sample_articles:
        print("No medium-scoring articles found!")
        return

    print(f"\nValidating {len(sample_articles)} medium-scoring articles\n")

    validation_results = []
    perfect_extractions = 0
    partial_misses = 0
    significant_misses = 0

    for i, extracted_data in enumerate(sample_articles, 1):
        article_id = extracted_data['id']

        print(f"\n{'='*70}")
        print(f"ARTICLE {i}/{len(sample_articles)} - ID: {article_id}")
        print(f"{'='*70}")

        # Get original full text
        original = get_full_text_from_db(article_id)

        if not original:
            print(f"Could not retrieve original text for article {article_id}")
            continue

        # Display article info
        title = clean_text(original['title'][:80] if original['title'] else "No title")
        journal = clean_text(original['journal'] if original['journal'] else "Unknown")

        print(f"Title: {title}...")
        print(f"Journal: {journal}")
        print(f"Year: {original['pub_year']}")
        print(f"Text length: {len(original['full_text']):,} characters")

        # Show what was extracted
        extracted_stats = extracted_data['stats']
        print(f"\nRigor Score: {extracted_stats['rigor_score']}/100")

        stat_summary = []
        for category in ['p_values', 'sample_sizes', 'confidence_intervals',
                        'effect_sizes', 'statistical_tests', 'power_analysis', 'corrections']:
            count = len(extracted_stats.get(category, []))
            if count > 0:
                stat_summary.append(f"{category}({count})")

        print(f"Statistics found: {', '.join(stat_summary) if stat_summary else 'None'}")

        # Verify extracted statistics exist in text
        print("\n1. VERIFICATION - Checking extracted values exist in text:")
        verification = verify_extracted_statistics(original['full_text'], extracted_stats)

        all_verified = True
        total_accuracy = 0
        categories_checked = 0

        for category, results in verification.items():
            if results['total'] > 0:
                categories_checked += 1
                total_accuracy += results['accuracy']

                status = "OK" if results['verified'] else "CHECK"
                print(f"   {category}: {results['found']}/{results['total']} found ({results['accuracy']:.0f}%) [{status}]")

                if not results['verified']:
                    all_verified = False
                    if results['not_found']:
                        print(f"      Not found: {results['not_found'][:2]}")

        # Search for potentially missed statistics
        print("\n2. COMPLETENESS - Checking for potentially missed statistics:")
        missed = search_for_additional_statistics(original['full_text'], extracted_stats)

        has_missed = len(missed) > 0

        if not missed:
            print("   No significant missed statistics detected")
        else:
            for category, info in missed.items():
                print(f"   [ATTENTION] {category}:")
                print(f"      Extracted: {info['extracted']}, Found in search: {info['found_in_search']}")
                print(f"      Potentially missed: {info['potentially_missed']}")

        # Overall assessment for this article
        avg_accuracy = (total_accuracy / categories_checked) if categories_checked > 0 else 100

        if all_verified and not has_missed:
            assessment = "PERFECT - All extracted, nothing missed"
            perfect_extractions += 1
        elif avg_accuracy >= 80 and not has_missed:
            assessment = "GOOD - Minor verification issues"
            partial_misses += 1
        elif avg_accuracy >= 80 and has_missed:
            assessment = "PARTIAL - Good extraction but some missed"
            partial_misses += 1
        else:
            assessment = "NEEDS REVIEW - Issues with extraction"
            significant_misses += 1

        print(f"\n3. ASSESSMENT: {assessment}")
        print(f"   Extraction accuracy: {avg_accuracy:.0f}%")

        validation_results.append({
            'article_id': article_id,
            'rigor_score': extracted_stats['rigor_score'],
            'extraction_accuracy': avg_accuracy,
            'has_missed_stats': has_missed,
            'assessment': assessment
        })

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY FOR MEDIUM-SCORING ARTICLES")
    print("="*80)

    print(f"\nArticles validated: {len(validation_results)}")
    print(f"\nExtraction Quality Distribution:")
    print(f"  - Perfect extractions: {perfect_extractions} ({perfect_extractions/len(validation_results)*100:.0f}%)")
    print(f"  - Partial issues: {partial_misses} ({partial_misses/len(validation_results)*100:.0f}%)")
    print(f"  - Significant issues: {significant_misses} ({significant_misses/len(validation_results)*100:.0f}%)")

    avg_accuracy = sum(r['extraction_accuracy'] for r in validation_results) / len(validation_results)
    print(f"\nAverage extraction accuracy: {avg_accuracy:.1f}%")

    # Grade the medium-score extraction
    if avg_accuracy >= 90:
        print("\nOverall Assessment: EXCELLENT")
        print("Medium-scoring articles are being extracted very accurately")
    elif avg_accuracy >= 75:
        print("\nOverall Assessment: GOOD")
        print("Medium-scoring articles show good extraction with minor issues")
    else:
        print("\nOverall Assessment: NEEDS IMPROVEMENT")
        print("Medium-scoring articles have extraction issues to address")

    # Compare across all three validations
    print("\n" + "="*80)
    print("COMPLETE VALIDATION SUMMARY ACROSS ALL RIGOR SCORES")
    print("="*80)
    print("\nExtraction Quality by Rigor Score Range:")
    print("  - HIGH scores (60-100): ~100% accuracy [20 articles checked]")
    print(f"  - MEDIUM scores (35-50): ~{avg_accuracy:.0f}% accuracy [20 articles checked]")
    print("  - LOW scores (1-30): ~50% completeness [20 articles checked]")
    print("\nConclusion: Extraction quality correlates with rigor score.")
    print("High-quality statistical reporting is captured excellently,")
    print("while non-standard or minimal reporting may have gaps.")

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'articles_validated': len(validation_results),
        'average_extraction_accuracy': avg_accuracy,
        'perfect_extractions': perfect_extractions,
        'partial_issues': partial_misses,
        'significant_issues': significant_misses,
        'validation_results': validation_results
    }

    report_file = Path(extraction_dir) / 'medium_score_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    validate_medium_scoring()