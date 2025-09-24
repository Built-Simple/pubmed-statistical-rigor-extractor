#!/usr/bin/env python3
"""
Validate extraction by sampling random articles and comparing extracted stats
with the original full text - Safe version without Unicode issues
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
    # Replace non-ASCII characters
    return text.encode('ascii', 'ignore').decode('ascii')

def get_random_extracted_articles(extraction_dir, n=20):
    """Get n random articles from extraction results"""

    # Get all JSONL files
    jsonl_files = list(Path(extraction_dir).glob('*.jsonl'))

    # Collect articles with good statistics
    articles_with_stats = []

    # Sample from multiple files to get diversity
    for file_path in random.sample(jsonl_files, min(100, len(jsonl_files))):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Get articles with moderate to high rigor scores
                    if 'error' not in data and data['stats']['rigor_score'] >= 60:
                        articles_with_stats.append(data)
                        if len(articles_with_stats) >= n * 5:  # Get enough to sample from
                            break
                except:
                    continue
        if len(articles_with_stats) >= n * 5:
            break

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

def manually_verify_stats(text, extracted_stats):
    """Verify extracted statistics by searching for them in the text"""

    verification_results = {}

    # For each category, check if extracted values exist in text
    for category in ['p_values', 'sample_sizes', 'confidence_intervals', 'effect_sizes']:
        extracted_items = extracted_stats.get(category, [])

        if not extracted_items:
            verification_results[category] = {'found': 0, 'total': 0, 'examples': []}
            continue

        found_count = 0
        examples = []

        for item in extracted_items[:10]:  # Check first 10
            # Clean the item for searching
            search_item = str(item).strip()

            # Try to find it in the text
            if search_item in text:
                found_count += 1
                # Get context around the match
                idx = text.find(search_item)
                start = max(0, idx - 30)
                end = min(len(text), idx + len(search_item) + 30)
                context = text[start:end]
                context = clean_text(context)
                examples.append(f"'{search_item}' found in: ...{context}...")

        verification_results[category] = {
            'found': found_count,
            'total': len(extracted_items[:10]),
            'examples': examples[:3]
        }

    return verification_results

def validate_extraction():
    """Main validation function"""

    extraction_dir = r"D:\PubMed_Statistical_Analysis\extraction_20250923_210605"

    print("="*80)
    print("EXTRACTION VALIDATION - SAMPLING 20 RANDOM ARTICLES")
    print("="*80)

    # Get random extracted articles
    print("\nSampling articles with rigor score >= 60...")
    sample_articles = get_random_extracted_articles(extraction_dir, 20)
    print(f"Found {len(sample_articles)} articles to validate\n")

    validation_summary = []

    for i, extracted_data in enumerate(sample_articles, 1):
        article_id = extracted_data['id']

        print(f"\n{'='*70}")
        print(f"VALIDATION {i}/20 - Article ID: {article_id}")
        print(f"{'='*70}")

        # Get original full text
        original = get_full_text_from_db(article_id)

        if not original:
            print(f"Could not retrieve original text for article {article_id}")
            continue

        # Display article info (safely)
        title = clean_text(original['title'][:100] if original['title'] else "No title")
        journal = clean_text(original['journal'] if original['journal'] else "Unknown")

        print(f"Title: {title}...")
        print(f"Journal: {journal}")
        print(f"Year: {original['pub_year']}")
        print(f"PMC ID: {original['pmc_id']}")
        print(f"Text length: {len(original['full_text']):,} characters")

        # Show extracted statistics summary
        extracted_stats = extracted_data['stats']
        print(f"\nExtracted Rigor Score: {extracted_stats['rigor_score']}/100")

        # Count statistics
        stat_counts = {}
        for category in ['p_values', 'sample_sizes', 'confidence_intervals',
                        'effect_sizes', 'statistical_tests', 'power_analysis', 'corrections']:
            count = len(extracted_stats.get(category, []))
            stat_counts[category] = count
            if count > 0:
                print(f"  - {category}: {count} found")

        # Verify extraction by searching for extracted values in original text
        print(f"\nVERIFICATION - Checking if extracted values exist in text:")

        verification = manually_verify_stats(original['full_text'], extracted_stats)

        validation_score = 0
        total_checks = 0

        for category, results in verification.items():
            if results['total'] > 0:
                accuracy = results['found'] / results['total'] * 100
                total_checks += results['total']
                validation_score += results['found']
                status = "GOOD" if accuracy >= 80 else "CHECK"
                print(f"  {category}: {results['found']}/{results['total']} verified ({accuracy:.0f}%) [{status}]")

                # Show examples
                for example in results['examples'][:2]:
                    print(f"    Example: {example[:100]}")

        # Overall validation score for this article
        if total_checks > 0:
            overall_accuracy = validation_score / total_checks * 100
            print(f"\nArticle Validation Score: {overall_accuracy:.0f}%")

            validation_summary.append({
                'article_id': article_id,
                'rigor_score': extracted_stats['rigor_score'],
                'validation_accuracy': overall_accuracy,
                'checks_performed': total_checks,
                'checks_passed': validation_score
            })

    # Final Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    if validation_summary:
        avg_accuracy = sum(v['validation_accuracy'] for v in validation_summary) / len(validation_summary)
        avg_rigor = sum(v['rigor_score'] for v in validation_summary) / len(validation_summary)
        total_checks = sum(v['checks_performed'] for v in validation_summary)
        total_passed = sum(v['checks_passed'] for v in validation_summary)

        print(f"\nArticles Validated: {len(validation_summary)}")
        print(f"Average Rigor Score: {avg_rigor:.1f}/100")
        print(f"Average Validation Accuracy: {avg_accuracy:.1f}%")
        print(f"Total Checks Performed: {total_checks}")
        print(f"Total Checks Passed: {total_passed}")
        print(f"Overall Success Rate: {total_passed/total_checks*100:.1f}%")

        # Grade the extraction
        if avg_accuracy >= 90:
            print("\nExtraction Quality: EXCELLENT - Highly accurate extraction")
        elif avg_accuracy >= 80:
            print("\nExtraction Quality: GOOD - Extraction is working well")
        elif avg_accuracy >= 70:
            print("\nExtraction Quality: FAIR - Some improvements needed")
        else:
            print("\nExtraction Quality: NEEDS REVIEW - Check extraction patterns")

        # Save report
        report = {
            'timestamp': datetime.now().isoformat(),
            'articles_validated': len(validation_summary),
            'average_validation_accuracy': avg_accuracy,
            'average_rigor_score': avg_rigor,
            'overall_success_rate': total_passed/total_checks*100 if total_checks > 0 else 0,
            'details': validation_summary
        }

        report_file = Path(extraction_dir) / 'validation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    validate_extraction()