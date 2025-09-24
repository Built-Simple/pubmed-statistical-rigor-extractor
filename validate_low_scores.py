#!/usr/bin/env python3
"""
Validate extraction for LOW SCORING articles to ensure we didn't miss statistics
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

def get_low_scoring_articles(extraction_dir, n=20):
    """Get n random articles with LOW rigor scores (0-30)"""

    # Get all JSONL files
    jsonl_files = list(Path(extraction_dir).glob('*.jsonl'))

    # Collect articles with LOW statistics
    low_scoring_articles = []

    print("Searching for low-scoring articles (rigor score 1-30)...")

    # Sample from multiple files
    for file_path in random.sample(jsonl_files, min(200, len(jsonl_files))):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Get articles with LOW rigor scores (but not 0)
                    rigor_score = data.get('stats', {}).get('rigor_score', 0)
                    if 'error' not in data and 1 <= rigor_score <= 30:
                        low_scoring_articles.append(data)
                        if len(low_scoring_articles) >= n * 3:  # Get enough to sample
                            break
                except:
                    continue
        if len(low_scoring_articles) >= n * 3:
            break

    print(f"Found {len(low_scoring_articles)} low-scoring articles")

    # Sample n articles
    return random.sample(low_scoring_articles, min(n, len(low_scoring_articles)))

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

def search_for_missed_statistics(text, extracted_stats):
    """Search for potentially missed statistics in the text"""

    # Limit text to first 100k characters for efficiency
    text_sample = text[:100000] if len(text) > 100000 else text

    missed_stats = {
        'p_values': [],
        'sample_sizes': [],
        'confidence_intervals': [],
        'effect_sizes': [],
        'statistical_tests': []
    }

    # More aggressive patterns to catch anything we might have missed
    patterns = {
        'p_values': [
            r'p\s*[<>=]\s*0?\.\d+',
            r'P\s*[<>=]\s*0?\.\d+',
            r'significance.*0\.\d+',
            r'significant\s+at\s+0\.\d+',
            r'p-value.*0\.\d+'
        ],
        'sample_sizes': [
            r'[nN]\s*=\s*\d+',
            r'\d+\s+patients',
            r'\d+\s+subjects',
            r'\d+\s+participants',
            r'sample.*\d+',
            r'recruited\s+\d+',
            r'enrolled\s+\d+'
        ],
        'confidence_intervals': [
            r'95%\s*CI',
            r'99%\s*CI',
            r'confidence\s+interval',
            r'CI\s*:?\s*\[',
            r'CI\s*:?\s*\('
        ],
        'effect_sizes': [
            r'd\s*=\s*[\d.-]+',
            r'[Ee]ffect\s+size',
            r'OR\s*[:=]',
            r'HR\s*[:=]',
            r'RR\s*[:=]',
            r'[Cc]orrelation.*r\s*='
        ],
        'statistical_tests': [
            r't-test',
            r'ANOVA',
            r'[Cc]hi-square',
            r'[Mm]ann-[Ww]hitney',
            r'[Ww]ilcoxon',
            r'[Kk]ruskal',
            r'regression',
            r'[Ff]isher.*exact'
        ]
    }

    # Check what was already extracted
    already_extracted = extracted_stats

    # Search for each pattern
    for category, pattern_list in patterns.items():
        found_items = []
        for pattern in pattern_list:
            matches = re.findall(pattern, text_sample, re.IGNORECASE)
            found_items.extend(matches)

        # Compare with what was extracted
        extracted_count = len(already_extracted.get(category, []))
        found_count = len(set(found_items))

        if found_count > extracted_count:
            # We found more than what was extracted
            missed_stats[category] = {
                'extracted': extracted_count,
                'found_now': found_count,
                'potentially_missed': found_count - extracted_count,
                'examples': list(set(found_items))[:5]
            }

    return missed_stats

def validate_low_scoring():
    """Main validation function for low-scoring articles"""

    extraction_dir = r"D:\PubMed_Statistical_Analysis\extraction_20250923_210605"

    print("="*80)
    print("VALIDATION OF LOW-SCORING ARTICLES")
    print("Checking if we missed any statistics...")
    print("="*80)

    # Get random low-scoring articles
    sample_articles = get_low_scoring_articles(extraction_dir, 20)

    if not sample_articles:
        print("No low-scoring articles found!")
        return

    print(f"\nValidating {len(sample_articles)} low-scoring articles\n")

    validation_results = []
    articles_with_missed_stats = 0

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
        print(f"\nCurrent Rigor Score: {extracted_stats['rigor_score']}/100")

        print("Extracted statistics:")
        for category in ['p_values', 'sample_sizes', 'confidence_intervals',
                        'effect_sizes', 'statistical_tests']:
            count = len(extracted_stats.get(category, []))
            if count > 0:
                print(f"  - {category}: {count}")
                # Show first few examples
                examples = extracted_stats[category][:3]
                for ex in examples:
                    print(f"      '{clean_text(str(ex))}'")

        # Search for potentially missed statistics
        print("\nSearching for potentially missed statistics...")
        missed = search_for_missed_statistics(original['full_text'], extracted_stats)

        has_missed = False
        for category, info in missed.items():
            if isinstance(info, dict) and info.get('potentially_missed', 0) > 0:
                has_missed = True
                print(f"\n  [ATTENTION] {category}:")
                print(f"    Extracted: {info['extracted']}")
                print(f"    Found now: {info['found_now']}")
                print(f"    Potentially missed: {info['potentially_missed']}")
                if info['examples']:
                    print(f"    Examples found in text:")
                    for ex in info['examples'][:3]:
                        print(f"      - {clean_text(str(ex))}")

        if has_missed:
            articles_with_missed_stats += 1
            print("\n  => Some statistics may have been missed")
        else:
            print("\n  => Extraction appears complete for this article")

        # Check if this is truly a low-statistics paper
        text_sample = original['full_text'][:10000]  # Check first 10k chars

        # Quick check for research article indicators
        is_research = any(word in text_sample.lower() for word in
                         ['methods', 'results', 'statistical analysis', 'data analysis'])

        if not is_research:
            print("\n  Note: This may be an editorial, review, or commentary (no Methods section)")

        validation_results.append({
            'article_id': article_id,
            'rigor_score': extracted_stats['rigor_score'],
            'has_missed_stats': has_missed,
            'is_research_article': is_research
        })

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY FOR LOW-SCORING ARTICLES")
    print("="*80)

    print(f"\nArticles checked: {len(validation_results)}")
    print(f"Articles with potentially missed statistics: {articles_with_missed_stats}")

    research_articles = [r for r in validation_results if r['is_research_article']]
    non_research = [r for r in validation_results if not r['is_research_article']]

    print(f"\nArticle types:")
    print(f"  - Likely research articles: {len(research_articles)}")
    print(f"  - Likely non-research (editorials/reviews): {len(non_research)}")

    if research_articles:
        missed_in_research = sum(1 for r in research_articles if r['has_missed_stats'])
        print(f"\nOf research articles:")
        print(f"  - With potentially missed stats: {missed_in_research}/{len(research_articles)}")

    # Calculate accuracy
    accuracy = (len(validation_results) - articles_with_missed_stats) / len(validation_results) * 100

    print(f"\nExtraction completeness: {accuracy:.1f}%")

    if accuracy >= 90:
        print("Assessment: EXCELLENT - Very few missed statistics")
    elif accuracy >= 80:
        print("Assessment: GOOD - Acceptable miss rate")
    elif accuracy >= 70:
        print("Assessment: FAIR - Some patterns may need adjustment")
    else:
        print("Assessment: NEEDS REVIEW - Many statistics may be missed")

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'articles_checked': len(validation_results),
        'articles_with_missed_stats': articles_with_missed_stats,
        'extraction_completeness': accuracy,
        'validation_results': validation_results
    }

    report_file = Path(extraction_dir) / 'low_score_validation_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

if __name__ == "__main__":
    validate_low_scoring()