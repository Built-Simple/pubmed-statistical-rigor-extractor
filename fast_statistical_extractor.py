#!/usr/bin/env python3
"""
PubMed Statistical Analysis - Windows Optimized for Ryzen 9 7900
Writes to D: drive, includes restart capability and comprehensive error logging
"""

import re
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import logging
import os
import sys
from typing import Dict, List, Any, Optional
import multiprocessing as mp
from functools import partial
import time
import pickle
import traceback
from pathlib import Path

# Windows-specific configuration
OUTPUT_BASE_DIR = r"D:\PubMed_Statistical_Analysis"
TEMP_DIR = r"C:\Temp\PubMed_Analysis"
LOG_DIR = r"D:\PubMed_Statistical_Analysis\logs"
CHECKPOINT_DIR = r"D:\PubMed_Statistical_Analysis\checkpoints"

# Create all necessary directories
for directory in [OUTPUT_BASE_DIR, TEMP_DIR, LOG_DIR, CHECKPOINT_DIR]:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Set up logging with both file and console output
def setup_logging():
    """Configure logging for Windows with timestamped log files"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Main log file
    log_file = os.path.join(LOG_DIR, f'extraction_{timestamp}.log')
    error_log_file = os.path.join(LOG_DIR, f'errors_{timestamp}.log')

    # Configure main logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Create separate error logger
    error_logger = logging.getLogger('errors')
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setFormatter(logging.Formatter(
        '%(asctime)s - ARTICLE_ID: %(article_id)s - ERROR: %(message)s\n%(traceback)s\n'
    ))
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.ERROR)

    return logging.getLogger(__name__), error_logger, error_log_file

logger, error_logger, ERROR_LOG_PATH = setup_logging()

class CheckpointManager:
    """Manages restart capability with checkpoint files"""

    def __init__(self):
        self.checkpoint_file = os.path.join(CHECKPOINT_DIR, 'extraction_checkpoint.json')
        self.processed_ids_file = os.path.join(CHECKPOINT_DIR, 'processed_ids.pkl')

    def save_checkpoint(self, batch_num: int, processed_ids: set, total_processed: int):
        """Save current progress for restart capability"""
        checkpoint = {
            'batch_num': batch_num,
            'total_processed': total_processed,
            'timestamp': datetime.now().isoformat(),
            'output_dir': OUTPUT_BASE_DIR
        }

        # Save checkpoint
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Save processed IDs separately (more efficient for large sets)
        with open(self.processed_ids_file, 'wb') as f:
            pickle.dump(processed_ids, f)

        logger.info(f"Checkpoint saved: Batch {batch_num}, {total_processed:,} articles processed")

    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if exists"""
        if os.path.exists(self.checkpoint_file) and os.path.exists(self.processed_ids_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)

                with open(self.processed_ids_file, 'rb') as f:
                    processed_ids = pickle.load(f)

                checkpoint['processed_ids'] = processed_ids
                logger.info(f"Checkpoint loaded: Resuming from batch {checkpoint['batch_num']}")
                return checkpoint
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return None

class FastStatisticalExtractor:
    """Optimized extractor for Ryzen 9 7900"""

    def __init__(self):
        """Initialize with pre-compiled patterns"""
        # Compile all patterns once
        self.patterns = {
            'p_values': re.compile(
                r'[Pp]\s*[<>=≤≥]\s*0?\.?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|'
                r'[Pp]\s*\(\s*[<>=≤≥]\s*0?\.?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*\)|'
                r'[Pp][-\s]?value[s]?\s*[<>=≤≥]\s*0?\.?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?|'
                r'[Pp]\s*=\s*NS|[Pp]\s*=\s*n\.?s\.?|'
                r'significance\s+at\s+[Pp]\s*[<>=≤≥]\s*0?\.?\d+',
                re.IGNORECASE
            ),

            'sample_sizes': re.compile(
                r'[nN]\s*=\s*[\d,]+|'
                r'[nN]\s*\(\w+\)\s*=\s*[\d,]+|'
                r'sample\s+size\s*:?\s*[\d,]+|'
                r'[\d,]+\s+(?:patients?|participants?|subjects?|cases?|controls?)|'
                r'total\s+[nN]\s*=\s*[\d,]+',
                re.IGNORECASE
            ),

            'confidence_intervals': re.compile(
                r'\d+(?:\.\d+)?%?\s*CI[\s:]+\[?[\d.-]+\s*[,;]\s*[\d.-]+\]?|'
                r'\d+(?:\.\d+)?%?\s*confidence\s+interval[\s:]+\[?[\d.-]+\s*[,;]\s*[\d.-]+\]?|'
                r'CI\s*\d+%?\s*[=:]\s*\[?[\d.-]+\s*(?:to|-|–|—)\s*[\d.-]+\]?|'
                r'\[[\d.-]+\s*[,;]\s*[\d.-]+\]\s*\(?(?:95|99)%?\s*CI\)?',
                re.IGNORECASE
            ),

            'effect_sizes': re.compile(
                r"Cohen'?s?\s*d\s*[=:]\s*[\d.-]+|"
                r'd\s*=\s*[\d.-]+\s*\(?(?:small|medium|large)?\)?|'
                r'OR\s*[=:]\s*[\d.-]+(?:\s*\[[\d.-]+\s*[,;]\s*[\d.-]+\])?|'
                r'RR\s*[=:]\s*[\d.-]+(?:\s*\[[\d.-]+\s*[,;]\s*[\d.-]+\])?|'
                r'HR\s*[=:]\s*[\d.-]+(?:\s*\[[\d.-]+\s*[,;]\s*[\d.-]+\])?|'
                r'r\s*[=]\s*[+-]?0?\.?\d+|'
                r'R\^?2\s*[=]\s*0?\.?\d+|'
                r'β\s*[=]\s*[\d.-]+|'
                r'beta\s*[=]\s*[\d.-]+',
                re.IGNORECASE
            ),

            'statistical_tests': re.compile(
                r't\s*\(\s*\d+(?:\.\d+)?\s*\)\s*[=]\s*[\d.-]+|'
                r't[-\s]?test|paired[-\s]?samples?\s+t[-\s]?test|'
                r'F\s*\(\s*\d+\s*,\s*\d+\s*\)\s*[=]\s*[\d.-]+|'
                r'ANOVA|ANCOVA|MANOVA|'
                r'χ2?\s*\(\s*\d+\s*\)\s*[=]\s*[\d.-]+|'
                r'chi[-\s]?squared?\s*[=]\s*[\d.-]+|'
                r'Mann[-\s]?Whitney|Wilcoxon|Kruskal[-\s]?Wallis|'
                r'regression\s+coefficient|'
                r'Bonferroni|Fisher\'?s?\s+exact|'
                r'Shapiro[-\s]?Wilk|Kolmogorov[-\s]?Smirnov',
                re.IGNORECASE
            ),

            'power_analysis': re.compile(
                r'power\s*[=:]\s*0?\.?\d+|'
                r'statistical\s+power\s*[=:]\s*0?\.?\d+|'
                r'achieved\s+power\s*[=:]\s*0?\.?\d+|'
                r'post[-\s]?hoc\s+power|'
                r'1\s*-\s*β\s*[=:]\s*0?\.?\d+',
                re.IGNORECASE
            ),

            'corrections': re.compile(
                r'Bonferroni[-\s]?(?:corrected|adjusted)|'
                r'False\s+Discovery\s+Rate|FDR[-\s]?(?:corrected|adjusted)|'
                r'Benjamini[-\s]?Hochberg|'
                r'q[-\s]?value\s*[<>=]\s*0?\.?\d+',
                re.IGNORECASE
            )
        }

    def extract_stats(self, text: str) -> Dict:
        """Extract all statistics from text"""
        results = {}

        for category, pattern in self.patterns.items():
            try:
                matches = pattern.findall(text)
                results[category] = list(set(matches)) if matches else []
            except Exception as e:
                results[category] = []
                results['extraction_error'] = str(e)

        # Calculate basic rigor score
        score = 0
        if results.get('p_values'): score += 25
        if results.get('sample_sizes'): score += 25
        if results.get('confidence_intervals'): score += 20
        if results.get('effect_sizes'): score += 20
        if results.get('statistical_tests'): score += 10

        results['rigor_score'] = score

        return results

def process_batch_worker(args):
    """Worker function for multiprocessing"""
    batch_data, output_file, extractor = args

    processed = 0
    errors = []

    with open(output_file, 'w', encoding='utf-8') as f:
        for article_id, text in batch_data:
            try:
                stats = extractor.extract_stats(text)
                result = {
                    'article_id': article_id,
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }
                f.write(json.dumps(result) + '\n')
                processed += 1

            except Exception as e:
                error_info = {
                    'article_id': article_id,
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                f.write(json.dumps(error_info) + '\n')
                errors.append((article_id, str(e)))

    return processed, errors

def main():
    """Main extraction pipeline for Windows"""

    # Database configuration - UPDATED WITH ACTUAL CREDENTIALS
    db_config = {
        'host': 'localhost',
        'database': 'pmc_fulltext',        # PubMed Central database with 4.27M articles
        'user': 'postgres',                # PostgreSQL username
        'password': 'Tapane2001!',         # PostgreSQL password
        'port': 5432
    }

    # Use 20 processes (leaving 4 threads for system)
    num_processes = 20

    # Initialize components
    checkpoint_mgr = CheckpointManager()
    extractor = FastStatisticalExtractor()

    # Check for existing checkpoint
    checkpoint = checkpoint_mgr.load_checkpoint()
    processed_ids = checkpoint['processed_ids'] if checkpoint else set()
    start_batch = checkpoint['batch_num'] + 1 if checkpoint else 0

    logger.info(f"Starting extraction with {num_processes} processes")
    if checkpoint:
        logger.info(f"Resuming from batch {start_batch}, {len(processed_ids):,} articles already processed")

    # Connect to database
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM articles WHERE full_text IS NOT NULL")
        total_articles = cursor.fetchone()[0]
        logger.info(f"Total articles in database: {total_articles:,}")

        # Prepare for batch processing
        cursor_name = f'article_cursor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        cursor_ss = conn.cursor(name=cursor_name)

        # Build query with checkpoint support
        if processed_ids:
            # This is tricky with large sets, might need different approach
            cursor_ss.execute("""
                SELECT id, full_text
                FROM articles
                WHERE full_text IS NOT NULL
                ORDER BY id
            """)
        else:
            cursor_ss.execute("""
                SELECT id, full_text
                FROM articles
                WHERE full_text IS NOT NULL
                ORDER BY id
            """)

        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(OUTPUT_BASE_DIR, f'extraction_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        # Process in batches
        batch_size = 10000  # Articles per batch file
        batch_num = start_batch
        total_processed = len(processed_ids)
        start_time = time.time()

        # Create process pool
        pool = mp.Pool(num_processes)

        while True:
            # Fetch batch
            batch = cursor_ss.fetchmany(batch_size)
            if not batch:
                break

            # Filter out already processed articles
            batch = [(aid, text) for aid, text in batch if aid not in processed_ids]
            if not batch:
                continue

            # Create batch file paths
            batch_files = []
            chunk_size = len(batch) // num_processes + 1
            chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]

            # Prepare worker arguments
            worker_args = []
            for i, chunk in enumerate(chunks):
                output_file = os.path.join(output_dir, f'batch_{batch_num:06d}_part_{i:02d}.jsonl')
                batch_files.append(output_file)
                worker_args.append((chunk, output_file, extractor))

            # Process in parallel
            results = pool.map(process_batch_worker, worker_args)

            # Collect results and errors
            batch_processed = 0
            batch_errors = []
            for processed, errors in results:
                batch_processed += processed
                batch_errors.extend(errors)

            # Log errors
            for article_id, error in batch_errors:
                error_logger.error(
                    "Failed to process article",
                    extra={'article_id': article_id, 'traceback': error}
                )

            # Update tracking
            total_processed += batch_processed
            for aid, _ in batch:
                processed_ids.add(aid)

            # Save checkpoint every batch
            checkpoint_mgr.save_checkpoint(batch_num, processed_ids, total_processed)

            # Progress report
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            remaining = total_articles - total_processed
            eta = remaining / rate if rate > 0 else 0

            logger.info(
                f"Batch {batch_num:,} complete | "
                f"Processed: {total_processed:,}/{total_articles:,} ({total_processed/total_articles*100:.1f}%) | "
                f"Rate: {rate:.0f} articles/sec | "
                f"ETA: {eta/60:.1f} minutes | "
                f"Errors this batch: {len(batch_errors)}"
            )

            batch_num += 1

        pool.close()
        pool.join()
        cursor_ss.close()
        cursor.close()
        conn.close()

        # Final report
        total_time = time.time() - start_time
        logger.info("="*80)
        logger.info("EXTRACTION COMPLETE!")
        logger.info(f"Total articles processed: {total_processed:,}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average rate: {total_processed/total_time:.0f} articles/second")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Error log: {ERROR_LOG_PATH}")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExtraction interrupted by user. Progress has been saved.")
        logger.info("Run the script again to resume from last checkpoint.")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        sys.exit(1)