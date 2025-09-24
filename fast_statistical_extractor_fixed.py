#!/usr/bin/env python3
"""
PubMed Statistical Analysis - Windows Optimized for Ryzen 9 7900
Fixed version with correct database schema
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
from typing import Dict, List, Any, Optional, Tuple
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

# Configuration options
INCLUDE_METADATA = True  # Set to True to include PMC ID, title, journal, etc.
BATCH_SIZE = 10000  # Articles per batch file
NUM_PROCESSES = 20  # Use 20 processes (leaving 4 threads for system)

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
        '%(asctime)s - ID: %(article_id)s - ERROR: %(message)s\n%(traceback)s\n'
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

def process_batch_worker(args: Tuple) -> Tuple[int, List]:
    """Worker function for multiprocessing - processes a batch of articles"""
    batch_data, output_file, extractor, include_metadata = args

    processed = 0
    errors = []

    with open(output_file, 'w', encoding='utf-8') as f:
        for article_record in batch_data:
            try:
                # Extract data based on whether metadata is included
                if include_metadata:
                    article_id = article_record['id']
                    text = article_record['full_text']
                    metadata = {
                        'pmc_id': article_record.get('pmc_id'),
                        'pmid': article_record.get('pmid'),
                        'doi': article_record.get('doi'),
                        'title': article_record.get('title'),
                        'journal': article_record.get('journal'),
                        'pub_year': article_record.get('pub_year')
                    }
                else:
                    article_id, text = article_record
                    metadata = None

                # Extract statistics
                stats = extractor.extract_stats(text)

                # Build result
                result = {
                    'id': article_id,  # Using correct column name
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                }

                # Add metadata if available
                if metadata:
                    result['metadata'] = metadata

                f.write(json.dumps(result) + '\n')
                processed += 1

            except Exception as e:
                article_id = article_record['id'] if include_metadata else article_record[0]
                error_info = {
                    'id': article_id,  # Using correct column name
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'timestamp': datetime.now().isoformat()
                }
                f.write(json.dumps(error_info) + '\n')
                errors.append((article_id, str(e)))

    return processed, errors

def main():
    """Main extraction pipeline for Windows"""

    # Database configuration - PubMed Central database
    db_config = {
        'host': 'localhost',
        'database': 'pmc_fulltext',        # PubMed Central database with 4.27M articles
        'user': 'postgres',                # PostgreSQL username
        'password': 'Tapane2001!',         # PostgreSQL password
        'port': 5432
    }

    # Initialize components
    checkpoint_mgr = CheckpointManager()
    extractor = FastStatisticalExtractor()

    # Check for existing checkpoint
    checkpoint = checkpoint_mgr.load_checkpoint()
    processed_ids = checkpoint['processed_ids'] if checkpoint else set()
    start_batch = checkpoint['batch_num'] + 1 if checkpoint else 0

    logger.info(f"Starting extraction with {NUM_PROCESSES} processes")
    logger.info(f"Include metadata: {INCLUDE_METADATA}")
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

        # Show sample of data schema
        cursor.execute("""
            SELECT id, pmc_id, pmid, doi, title, journal, pub_year,
                   LENGTH(full_text) as text_length
            FROM articles
            WHERE full_text IS NOT NULL
            LIMIT 1
        """)
        sample = cursor.fetchone()
        logger.info(f"Sample article - ID: {sample[0]}, PMC: {sample[1]}, Text length: {sample[7]:,} chars")

        # Prepare for batch processing
        cursor_name = f'article_cursor_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        # Use server-side cursor for efficient batch processing
        if INCLUDE_METADATA:
            # Fetch with metadata
            cursor_ss = conn.cursor(name=cursor_name, cursor_factory=RealDictCursor)
            cursor_ss.execute("""
                SELECT id, pmc_id, pmid, doi, title, journal, pub_year, full_text
                FROM articles
                WHERE full_text IS NOT NULL
                ORDER BY id
            """)
        else:
            # Fetch only ID and text for faster processing
            cursor_ss = conn.cursor(name=cursor_name)
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

        # Save extraction configuration
        config_file = os.path.join(output_dir, 'extraction_config.json')
        with open(config_file, 'w') as f:
            json.dump({
                'database': db_config['database'],
                'total_articles': total_articles,
                'include_metadata': INCLUDE_METADATA,
                'batch_size': BATCH_SIZE,
                'num_processes': NUM_PROCESSES,
                'start_time': datetime.now().isoformat()
            }, f, indent=2)

        # Process in batches
        batch_num = start_batch
        total_processed = len(processed_ids)
        start_time = time.time()

        # Create process pool
        pool = mp.Pool(NUM_PROCESSES)

        while True:
            # Fetch batch
            batch = cursor_ss.fetchmany(BATCH_SIZE)
            if not batch:
                break

            # Filter out already processed articles
            if INCLUDE_METADATA:
                batch = [record for record in batch if record['id'] not in processed_ids]
            else:
                batch = [(aid, text) for aid, text in batch if aid not in processed_ids]

            if not batch:
                continue

            # Create batch file paths and prepare chunks for parallel processing
            batch_files = []
            chunk_size = len(batch) // NUM_PROCESSES + 1
            chunks = [batch[i:i+chunk_size] for i in range(0, len(batch), chunk_size)]

            # Prepare worker arguments
            worker_args = []
            for i, chunk in enumerate(chunks):
                output_file = os.path.join(output_dir, f'batch_{batch_num:06d}_part_{i:02d}.jsonl')
                batch_files.append(output_file)
                worker_args.append((chunk, output_file, extractor, INCLUDE_METADATA))

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
            if INCLUDE_METADATA:
                for record in batch:
                    processed_ids.add(record['id'])
            else:
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

        # Save final summary
        summary_file = os.path.join(output_dir, 'extraction_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'total_articles_processed': total_processed,
                'total_time_seconds': total_time,
                'average_rate': total_processed / total_time if total_time > 0 else 0,
                'output_directory': output_dir,
                'error_log': ERROR_LOG_PATH,
                'completion_time': datetime.now().isoformat()
            }, f, indent=2)

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