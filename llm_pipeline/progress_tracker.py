"""
Progress Tracker with Checkpoint and Resume Capabilities
Handles tracking, saving, and resuming pipeline progress
"""

import json
import os
import time
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
import psycopg2
import threading
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Track and persist progress of the extraction pipeline"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", db_config: Dict[str, Any] = None):
        """
        Initialize the progress tracker
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            db_config: Database configuration for persistent storage
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
        # Progress metrics
        self.start_time = None
        self.total_articles = 0
        self.processed_articles = set()
        self.failed_articles = set()
        self.current_batch = 0
        self.total_batches = 0
        
        # Performance tracking
        self.processing_times = deque(maxlen=1000)
        self.extraction_counts = {
            'confidence_intervals': 0,
            'p_values': 0,
            'sample_sizes': 0,
            'effect_sizes': 0,
            'total': 0
        }
        
        # Checkpoint management
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = 300  # Save every 5 minutes
        self.auto_save_thread = None
        self.stop_auto_save = threading.Event()
        
    def connect_db(self):
        """Connect to PostgreSQL database"""
        if self.db_config:
            try:
                self.conn = psycopg2.connect(**self.db_config)
                self.cursor = self.conn.cursor()
                self._create_progress_tables()
                logger.info("Connected to progress database")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                
    def _create_progress_tables(self):
        """Create tables for tracking progress"""
        try:
            # Main progress table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_extraction_progress (
                    id SERIAL PRIMARY KEY,
                    run_id VARCHAR(50) UNIQUE,
                    start_time TIMESTAMP,
                    last_update TIMESTAMP,
                    total_articles INTEGER,
                    processed_articles INTEGER,
                    failed_articles INTEGER,
                    current_batch INTEGER,
                    total_batches INTEGER,
                    status VARCHAR(20),
                    checkpoint_file VARCHAR(255)
                )
            """)
            
            # Detailed article processing status
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_article_status (
                    pmc_id VARCHAR(20) PRIMARY KEY,
                    processed_at TIMESTAMP,
                    status VARCHAR(20),
                    findings_count INTEGER,
                    confidence_intervals INTEGER,
                    p_values INTEGER,
                    sample_sizes INTEGER,
                    effect_sizes INTEGER,
                    processing_time FLOAT,
                    error_message TEXT
                )
            """)
            
            # Performance metrics table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_performance_metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    articles_per_minute FLOAT,
                    average_findings_per_article FLOAT,
                    error_rate FLOAT,
                    average_processing_time FLOAT,
                    memory_usage_mb FLOAT,
                    cpu_usage_percent FLOAT
                )
            """)
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error creating progress tables: {e}")
            
    def start_run(self, run_id: str, total_articles: int, resume: bool = False):
        """Start or resume a processing run"""
        self.run_id = run_id
        self.total_articles = total_articles
        
        if resume:
            checkpoint = self.load_latest_checkpoint()
            if checkpoint:
                self._restore_from_checkpoint(checkpoint)
                logger.info(f"Resumed run {run_id} from checkpoint")
            else:
                logger.info(f"No checkpoint found, starting fresh run {run_id}")
                self.start_time = datetime.now()
        else:
            self.start_time = datetime.now()
            self.processed_articles = set()
            self.failed_articles = set()
            self.current_batch = 0
            
        # Start auto-save thread
        self.start_auto_save()
        
        # Update database
        if self.cursor:
            try:
                self.cursor.execute("""
                    INSERT INTO llm_extraction_progress 
                    (run_id, start_time, total_articles, processed_articles, status)
                    VALUES (%s, %s, %s, %s, 'running')
                    ON CONFLICT (run_id) DO UPDATE
                    SET last_update = CURRENT_TIMESTAMP,
                        status = 'running'
                """, (run_id, self.start_time, total_articles, len(self.processed_articles)))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error updating database: {e}")
                
    def update_progress(self, 
                       batch_articles: List[str],
                       results: List[Dict[str, Any]],
                       batch_time: float):
        """Update progress after processing a batch"""
        # Update processed articles
        for article_id in batch_articles:
            if article_id not in self.failed_articles:
                self.processed_articles.add(article_id)
                
        # Update timing
        self.processing_times.append(batch_time)
        
        # Update extraction counts
        for result in results:
            if 'error' not in result:
                self.extraction_counts['total'] += 1
                if 'confidence_intervals' in result:
                    self.extraction_counts['confidence_intervals'] += len(result['confidence_intervals'])
                if 'p_values' in result:
                    self.extraction_counts['p_values'] += len(result['p_values'])
                if 'sample_sizes' in result:
                    self.extraction_counts['sample_sizes'] += len(result['sample_sizes'])
                if 'effect_sizes' in result:
                    self.extraction_counts['effect_sizes'] += len(result['effect_sizes'])
                    
        self.current_batch += 1
        
        # Update database
        if self.cursor:
            self._update_database(batch_articles, results, batch_time)
            
        # Check if checkpoint needed
        if time.time() - self.last_checkpoint_time > self.checkpoint_interval:
            self.save_checkpoint()
            
    def _update_database(self, 
                        batch_articles: List[str],
                        results: List[Dict[str, Any]],
                        batch_time: float):
        """Update database with batch results"""
        try:
            # Update individual article status
            for article_id, result in zip(batch_articles, results):
                if 'error' in result:
                    status = 'failed'
                    error_msg = result.get('error', '')
                else:
                    status = 'completed'
                    error_msg = None
                    
                self.cursor.execute("""
                    INSERT INTO llm_article_status 
                    (pmc_id, processed_at, status, findings_count, 
                     confidence_intervals, p_values, sample_sizes, 
                     effect_sizes, processing_time, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (pmc_id) DO UPDATE
                    SET processed_at = EXCLUDED.processed_at,
                        status = EXCLUDED.status,
                        findings_count = EXCLUDED.findings_count,
                        error_message = EXCLUDED.error_message
                """, (
                    article_id,
                    datetime.now(),
                    status,
                    result.get('total_findings', 0),
                    len(result.get('confidence_intervals', [])),
                    len(result.get('p_values', [])),
                    len(result.get('sample_sizes', [])),
                    len(result.get('effect_sizes', [])),
                    batch_time / len(batch_articles),
                    error_msg
                ))
                
            # Update main progress
            self.cursor.execute("""
                UPDATE llm_extraction_progress
                SET processed_articles = %s,
                    failed_articles = %s,
                    current_batch = %s,
                    last_update = CURRENT_TIMESTAMP
                WHERE run_id = %s
            """, (
                len(self.processed_articles),
                len(self.failed_articles),
                self.current_batch,
                self.run_id
            ))
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            self.conn.rollback()
            
    def mark_failed(self, article_ids: List[str], error: str):
        """Mark articles as failed"""
        self.failed_articles.update(article_ids)
        
        if self.cursor:
            try:
                for article_id in article_ids:
                    self.cursor.execute("""
                        INSERT INTO llm_article_status 
                        (pmc_id, processed_at, status, error_message)
                        VALUES (%s, %s, 'failed', %s)
                        ON CONFLICT (pmc_id) DO UPDATE
                        SET status = 'failed',
                            error_message = EXCLUDED.error_message
                    """, (article_id, datetime.now(), error))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error marking failed articles: {e}")
                
    def save_checkpoint(self, force: bool = False):
        """Save current progress to checkpoint file"""
        if not force and time.time() - self.last_checkpoint_time < 60:
            return  # Don't save too frequently
            
        checkpoint_data = {
            'run_id': getattr(self, 'run_id', None),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'total_articles': self.total_articles,
            'processed_articles': list(self.processed_articles),
            'failed_articles': list(self.failed_articles),
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'extraction_counts': self.extraction_counts,
            'processing_times': list(self.processing_times),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{self.run_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        # Also save as latest
        latest_file = self.checkpoint_dir / f"latest_{self.run_id}.json"
        with open(latest_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        self.last_checkpoint_time = time.time()
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Update database
        if self.cursor:
            try:
                self.cursor.execute("""
                    UPDATE llm_extraction_progress
                    SET checkpoint_file = %s
                    WHERE run_id = %s
                """, (str(checkpoint_file), self.run_id))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error updating checkpoint in database: {e}")
                
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint"""
        latest_file = self.checkpoint_dir / f"latest_{self.run_id}.json"
        if latest_file.exists():
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        # Look for any checkpoint for this run
        checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_{self.run_id}_*.json"))
        if checkpoints:
            checkpoints.sort()
            with open(checkpoints[-1], 'r') as f:
                return json.load(f)
                
        return None
        
    def _restore_from_checkpoint(self, checkpoint: Dict[str, Any]):
        """Restore state from checkpoint"""
        self.start_time = datetime.fromisoformat(checkpoint['start_time'])
        self.total_articles = checkpoint['total_articles']
        self.processed_articles = set(checkpoint['processed_articles'])
        self.failed_articles = set(checkpoint['failed_articles'])
        self.current_batch = checkpoint['current_batch']
        self.total_batches = checkpoint['total_batches']
        self.extraction_counts = checkpoint['extraction_counts']
        self.processing_times = deque(checkpoint['processing_times'], maxlen=1000)
        
    def get_eta(self) -> Optional[datetime]:
        """Calculate estimated time of completion"""
        if not self.processing_times or not self.processed_articles:
            return None
            
        avg_time = sum(self.processing_times) / len(self.processing_times)
        articles_per_batch = max(1, len(self.processed_articles) / max(1, self.current_batch))
        remaining_articles = self.total_articles - len(self.processed_articles)
        remaining_batches = remaining_articles / articles_per_batch
        remaining_seconds = remaining_batches * avg_time
        
        return datetime.now() + timedelta(seconds=remaining_seconds)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        elapsed_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        processed_count = len(self.processed_articles)
        
        stats = {
            'run_id': getattr(self, 'run_id', 'N/A'),
            'status': 'running' if processed_count < self.total_articles else 'completed',
            'total_articles': self.total_articles,
            'processed': processed_count,
            'failed': len(self.failed_articles),
            'remaining': self.total_articles - processed_count,
            'progress_percentage': (processed_count / self.total_articles * 100) if self.total_articles else 0,
            'current_batch': self.current_batch,
            'elapsed_time': str(timedelta(seconds=int(elapsed_time))),
            'articles_per_minute': (processed_count / (elapsed_time / 60)) if elapsed_time > 0 else 0,
            'extraction_counts': self.extraction_counts,
            'eta': self.get_eta().isoformat() if self.get_eta() else 'N/A',
            'average_processing_time': sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        }
        
        return stats
        
    def print_progress(self):
        """Print formatted progress update"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print(f"EXTRACTION PROGRESS - Run: {stats['run_id']}")
        print("="*60)
        print(f"Status: {stats['status']}")
        print(f"Progress: {stats['processed']:,}/{stats['total_articles']:,} ({stats['progress_percentage']:.1f}%)")
        print(f"Failed: {stats['failed']:,}")
        print(f"Current Batch: {stats['current_batch']}")
        print(f"Elapsed Time: {stats['elapsed_time']}")
        print(f"Speed: {stats['articles_per_minute']:.1f} articles/minute")
        print(f"ETA: {stats['eta']}")
        print("\nExtraction Counts:")
        for key, value in stats['extraction_counts'].items():
            print(f"  {key}: {value:,}")
        print("="*60)
        
    def start_auto_save(self):
        """Start automatic checkpoint saving"""
        self.stop_auto_save.clear()
        self.auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.auto_save_thread.start()
        
    def _auto_save_loop(self):
        """Background thread for automatic checkpointing"""
        while not self.stop_auto_save.is_set():
            time.sleep(self.checkpoint_interval)
            if not self.stop_auto_save.is_set():
                self.save_checkpoint()
                
    def stop(self):
        """Stop the progress tracker and save final checkpoint"""
        self.stop_auto_save.set()
        if self.auto_save_thread:
            self.auto_save_thread.join(timeout=5)
            
        self.save_checkpoint(force=True)
        
        # Update database status
        if self.cursor:
            try:
                self.cursor.execute("""
                    UPDATE llm_extraction_progress
                    SET status = %s,
                        last_update = CURRENT_TIMESTAMP
                    WHERE run_id = %s
                """, ('completed' if len(self.processed_articles) >= self.total_articles else 'stopped', 
                      self.run_id))
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error updating final status: {e}")
                
        if self.conn:
            self.conn.close()
            
    def get_unprocessed_articles(self) -> Set[str]:
        """Get list of articles not yet processed"""
        if self.cursor:
            try:
                self.cursor.execute("""
                    SELECT pmc_id FROM articles
                    WHERE pmc_id NOT IN (
                        SELECT pmc_id FROM llm_article_status
                        WHERE status = 'completed'
                    )
                """)
                return set(row[0] for row in self.cursor.fetchall())
            except Exception as e:
                logger.error(f"Error getting unprocessed articles: {e}")
                
        return set()


def test_progress_tracker():
    """Test the progress tracker"""
    tracker = ProgressTracker(checkpoint_dir="test_checkpoints")
    
    # Start a run
    tracker.start_run("test_run_001", total_articles=1000)
    
    # Simulate processing
    for batch in range(5):
        batch_articles = [f"PMC{i}" for i in range(batch*10, (batch+1)*10)]
        results = [{'total_findings': 5, 'confidence_intervals': [1,2,3]} for _ in batch_articles]
        tracker.update_progress(batch_articles, results, batch_time=15.5)
        
        tracker.print_progress()
        time.sleep(1)
        
    # Save checkpoint
    tracker.save_checkpoint(force=True)
    
    # Test resume
    tracker2 = ProgressTracker(checkpoint_dir="test_checkpoints")
    tracker2.start_run("test_run_001", total_articles=1000, resume=True)
    tracker2.print_progress()
    
    tracker.stop()
    tracker2.stop()


if __name__ == "__main__":
    test_progress_tracker()
