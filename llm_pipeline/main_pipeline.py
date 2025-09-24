"""
Main Pipeline Orchestrator
Coordinates all components for large-scale statistical extraction from PubMed articles
"""

import asyncio
import argparse
import logging
import sys
import signal
import time
import psutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import psycopg2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import all pipeline components
from batch_processor import BatchProcessor
from ollama_manager import OllamaManager, DEFAULT_INSTANCES
from extraction_engine import ExtractionEngine
from progress_tracker import ProgressTracker
from validator import Validator
from merger import ExtractionMerger
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MainPipeline:
    """Main orchestrator for the extraction pipeline"""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the pipeline with command-line arguments"""
        self.args = args
        self.config = config
        
        # Initialize components
        self.batch_processor = None
        self.ollama_manager = None
        self.extraction_engine = None
        self.progress_tracker = None
        self.validator = None
        self.merger = None
        
        # Control flags
        self.running = False
        self.paused = False
        self.shutdown_event = asyncio.Event()
        
        # Performance metrics
        self.performance_metrics = {
            'start_time': None,
            'articles_per_minute': 0,
            'average_processing_time': 0,
            'error_rate': 0,
            'memory_usage_mb': 0,
            'cpu_usage_percent': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.running = False
        self.shutdown_event.set()
        
    async def initialize(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            db_config=self.config.DATABASE_CONFIG,
            batch_size_limits=self.config.BATCH_SIZE_LIMITS
        )
        self.batch_processor.connect()
        self.batch_processor.load_processed_articles()
        
        # Initialize Ollama manager
        instances = self.config.OLLAMA_INSTANCES if not self.args.single_instance else [DEFAULT_INSTANCES[0]]
        self.ollama_manager = OllamaManager(instances)
        await self.ollama_manager.start()
        
        # Initialize extraction engine
        self.extraction_engine = ExtractionEngine()
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(
            checkpoint_dir=self.config.CHECKPOINT_DIR,
            db_config=self.config.DATABASE_CONFIG
        )
        self.progress_tracker.connect_db()
        
        # Initialize validator
        self.validator = Validator()
        
        # Initialize merger
        self.merger = ExtractionMerger(
            regex_dir=self.config.REGEX_EXTRACTION_DIR,
            llm_output_dir=self.config.LLM_OUTPUT_DIR,
            db_config=self.config.DATABASE_CONFIG
        )
        self.merger.connect_db()
        
        logger.info("All components initialized successfully")
        
    async def run(self):
        """Main pipeline execution loop"""
        self.running = True
        self.performance_metrics['start_time'] = datetime.now()
        
        # Get total article count
        stats = self.batch_processor.get_processing_stats()
        total_articles = stats['total_articles']
        
        # Start or resume run
        run_id = self.args.run_id or f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self.progress_tracker.start_run(
            run_id=run_id,
            total_articles=total_articles,
            resume=self.args.resume
        )
        
        logger.info(f"Starting pipeline run: {run_id}")
        logger.info(f"Total articles to process: {total_articles:,}")
        logger.info(f"Already processed: {len(self.batch_processor.processed_articles):,}")
        
        # Create tasks for parallel processing
        tasks = []
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_performance())
        tasks.append(monitor_task)
        
        # Start extraction workers
        num_workers = self.args.workers or self.config.NUM_WORKERS
        for i in range(num_workers):
            worker_task = asyncio.create_task(self._extraction_worker(i))
            tasks.append(worker_task)
            
        # Start batch feeder
        feeder_task = asyncio.create_task(self._batch_feeder())
        tasks.append(feeder_task)
        
        # Wait for completion or shutdown
        try:
            await self.shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.running = False
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
                
            # Wait for tasks to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cleanup
            await self.cleanup()
            
    async def _batch_feeder(self):
        """Feed batches to the processing queue"""
        offset = 0
        batch_queue = asyncio.Queue(maxsize=10)
        
        while self.running:
            if self.paused:
                await asyncio.sleep(1)
                continue
                
            try:
                # Get articles from database
                articles = self.batch_processor.get_article_batch(
                    batch_size=self.config.ARTICLES_PER_FETCH,
                    offset=offset,
                    filter_processed=True,
                    priority_journals=self.args.priority_journals
                )
                
                if not articles:
                    logger.info("No more articles to process")
                    break
                    
                # Create intelligent batches
                batches = self.batch_processor.create_intelligent_batches(
                    articles,
                    target_tokens=self.config.TARGET_TOKENS_PER_BATCH
                )
                
                # Queue batches for processing
                for batch in batches:
                    await batch_queue.put(batch)
                    
                offset += self.config.ARTICLES_PER_FETCH
                
            except Exception as e:
                logger.error(f"Error in batch feeder: {e}")
                await asyncio.sleep(5)
                
        # Signal completion
        await batch_queue.put(None)
        
    async def _extraction_worker(self, worker_id: int):
        """Worker to process article batches"""
        batch_queue = asyncio.Queue(maxsize=10)
        
        while self.running:
            try:
                # Get batch from queue
                batch = await asyncio.wait_for(batch_queue.get(), timeout=30)
                
                if batch is None:
                    break
                    
                # Process batch
                await self._process_batch(batch, worker_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
                
    async def _process_batch(self, batch: List[Dict[str, Any]], worker_id: int):
        """Process a single batch of articles"""
        batch_start_time = time.time()
        batch_ids = [article['pmc_id'] for article in batch]
        
        logger.info(f"Worker {worker_id}: Processing batch of {len(batch)} articles")
        
        try:
            # Prepare articles for LLM
            prepared_articles = []
            for article in batch:
                llm_input = self.extraction_engine.extract_statistics(article, use_llm=True)
                prepared_articles.append(llm_input)
                
            # Send to Ollama
            llm_results = await self.ollama_manager.process_batch(
                articles=prepared_articles,
                prompt_template=ExtractionEngine.EXTRACTION_PROMPT_TEMPLATE,
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                max_retries=3
            )
            
            # Parse LLM responses
            parsed_results = []
            for article, llm_response in zip(batch, llm_results):
                findings = self.extraction_engine.parse_llm_response(llm_response, article)
                
                # Validate findings
                validated_findings = self.extraction_engine.validate_findings(findings)
                
                # Convert to result format
                result = {
                    'pmc_id': article['pmc_id'],
                    'confidence_intervals': [],
                    'p_values': [],
                    'sample_sizes': [],
                    'effect_sizes': [],
                    'total_findings': len(validated_findings)
                }
                
                for finding in validated_findings:
                    if finding.type == 'confidence_interval':
                        result['confidence_intervals'].append(finding.to_dict())
                    elif finding.type == 'p_value':
                        result['p_values'].append(finding.to_dict())
                    elif finding.type == 'sample_size':
                        result['sample_sizes'].append(finding.to_dict())
                    elif finding.type == 'effect_size':
                        result['effect_sizes'].append(finding.to_dict())
                        
                parsed_results.append(result)
                
            # Validate batch
            validated_results, issues = self.validator.validate_batch(parsed_results)
            
            if issues:
                logger.warning(f"Validation issues in batch: {issues[:3]}")
                
            # Save results
            await self._save_results(validated_results)
            
            # Update progress
            batch_time = time.time() - batch_start_time
            self.progress_tracker.update_progress(batch_ids, validated_results, batch_time)
            
            # Mark as processed
            self.batch_processor.mark_processed(batch_ids)
            
            # Print progress periodically
            if worker_id == 0 and len(self.batch_processor.processed_articles) % 100 == 0:
                self.progress_tracker.print_progress()
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            self.progress_tracker.mark_failed(batch_ids, str(e))
            
    async def _save_results(self, results: List[Dict[str, Any]]):
        """Save extraction results"""
        output_file = Path(self.config.LLM_OUTPUT_DIR) / f"extractions_{datetime.now():%Y%m%d}.jsonl"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
                
    async def _monitor_performance(self):
        """Monitor system performance"""
        while self.running:
            try:
                # Get system metrics
                process = psutil.Process()
                self.performance_metrics['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
                self.performance_metrics['cpu_usage_percent'] = process.cpu_percent()
                
                # Calculate processing metrics
                stats = self.progress_tracker.get_statistics()
                self.performance_metrics['articles_per_minute'] = stats['articles_per_minute']
                self.performance_metrics['average_processing_time'] = stats['average_processing_time']
                
                # Check for performance issues
                if self.performance_metrics['memory_usage_mb'] > self.config.MAX_MEMORY_MB:
                    logger.warning(f"High memory usage: {self.performance_metrics['memory_usage_mb']:.1f} MB")
                    self.batch_processor.optimize_batch_strategy(self.performance_metrics)
                    
                if self.performance_metrics['cpu_usage_percent'] > 90:
                    logger.warning(f"High CPU usage: {self.performance_metrics['cpu_usage_percent']:.1f}%")
                    
                # Log metrics periodically
                if int(time.time()) % 300 == 0:  # Every 5 minutes
                    logger.info(f"Performance metrics: {self.performance_metrics}")
                    
                    # Get Ollama status
                    ollama_status = self.ollama_manager.get_status()
                    logger.info(f"Ollama instances status: {ollama_status}")
                    
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(30)
                
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up pipeline resources...")
        
        # Stop progress tracker
        if self.progress_tracker:
            self.progress_tracker.stop()
            
        # Stop Ollama manager
        if self.ollama_manager:
            await self.ollama_manager.stop()
            
        # Disconnect databases
        if self.batch_processor:
            self.batch_processor.disconnect()
            
        # Final merge if requested
        if self.args.merge_on_complete and self.merger:
            logger.info("Starting final merge process...")
            self.merger.process_all()
            
        logger.info("Cleanup completed")
        
    def pause(self):
        """Pause pipeline processing"""
        self.paused = True
        logger.info("Pipeline paused")
        
    def resume(self):
        """Resume pipeline processing"""
        self.paused = False
        logger.info("Pipeline resumed")
        
    async def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        status = {
            'running': self.running,
            'paused': self.paused,
            'progress': self.progress_tracker.get_statistics() if self.progress_tracker else {},
            'performance': self.performance_metrics,
            'ollama': self.ollama_manager.get_status() if self.ollama_manager else {}
        }
        return status


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='PubMed Statistical Extraction Pipeline - LLM Enhancement'
    )
    
    # Basic options
    parser.add_argument(
        '--run-id',
        type=str,
        help='Unique identifier for this run (default: auto-generated)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--start-from-beginning',
        action='store_true',
        help='Start fresh, ignoring any previous progress'
    )
    
    # Processing options
    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='Number of parallel extraction workers (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override default batch size'
    )
    parser.add_argument(
        '--single-instance',
        action='store_true',
        help='Use only one Ollama instance (for testing)'
    )
    
    # Article selection
    parser.add_argument(
        '--use-all-articles',
        action='store_true',
        help='Process all articles, not just those with low scores'
    )
    parser.add_argument(
        '--priority-journals',
        nargs='+',
        help='List of journals to prioritize'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of articles to process (for testing)'
    )
    
    # Output options
    parser.add_argument(
        '--merge-on-complete',
        action='store_true',
        help='Automatically merge with regex results when complete'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override default output directory'
    )
    
    # Performance options
    parser.add_argument(
        '--max-memory',
        type=int,
        help='Maximum memory usage in MB'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=300,
        help='Checkpoint save interval in seconds (default: 300)'
    )
    
    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run without actually processing (for testing)'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Override config if needed
    if args.output_dir:
        config.LLM_OUTPUT_DIR = args.output_dir
    if args.max_memory:
        config.MAX_MEMORY_MB = args.max_memory
    if args.checkpoint_interval:
        config.CHECKPOINT_INTERVAL = args.checkpoint_interval
        
    # Create pipeline
    pipeline = MainPipeline(args)
    
    try:
        # Initialize
        await pipeline.initialize()
        
        # Run pipeline
        await pipeline.run()
        
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    # Run the pipeline
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
