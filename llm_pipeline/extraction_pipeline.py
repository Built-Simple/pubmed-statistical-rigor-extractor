"""
Main LLM Extraction Pipeline for ALL PubMed Articles
Targets: 4.5M articles with focus on missed statistics in tables/figures
Author: PubMed Statistical Rigor Extraction System
Date: December 2024
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib
import gzip

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from config import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_CONFIG['log_dir'] / f'extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LLMExtractionPipeline:
    """
    Complete extraction pipeline for finding missed statistics in all PubMed articles.
    Focuses on tables, figures, and non-standard formats that regex missed.
    """
    
    def __init__(self):
        self.db_pool = self._create_db_pool()
        self.checkpoints = self._load_checkpoints()
        self.stats_buffer = []
        self.processed_count = 0
        self.missed_stats_count = 0
        self.extraction_stats = {
            'total_processed': 0,
            'new_cis_found': 0,
            'new_pvalues_found': 0,
            'new_effect_sizes_found': 0,
            'articles_with_tables': 0,
            'articles_with_figures': 0,
            'extraction_failures': 0,
        }
        self.ollama_sessions = {}
        
    def _create_db_pool(self) -> ThreadedConnectionPool:
        """Create a connection pool for database operations."""
        return ThreadedConnectionPool(
            minconn=2,
            maxconn=DB_CONFIG['pool_size'],
            host=DB_CONFIG['host'],
            database=DB_CONFIG['database'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            port=DB_CONFIG['port']
        )
    
    def _load_checkpoints(self) -> Dict:
        """Load checkpoint data to resume processing."""
        checkpoint_file = OUTPUT_CONFIG['checkpoint_dir'] / 'latest_checkpoint.json'
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return {'last_article_id': 0, 'processed_ids': set()}
    
    def _save_checkpoint(self):
        """Save current progress for resume capability."""
        checkpoint_file = OUTPUT_CONFIG['checkpoint_dir'] / 'latest_checkpoint.json'
        checkpoint_data = {
            'last_article_id': self.processed_count,
            'timestamp': datetime.now().isoformat(),
            'stats': self.extraction_stats,
            'processed_ids': list(self.checkpoints.get('processed_ids', set()))[-10000:]  # Keep last 10k
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    async def _init_ollama_sessions(self):
        """Initialize async sessions for all Ollama instances."""
        for instance in OLLAMA_INSTANCES:
            self.ollama_sessions[instance['url']] = aiohttp.ClientSession()
    
    async def _close_ollama_sessions(self):
        """Close all Ollama sessions."""
        for session in self.ollama_sessions.values():
            await session.close()
    
    def get_articles_batch(self, batch_size: int = 100, offset: int = 0) -> List[Dict]:
        """
        Fetch a batch of articles from the database.
        Prioritizes articles likely to have missed statistics.
        """
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Priority: Articles with high scores but no CIs detected
                query = """
                    WITH article_stats AS (
                        SELECT 
                            a.id,
                            a.pmc_id,
                            a.title,
                            a.abstract,
                            a.body,
                            a.journal,
                            a.pub_year,
                            COALESCE(
                                (SELECT rigor_score FROM extracted_stats WHERE article_id = a.id),
                                0
                            ) as current_score,
                            CASE 
                                WHEN a.body LIKE '%Table%' THEN 1 
                                ELSE 0 
                            END as has_tables,
                            CASE 
                                WHEN a.body LIKE '%Figure%' OR a.body LIKE '%Fig%' THEN 1 
                                ELSE 0 
                            END as has_figures,
                            CASE
                                WHEN EXISTS (
                                    SELECT 1 FROM extracted_stats 
                                    WHERE article_id = a.id 
                                    AND confidence_intervals IS NOT NULL
                                    AND array_length(confidence_intervals, 1) > 0
                                ) THEN 1 
                                ELSE 0
                            END as has_cis
                        FROM articles a
                        WHERE a.id > %s
                    )
                    SELECT *
                    FROM article_stats
                    ORDER BY 
                        -- Prioritize high-scoring articles without CIs
                        CASE 
                            WHEN current_score >= 50 AND has_cis = 0 THEN 0
                            WHEN current_score >= 30 AND has_cis = 0 AND (has_tables = 1 OR has_figures = 1) THEN 1
                            WHEN has_tables = 1 OR has_figures = 1 THEN 2
                            ELSE 3
                        END,
                        id
                    LIMIT %s
                """
                
                cursor.execute(query, (self.checkpoints.get('last_article_id', 0), batch_size))
                return cursor.fetchall()
        finally:
            self.db_pool.putconn(conn)
    
    async def extract_with_llm(self, article: Dict, instance_config: Dict, prompt_type: str = 'comprehensive') -> Dict:
        """
        Extract statistics from an article using LLM.
        
        Args:
            article: Article data including text
            instance_config: Ollama instance configuration
            prompt_type: Type of extraction prompt to use
        
        Returns:
            Extracted statistics dictionary
        """
        session = self.ollama_sessions[instance_config['url']]
        
        # Prepare the prompt
        text = f"{article.get('title', '')}\n{article.get('abstract', '')}\n{article.get('body', '')}"
        
        # Truncate if too long
        if len(text) > PROCESSING_CONFIG['context_window'] * 3:  # Rough character estimate
            text = text[:PROCESSING_CONFIG['context_window'] * 3]
        
        prompt = EXTRACTION_PROMPTS[prompt_type].format(text=text)
        
        try:
            async with session.post(
                f"{instance_config['url']}/api/generate",
                json={
                    'model': instance_config['model'],
                    'prompt': prompt,
                    'temperature': PROCESSING_CONFIG['temperature'],
                    'stream': False,
                    'options': {
                        'num_ctx': PROCESSING_CONFIG['context_window'],
                        'num_predict': 4096,
                    }
                },
                timeout=aiohttp.ClientTimeout(total=PROCESSING_CONFIG['timeout'])
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    
                    # Parse the response
                    try:
                        extracted = json.loads(result['response'])
                        
                        # Add metadata
                        extracted['article_id'] = article['id']
                        extracted['extraction_timestamp'] = datetime.now().isoformat()
                        extracted['model_used'] = instance_config['model']
                        extracted['prompt_type'] = prompt_type
                        
                        # Track statistics
                        if extracted.get('confidence_intervals'):
                            self.extraction_stats['new_cis_found'] += len(extracted['confidence_intervals'])
                        if extracted.get('tables_mentioned'):
                            self.extraction_stats['articles_with_tables'] += 1
                        if extracted.get('figures_mentioned'):
                            self.extraction_stats['articles_with_figures'] += 1
                        
                        return extracted
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON for article {article['id']}")
                        # Try to extract what we can from raw text
                        return self._parse_raw_response(result['response'], article['id'])
                else:
                    logger.error(f"LLM request failed with status {response.status}")
                    self.extraction_stats['extraction_failures'] += 1
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for article {article['id']}")
            self.extraction_stats['extraction_failures'] += 1
            return None
        except Exception as e:
            logger.error(f"Error extracting from article {article['id']}: {e}")
            self.extraction_stats['extraction_failures'] += 1
            return None
    
    def _parse_raw_response(self, response_text: str, article_id: int) -> Dict:
        """
        Fallback parser for non-JSON responses.
        Attempts to extract statistics from raw text.
        """
        extracted = {
            'article_id': article_id,
            'confidence_intervals': [],
            'p_values': [],
            'sample_sizes': [],
            'extraction_timestamp': datetime.now().isoformat(),
            'parse_method': 'fallback'
        }
        
        # Use the missed patterns to extract from raw response
        for pattern_type, patterns in MISSED_PATTERNS.items():
            for pattern in patterns:
                import re
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    if 'ci' in pattern_type.lower() or 'interval' in pattern_type.lower():
                        extracted['confidence_intervals'].extend(matches)
                    # Add other pattern types as needed
        
        return extracted
    
    async def process_article_batch(self, articles: List[Dict], instance_configs: List[Dict]):
        """
        Process a batch of articles in parallel across multiple LLM instances.
        """
        tasks = []
        
        # Distribute articles across instances
        for i, article in enumerate(articles):
            instance = instance_configs[i % len(instance_configs)]
            
            # Choose prompt based on article characteristics
            if article.get('has_tables') or article.get('has_figures'):
                prompt_type = 'table_focused' if article.get('has_tables') else 'comprehensive'
            elif article.get('current_score', 0) >= 50 and not article.get('has_cis'):
                prompt_type = 'ci_focused'
            else:
                prompt_type = 'comprehensive'
            
            task = self.extract_with_llm(article, instance, prompt_type)
            tasks.append(task)
        
        # Process all articles in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for article, result in zip(articles, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process article {article['id']}: {result}")
                continue
            
            if result:
                await self.save_extraction(result)
                self.processed_count += 1
                
                # Update checkpoint periodically
                if self.processed_count % PROCESSING_CONFIG['checkpoint_interval'] == 0:
                    self._save_checkpoint()
                    logger.info(f"Checkpoint saved at {self.processed_count} articles")
    
    async def save_extraction(self, extraction_data: Dict):
        """
        Save extracted statistics to database and file.
        """
        # Buffer extractions for batch saving
        self.stats_buffer.append(extraction_data)
        
        if len(self.stats_buffer) >= 100:
            await self._flush_buffer()
    
    async def _flush_buffer(self):
        """Flush the statistics buffer to database and file."""
        if not self.stats_buffer:
            return
        
        # Save to database
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                for stats in self.stats_buffer:
                    # Update or insert statistics
                    query = """
                        INSERT INTO llm_extracted_stats (
                            article_id, confidence_intervals, p_values, sample_sizes,
                            effect_sizes, statistical_tests, power_analysis, corrections,
                            tables_found, figures_found, extraction_model, extraction_timestamp
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (article_id) DO UPDATE SET
                            confidence_intervals = EXCLUDED.confidence_intervals,
                            p_values = EXCLUDED.p_values,
                            sample_sizes = EXCLUDED.sample_sizes,
                            effect_sizes = EXCLUDED.effect_sizes,
                            statistical_tests = EXCLUDED.statistical_tests,
                            power_analysis = EXCLUDED.power_analysis,
                            corrections = EXCLUDED.corrections,
                            tables_found = EXCLUDED.tables_found,
                            figures_found = EXCLUDED.figures_found,
                            extraction_model = EXCLUDED.extraction_model,
                            extraction_timestamp = EXCLUDED.extraction_timestamp,
                            updated_at = NOW()
                    """
                    
                    cursor.execute(query, (
                        stats.get('article_id'),
                        json.dumps(stats.get('confidence_intervals', [])),
                        json.dumps(stats.get('p_values', [])),
                        json.dumps(stats.get('sample_sizes', [])),
                        json.dumps(stats.get('effect_sizes', [])),
                        json.dumps(stats.get('statistical_tests', [])),
                        json.dumps(stats.get('power_analysis', [])),
                        json.dumps(stats.get('corrections', [])),
                        stats.get('tables_mentioned', False),
                        stats.get('figures_mentioned', False),
                        stats.get('model_used', 'unknown'),
                        stats.get('extraction_timestamp')
                    ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Database save error: {e}")
            conn.rollback()
        finally:
            self.db_pool.putconn(conn)
        
        # Save to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = OUTPUT_CONFIG['base_dir'] / f'extracted_batch_{timestamp}.jsonl.gz'
        
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            for stats in self.stats_buffer:
                f.write(json.dumps(stats) + '\n')
        
        logger.info(f"Saved {len(self.stats_buffer)} extractions to {output_file}")
        self.stats_buffer = []
    
    async def run_extraction(self, max_articles: Optional[int] = None):
        """
        Main extraction loop - process all articles.
        
        Args:
            max_articles: Maximum number of articles to process (None for all)
        """
        await self._init_ollama_sessions()
        
        # Get active Ollama instances
        active_instances = [inst for inst in OLLAMA_INSTANCES if inst.get('active', True)]
        
        total_articles = max_articles or TARGET_CONFIG['total_articles']
        batch_size = PROCESSING_CONFIG['batch_size']
        
        logger.info(f"Starting extraction of {total_articles} articles")
        logger.info(f"Using {len(active_instances)} Ollama instances")
        
        progress_bar = tqdm(total=total_articles, desc="Processing articles")
        
        try:
            while self.processed_count < total_articles:
                # Get next batch
                articles = self.get_articles_batch(batch_size)
                
                if not articles:
                    logger.info("No more articles to process")
                    break
                
                # Process batch
                await self.process_article_batch(articles, active_instances)
                
                # Update progress
                progress_bar.update(len(articles))
                
                # Log statistics periodically
                if self.processed_count % 1000 == 0:
                    self._log_statistics()
            
            # Final flush
            await self._flush_buffer()
            
        finally:
            progress_bar.close()
            await self._close_ollama_sessions()
            self._save_checkpoint()
            self._log_statistics()
    
    def _log_statistics(self):
        """Log current extraction statistics."""
        logger.info("=" * 50)
        logger.info(f"Extraction Statistics:")
        logger.info(f"  Articles processed: {self.processed_count}")
        logger.info(f"  New CIs found: {self.extraction_stats['new_cis_found']}")
        logger.info(f"  New p-values found: {self.extraction_stats['new_pvalues_found']}")
        logger.info(f"  Articles with tables: {self.extraction_stats['articles_with_tables']}")
        logger.info(f"  Articles with figures: {self.extraction_stats['articles_with_figures']}")
        logger.info(f"  Extraction failures: {self.extraction_stats['extraction_failures']}")
        
        if self.processed_count > 0:
            ci_rate = self.extraction_stats['new_cis_found'] / self.processed_count
            logger.info(f"  Average CIs per article: {ci_rate:.2f}")
        logger.info("=" * 50)


def create_llm_stats_table():
    """Create the table for LLM-extracted statistics if it doesn't exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS llm_extracted_stats (
        id SERIAL PRIMARY KEY,
        article_id INTEGER UNIQUE REFERENCES articles(id),
        confidence_intervals JSONB,
        p_values JSONB,
        sample_sizes JSONB,
        effect_sizes JSONB,
        statistical_tests JSONB,
        power_analysis JSONB,
        corrections JSONB,
        tables_found BOOLEAN DEFAULT FALSE,
        figures_found BOOLEAN DEFAULT FALSE,
        extraction_model VARCHAR(100),
        extraction_timestamp TIMESTAMP,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_llm_stats_article_id ON llm_extracted_stats(article_id);
    CREATE INDEX IF NOT EXISTS idx_llm_stats_tables ON llm_extracted_stats(tables_found);
    CREATE INDEX IF NOT EXISTS idx_llm_stats_figures ON llm_extracted_stats(figures_found);
    """
    
    try:
        cursor.execute(create_table_query)
        conn.commit()
        logger.info("LLM stats table created/verified successfully")
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()


async def main():
    """Main entry point for the extraction pipeline."""
    
    # Create necessary tables
    create_llm_stats_table()
    
    # Initialize pipeline
    pipeline = LLMExtractionPipeline()
    
    # Run extraction
    # For testing, process first 1000 articles
    # For production, set to None to process all
    await pipeline.run_extraction(max_articles=1000)  # Change to None for full run
    
    logger.info("Extraction complete!")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
