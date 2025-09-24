"""
Intelligent Batch Processor for Articles
Groups similar articles and optimizes batch sizes for LLM processing
"""

import psycopg2
import psycopg2.extras
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import hashlib
import json
from datetime import datetime
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """Intelligent batching system for article processing"""
    
    def __init__(self, db_config: Dict[str, Any], batch_size_limits: Dict[str, int] = None):
        """
        Initialize the batch processor
        
        Args:
            db_config: Database connection configuration
            batch_size_limits: Min/max batch sizes by article length
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
        # Default batch size limits based on article length
        self.batch_size_limits = batch_size_limits or {
            'short': {'min': 20, 'max': 50, 'char_limit': 10000},
            'medium': {'min': 10, 'max': 25, 'char_limit': 50000},
            'long': {'min': 5, 'max': 10, 'char_limit': 100000},
            'very_long': {'min': 1, 'max': 5, 'char_limit': 200000}
        }
        
        self.processed_articles = set()
        self.batch_stats = defaultdict(int)
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
            
    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def get_article_batch(self, 
                         batch_size: int = 50,
                         offset: int = 0,
                         filter_processed: bool = True,
                         priority_journals: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get a batch of articles from the database
        
        Args:
            batch_size: Number of articles to retrieve
            offset: Offset for pagination
            filter_processed: Skip already processed articles
            priority_journals: List of journals to prioritize
            
        Returns:
            List of article dictionaries
        """
        query = """
            SELECT id, pmc_id, title, abstract, body, journal, pub_year,
                   LENGTH(COALESCE(body, '')) as body_length
            FROM articles
            WHERE 1=1
        """
        
        params = []
        
        if filter_processed and self.processed_articles:
            placeholders = ','.join(['%s'] * len(self.processed_articles))
            query += f" AND pmc_id NOT IN ({placeholders})"
            params.extend(list(self.processed_articles))
            
        if priority_journals:
            journal_placeholders = ','.join(['%s'] * len(priority_journals))
            query += f" AND journal IN ({journal_placeholders})"
            params.extend(priority_journals)
            
        query += """
            ORDER BY 
                CASE 
                    WHEN body LIKE '%Table%' THEN 1
                    WHEN body LIKE '%Figure%' THEN 2
                    WHEN body LIKE '%CI%' OR body LIKE '%confidence interval%' THEN 3
                    ELSE 4
                END,
                pub_year DESC
            LIMIT %s OFFSET %s
        """
        params.extend([batch_size, offset])
        
        try:
            self.cursor.execute(query, params)
            articles = [dict(row) for row in self.cursor.fetchall()]
            return articles
        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []
            
    def create_intelligent_batches(self, 
                                  articles: List[Dict[str, Any]],
                                  target_tokens: int = 8000) -> List[List[Dict[str, Any]]]:
        """
        Create intelligent batches based on article similarity and size
        
        Args:
            articles: List of articles to batch
            target_tokens: Target token count per batch (approximate)
            
        Returns:
            List of article batches
        """
        # Group articles by characteristics
        grouped = self._group_by_characteristics(articles)
        
        batches = []
        for group_key, group_articles in grouped.items():
            # Create batches within each group
            group_batches = self._create_size_optimized_batches(group_articles, target_tokens)
            batches.extend(group_batches)
            
        return batches
        
    def _group_by_characteristics(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Group articles by similar characteristics"""
        groups = defaultdict(list)
        
        for article in articles:
            # Determine article characteristics
            char_key = self._get_article_characteristics(article)
            groups[char_key].append(article)
            
        return groups
        
    def _get_article_characteristics(self, article: Dict[str, Any]) -> str:
        """Generate a characteristics key for grouping similar articles"""
        characteristics = []
        
        body = article.get('body', '') or ''
        abstract = article.get('abstract', '') or ''
        content = body + abstract
        
        # Check for statistical content patterns
        if 'Table' in content or 'TABLE' in content:
            characteristics.append('has_tables')
        if 'Figure' in content or 'FIGURE' in content:
            characteristics.append('has_figures')
        if re.search(r'CI[:=\s]|confidence interval', content, re.IGNORECASE):
            characteristics.append('has_ci')
        if re.search(r'p\s*[<>=]\s*0\.\d+', content, re.IGNORECASE):
            characteristics.append('has_pvalues')
        if re.search(r'n\s*=\s*\d+', content, re.IGNORECASE):
            characteristics.append('has_sample_size')
            
        # Add journal as a characteristic
        journal = article.get('journal', 'unknown')
        if journal:
            # Simplify journal name for grouping
            journal_key = journal.split()[0].lower() if journal else 'unknown'
            characteristics.append(f'journal_{journal_key}')
            
        # Add size category
        body_length = len(body)
        if body_length < 10000:
            characteristics.append('size_short')
        elif body_length < 50000:
            characteristics.append('size_medium')
        elif body_length < 100000:
            characteristics.append('size_long')
        else:
            characteristics.append('size_very_long')
            
        return '_'.join(sorted(characteristics))
        
    def _create_size_optimized_batches(self, 
                                      articles: List[Dict[str, Any]], 
                                      target_tokens: int) -> List[List[Dict[str, Any]]]:
        """Create batches optimized for size constraints"""
        batches = []
        current_batch = []
        current_tokens = 0
        
        # Sort articles by size for better packing
        sorted_articles = sorted(articles, key=lambda x: len(x.get('body', '') or ''))
        
        for article in sorted_articles:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            article_tokens = self._estimate_tokens(article)
            
            # Determine if we should start a new batch
            if current_batch and (current_tokens + article_tokens > target_tokens or 
                                 len(current_batch) >= self._get_max_batch_size(article)):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
                
            current_batch.append(article)
            current_tokens += article_tokens
            
        if current_batch:
            batches.append(current_batch)
            
        return batches
        
    def _estimate_tokens(self, article: Dict[str, Any]) -> int:
        """Estimate token count for an article"""
        text_length = len(article.get('title', '') or '')
        text_length += len(article.get('abstract', '') or '')
        text_length += len(article.get('body', '') or '')
        
        # Rough approximation: 1 token ≈ 4 characters
        return text_length // 4
        
    def _get_max_batch_size(self, article: Dict[str, Any]) -> int:
        """Get maximum batch size based on article length"""
        body_length = len(article.get('body', '') or '')
        
        if body_length < self.batch_size_limits['short']['char_limit']:
            return self.batch_size_limits['short']['max']
        elif body_length < self.batch_size_limits['medium']['char_limit']:
            return self.batch_size_limits['medium']['max']
        elif body_length < self.batch_size_limits['long']['char_limit']:
            return self.batch_size_limits['long']['max']
        else:
            return self.batch_size_limits['very_long']['max']
            
    def mark_processed(self, pmc_ids: List[str]):
        """Mark articles as processed"""
        self.processed_articles.update(pmc_ids)
        
        # Also update in database
        if self.conn and self.cursor:
            try:
                # Create processing status table if not exists
                self.cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_processing_status (
                        pmc_id VARCHAR(20) PRIMARY KEY,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        batch_id VARCHAR(50),
                        status VARCHAR(20) DEFAULT 'completed'
                    )
                """)
                
                # Insert processed records
                for pmc_id in pmc_ids:
                    self.cursor.execute("""
                        INSERT INTO llm_processing_status (pmc_id, status)
                        VALUES (%s, 'completed')
                        ON CONFLICT (pmc_id) DO UPDATE SET processed_at = CURRENT_TIMESTAMP
                    """, (pmc_id,))
                    
                self.conn.commit()
            except Exception as e:
                logger.error(f"Error marking articles as processed: {e}")
                self.conn.rollback()
                
    def load_processed_articles(self):
        """Load previously processed articles from database"""
        try:
            self.cursor.execute("""
                SELECT pmc_id FROM llm_processing_status WHERE status = 'completed'
            """)
            results = self.cursor.fetchall()
            self.processed_articles = set(row[0] for row in results)
            logger.info(f"Loaded {len(self.processed_articles)} processed articles")
        except Exception as e:
            logger.warning(f"Could not load processed articles (table may not exist): {e}")
            
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = {
            'total_processed': len(self.processed_articles),
            'batch_stats': dict(self.batch_stats)
        }
        
        # Get database stats
        try:
            self.cursor.execute("SELECT COUNT(*) FROM articles")
            stats['total_articles'] = self.cursor.fetchone()[0]
            
            self.cursor.execute("""
                SELECT COUNT(*) FROM articles 
                WHERE body LIKE '%Table%' OR body LIKE '%Figure%'
            """)
            stats['articles_with_tables_figures'] = self.cursor.fetchone()[0]
            
            stats['remaining'] = stats['total_articles'] - stats['total_processed']
            stats['completion_percentage'] = (
                stats['total_processed'] / stats['total_articles'] * 100
                if stats['total_articles'] > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            
        return stats
        
    def optimize_batch_strategy(self, performance_metrics: Dict[str, Any]):
        """Adjust batching strategy based on performance metrics"""
        # Analyze performance and adjust batch sizes
        if 'average_processing_time' in performance_metrics:
            avg_time = performance_metrics['average_processing_time']
            
            # If processing is too slow, reduce batch sizes
            if avg_time > 60:  # More than 60 seconds per batch
                for category in self.batch_size_limits:
                    self.batch_size_limits[category]['max'] = max(
                        1,
                        int(self.batch_size_limits[category]['max'] * 0.8)
                    )
                logger.info("Reduced batch sizes due to slow processing")
                
            # If processing is fast, increase batch sizes
            elif avg_time < 20:  # Less than 20 seconds per batch
                for category in self.batch_size_limits:
                    self.batch_size_limits[category]['max'] = min(
                        100,
                        int(self.batch_size_limits[category]['max'] * 1.2)
                    )
                logger.info("Increased batch sizes due to fast processing")
                
        # Adjust based on error rates
        if 'error_rate' in performance_metrics:
            error_rate = performance_metrics['error_rate']
            if error_rate > 0.1:  # More than 10% errors
                logger.warning(f"High error rate: {error_rate:.2%}. Consider reducing batch sizes.")


def test_batch_processor():
    """Test the batch processor"""
    db_config = {
        'dbname': 'pmc_fulltext',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'localhost',
        'port': 5432
    }
    
    processor = BatchProcessor(db_config)
    processor.connect()
    
    try:
        # Load processed articles
        processor.load_processed_articles()
        
        # Get a batch of articles
        articles = processor.get_article_batch(batch_size=100)
        print(f"Retrieved {len(articles)} articles")
        
        # Create intelligent batches
        batches = processor.create_intelligent_batches(articles)
        print(f"Created {len(batches)} batches")
        
        for i, batch in enumerate(batches[:3]):
            print(f"Batch {i+1}: {len(batch)} articles")
            for article in batch[:2]:
                print(f"  - {article['pmc_id']}: {article.get('title', 'No title')[:50]}...")
                
        # Get stats
        stats = processor.get_processing_stats()
        print(f"Processing stats: {stats}")
        
    finally:
        processor.disconnect()


if __name__ == "__main__":
    test_batch_processor()
