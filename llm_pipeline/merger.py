"""
Merger Module for Combining LLM and Regex Extraction Results
Merges and deduplicates statistical findings from multiple sources
"""

import json
import os
import gzip
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import hashlib
import psycopg2
import psycopg2.extras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExtractionMerger:
    """Merge LLM extraction results with existing regex results"""
    
    def __init__(self, 
                 regex_dir: str = "D:/PubMed_Statistical_Analysis/extraction_20250923_210605/",
                 llm_output_dir: str = "llm_extractions",
                 db_config: Dict[str, Any] = None):
        """
        Initialize the merger
        
        Args:
            regex_dir: Directory containing regex extraction JSONL files
            llm_output_dir: Directory for LLM extraction results
            db_config: Database configuration
        """
        self.regex_dir = Path(regex_dir)
        self.llm_output_dir = Path(llm_output_dir)
        self.llm_output_dir.mkdir(exist_ok=True)
        
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        
        # Merged output directory
        self.merged_dir = Path("merged_extractions")
        self.merged_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.merge_stats = {
            'total_articles': 0,
            'regex_only': 0,
            'llm_only': 0,
            'both': 0,
            'total_statistics': defaultdict(int),
            'source_statistics': defaultdict(lambda: defaultdict(int))
        }
        
    def connect_db(self):
        """Connect to database"""
        if self.db_config:
            try:
                self.conn = psycopg2.connect(**self.db_config)
                self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                self._create_merge_tables()
                logger.info("Connected to merge database")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                
    def _create_merge_tables(self):
        """Create tables for merged results"""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS merged_statistics (
                    id SERIAL PRIMARY KEY,
                    pmc_id VARCHAR(20),
                    statistic_type VARCHAR(50),
                    value JSONB,
                    context TEXT,
                    location VARCHAR(50),
                    source VARCHAR(20),  -- 'regex', 'llm', or 'both'
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pmc_id, statistic_type, value)
                )
            """)
            
            self.cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_merged_pmc 
                ON merged_statistics(pmc_id)
            """)
            
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS merge_summary (
                    pmc_id VARCHAR(20) PRIMARY KEY,
                    regex_count INTEGER,
                    llm_count INTEGER,
                    merged_count INTEGER,
                    confidence_intervals INTEGER,
                    p_values INTEGER,
                    sample_sizes INTEGER,
                    effect_sizes INTEGER,
                    merge_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error creating merge tables: {e}")
            
    def load_regex_extractions(self, batch_size: int = 1000) -> Dict[str, Dict[str, Any]]:
        """
        Load regex extraction results from JSONL files
        
        Args:
            batch_size: Number of articles to load at once
            
        Returns:
            Dictionary mapping PMC IDs to extraction results
        """
        regex_extractions = {}
        files_processed = 0
        
        # Find all JSONL files (compressed or not)
        jsonl_files = list(self.regex_dir.glob("*.jsonl")) + list(self.regex_dir.glob("*.jsonl.gz"))
        
        logger.info(f"Found {len(jsonl_files)} regex extraction files")
        
        for file_path in jsonl_files[:10]:  # Process first 10 files for testing
            try:
                if file_path.suffix == '.gz':
                    open_func = gzip.open
                    mode = 'rt'
                else:
                    open_func = open
                    mode = 'r'
                    
                with open_func(file_path, mode) as f:
                    for line in f:
                        try:
                            article = json.loads(line)
                            pmc_id = article.get('pmc_id')
                            if pmc_id:
                                regex_extractions[pmc_id] = self._parse_regex_extraction(article)
                                
                                if len(regex_extractions) >= batch_size:
                                    yield regex_extractions
                                    regex_extractions = {}
                        except json.JSONDecodeError:
                            continue
                            
                files_processed += 1
                if files_processed % 100 == 0:
                    logger.info(f"Processed {files_processed} regex files")
                    
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
        # Yield remaining
        if regex_extractions:
            yield regex_extractions
            
    def _parse_regex_extraction(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Parse regex extraction into standard format"""
        extraction = {
            'pmc_id': article.get('pmc_id'),
            'confidence_intervals': [],
            'p_values': [],
            'sample_sizes': [],
            'effect_sizes': [],
            'other_statistics': []
        }
        
        # Extract confidence intervals
        if 'confidence_intervals' in article:
            for ci in article['confidence_intervals']:
                extraction['confidence_intervals'].append({
                    'lower': ci.get('lower'),
                    'upper': ci.get('upper'),
                    'level': ci.get('level', 95),
                    'context': ci.get('context', ''),
                    'source': 'regex'
                })
                
        # Extract p-values
        if 'p_values' in article:
            for p in article['p_values']:
                extraction['p_values'].append({
                    'value': p if isinstance(p, (int, float)) else p.get('value'),
                    'context': p.get('context', '') if isinstance(p, dict) else '',
                    'source': 'regex'
                })
                
        # Extract sample sizes
        if 'sample_sizes' in article:
            for n in article['sample_sizes']:
                extraction['sample_sizes'].append({
                    'value': n if isinstance(n, (int, float)) else n.get('value'),
                    'context': n.get('context', '') if isinstance(n, dict) else '',
                    'source': 'regex'
                })
                
        return extraction
        
    def load_llm_extractions(self) -> Dict[str, Dict[str, Any]]:
        """Load LLM extraction results"""
        llm_extractions = {}
        
        # Find all LLM result files
        llm_files = list(self.llm_output_dir.glob("*.json")) + list(self.llm_output_dir.glob("*.jsonl"))
        
        for file_path in llm_files:
            try:
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            article = json.loads(line)
                            pmc_id = article.get('pmc_id')
                            if pmc_id:
                                llm_extractions[pmc_id] = self._parse_llm_extraction(article)
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for article in data:
                                pmc_id = article.get('pmc_id')
                                if pmc_id:
                                    llm_extractions[pmc_id] = self._parse_llm_extraction(article)
                        else:
                            pmc_id = data.get('pmc_id')
                            if pmc_id:
                                llm_extractions[pmc_id] = self._parse_llm_extraction(data)
                                
            except Exception as e:
                logger.error(f"Error reading LLM file {file_path}: {e}")
                
        logger.info(f"Loaded {len(llm_extractions)} LLM extractions")
        return llm_extractions
        
    def _parse_llm_extraction(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM extraction into standard format"""
        extraction = {
            'pmc_id': article.get('pmc_id'),
            'confidence_intervals': [],
            'p_values': [],
            'sample_sizes': [],
            'effect_sizes': [],
            'other_statistics': []
        }
        
        # Handle findings format
        if 'findings' in article:
            for finding in article['findings']:
                stat_type = finding.get('type')
                if stat_type == 'confidence_interval':
                    extraction['confidence_intervals'].append({
                        **finding.get('value', {}),
                        'context': finding.get('context', ''),
                        'location': finding.get('location', 'text'),
                        'confidence': finding.get('confidence', 0.8),
                        'source': 'llm'
                    })
                elif stat_type == 'p_value':
                    extraction['p_values'].append({
                        'value': finding.get('value'),
                        'context': finding.get('context', ''),
                        'location': finding.get('location', 'text'),
                        'confidence': finding.get('confidence', 0.8),
                        'source': 'llm'
                    })
                elif stat_type == 'sample_size':
                    extraction['sample_sizes'].append({
                        'value': finding.get('value'),
                        'context': finding.get('context', ''),
                        'location': finding.get('location', 'text'),
                        'confidence': finding.get('confidence', 0.8),
                        'source': 'llm'
                    })
                elif stat_type == 'effect_size':
                    extraction['effect_sizes'].append({
                        **finding.get('value', {}) if isinstance(finding.get('value'), dict) else {'value': finding.get('value')},
                        'context': finding.get('context', ''),
                        'location': finding.get('location', 'text'),
                        'confidence': finding.get('confidence', 0.8),
                        'source': 'llm'
                    })
        else:
            # Direct format
            for key in ['confidence_intervals', 'p_values', 'sample_sizes', 'effect_sizes']:
                if key in article:
                    extraction[key] = article[key]
                    
        return extraction
        
    def merge_extractions(self, 
                         regex_extraction: Dict[str, Any],
                         llm_extraction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge regex and LLM extractions for a single article
        
        Args:
            regex_extraction: Regex extraction results
            llm_extraction: LLM extraction results
            
        Returns:
            Merged extraction results
        """
        pmc_id = regex_extraction.get('pmc_id') or llm_extraction.get('pmc_id')
        
        merged = {
            'pmc_id': pmc_id,
            'confidence_intervals': [],
            'p_values': [],
            'sample_sizes': [],
            'effect_sizes': [],
            'other_statistics': [],
            'metadata': {
                'merge_timestamp': datetime.now().isoformat(),
                'sources': []
            }
        }
        
        # Track what sources contributed
        if regex_extraction and any(regex_extraction.get(k) for k in ['confidence_intervals', 'p_values', 'sample_sizes']):
            merged['metadata']['sources'].append('regex')
        if llm_extraction and any(llm_extraction.get(k) for k in ['confidence_intervals', 'p_values', 'sample_sizes']):
            merged['metadata']['sources'].append('llm')
            
        # Merge each statistic type
        merged['confidence_intervals'] = self._merge_confidence_intervals(
            regex_extraction.get('confidence_intervals', []) if regex_extraction else [],
            llm_extraction.get('confidence_intervals', []) if llm_extraction else []
        )
        
        merged['p_values'] = self._merge_p_values(
            regex_extraction.get('p_values', []) if regex_extraction else [],
            llm_extraction.get('p_values', []) if llm_extraction else []
        )
        
        merged['sample_sizes'] = self._merge_sample_sizes(
            regex_extraction.get('sample_sizes', []) if regex_extraction else [],
            llm_extraction.get('sample_sizes', []) if llm_extraction else []
        )
        
        merged['effect_sizes'] = self._merge_effect_sizes(
            regex_extraction.get('effect_sizes', []) if regex_extraction else [],
            llm_extraction.get('effect_sizes', []) if llm_extraction else []
        )
        
        # Update statistics
        self._update_merge_statistics(merged, regex_extraction, llm_extraction)
        
        return merged
        
    def _merge_confidence_intervals(self, 
                                   regex_cis: List[Dict],
                                   llm_cis: List[Dict]) -> List[Dict]:
        """Merge and deduplicate confidence intervals"""
        merged = []
        seen = set()
        
        # Process LLM CIs first (usually more complete)
        for ci in llm_cis:
            ci_key = self._get_ci_key(ci)
            if ci_key and ci_key not in seen:
                ci['source'] = ci.get('source', 'llm')
                merged.append(ci)
                seen.add(ci_key)
                
        # Add regex CIs not found by LLM
        for ci in regex_cis:
            ci_key = self._get_ci_key(ci)
            if ci_key and ci_key not in seen:
                # Check for close matches
                if not self._has_close_match(ci, merged, 'ci'):
                    ci['source'] = 'regex'
                    merged.append(ci)
                    seen.add(ci_key)
                    
        return merged
        
    def _merge_p_values(self, 
                       regex_ps: List[Dict],
                       llm_ps: List[Dict]) -> List[Dict]:
        """Merge and deduplicate p-values"""
        merged = []
        seen = set()
        
        for p in llm_ps:
            p_key = self._get_p_value_key(p)
            if p_key and p_key not in seen:
                p['source'] = p.get('source', 'llm')
                merged.append(p)
                seen.add(p_key)
                
        for p in regex_ps:
            p_key = self._get_p_value_key(p)
            if p_key and p_key not in seen:
                if not self._has_close_match(p, merged, 'p_value'):
                    p['source'] = 'regex'
                    merged.append(p)
                    seen.add(p_key)
                    
        return merged
        
    def _merge_sample_sizes(self, 
                          regex_ns: List[Dict],
                          llm_ns: List[Dict]) -> List[Dict]:
        """Merge and deduplicate sample sizes"""
        merged = []
        seen = set()
        
        for n in llm_ns:
            n_key = self._get_sample_size_key(n)
            if n_key and n_key not in seen:
                n['source'] = n.get('source', 'llm')
                merged.append(n)
                seen.add(n_key)
                
        for n in regex_ns:
            n_key = self._get_sample_size_key(n)
            if n_key and n_key not in seen:
                n['source'] = 'regex'
                merged.append(n)
                seen.add(n_key)
                
        return merged
        
    def _merge_effect_sizes(self, 
                          regex_effects: List[Dict],
                          llm_effects: List[Dict]) -> List[Dict]:
        """Merge effect sizes"""
        # LLM typically better at effect sizes
        merged = llm_effects.copy()
        
        # Add unique regex findings
        for effect in regex_effects:
            if not self._has_close_match(effect, merged, 'effect'):
                effect['source'] = 'regex'
                merged.append(effect)
                
        return merged
        
    def _get_ci_key(self, ci: Dict) -> Optional[str]:
        """Generate unique key for CI"""
        try:
            lower = round(float(ci.get('lower', 0)), 3)
            upper = round(float(ci.get('upper', 0)), 3)
            return f"{lower}_{upper}"
        except:
            return None
            
    def _get_p_value_key(self, p: Dict) -> Optional[str]:
        """Generate unique key for p-value"""
        try:
            value = float(p.get('value', 0))
            return f"{value:.6f}"
        except:
            return None
            
    def _get_sample_size_key(self, n: Dict) -> Optional[str]:
        """Generate unique key for sample size"""
        try:
            value = int(float(n.get('value', 0)))
            return str(value)
        except:
            return None
            
    def _has_close_match(self, item: Dict, items: List[Dict], item_type: str) -> bool:
        """Check if item has a close match in list"""
        if item_type == 'ci':
            for existing in items:
                try:
                    if (abs(float(item.get('lower', 0)) - float(existing.get('lower', 0))) < 0.01 and
                        abs(float(item.get('upper', 0)) - float(existing.get('upper', 0))) < 0.01):
                        return True
                except:
                    continue
        elif item_type == 'p_value':
            for existing in items:
                try:
                    if abs(float(item.get('value', 0)) - float(existing.get('value', 0))) < 0.0001:
                        return True
                except:
                    continue
        elif item_type == 'effect':
            for existing in items:
                try:
                    if (item.get('type') == existing.get('type') and 
                        abs(float(item.get('value', 0)) - float(existing.get('value', 0))) < 0.01):
                        return True
                except:
                    continue
                    
        return False
        
    def _update_merge_statistics(self, 
                                merged: Dict[str, Any],
                                regex_extraction: Dict[str, Any],
                                llm_extraction: Dict[str, Any]):
        """Update merge statistics"""
        self.merge_stats['total_articles'] += 1
        
        if regex_extraction and not llm_extraction:
            self.merge_stats['regex_only'] += 1
        elif llm_extraction and not regex_extraction:
            self.merge_stats['llm_only'] += 1
        else:
            self.merge_stats['both'] += 1
            
        # Count statistics by type
        for stat_type in ['confidence_intervals', 'p_values', 'sample_sizes', 'effect_sizes']:
            count = len(merged.get(stat_type, []))
            self.merge_stats['total_statistics'][stat_type] += count
            
            # Track source contribution
            for item in merged.get(stat_type, []):
                source = item.get('source', 'unknown')
                self.merge_stats['source_statistics'][source][stat_type] += 1
                
    def save_merged_result(self, merged: Dict[str, Any], format: str = 'jsonl'):
        """Save merged result to file"""
        output_file = self.merged_dir / f"merged_{datetime.now():%Y%m%d}.{format}"
        
        if format == 'jsonl':
            with open(output_file, 'a') as f:
                f.write(json.dumps(merged) + '\n')
        else:
            # Save as regular JSON (append to list)
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            data.append(merged)
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
    def save_to_database(self, merged: Dict[str, Any]):
        """Save merged results to database"""
        if not self.cursor:
            return
            
        try:
            pmc_id = merged['pmc_id']
            
            # Save individual statistics
            for stat_type in ['confidence_intervals', 'p_values', 'sample_sizes', 'effect_sizes']:
                for item in merged.get(stat_type, []):
                    self.cursor.execute("""
                        INSERT INTO merged_statistics 
                        (pmc_id, statistic_type, value, context, location, source, confidence)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (pmc_id, statistic_type, value) DO UPDATE
                        SET context = EXCLUDED.context,
                            source = EXCLUDED.source
                    """, (
                        pmc_id,
                        stat_type.rstrip('s'),  # Remove plural
                        json.dumps(item),
                        item.get('context', ''),
                        item.get('location', 'text'),
                        item.get('source', 'unknown'),
                        item.get('confidence', 0.5)
                    ))
                    
            # Save summary
            self.cursor.execute("""
                INSERT INTO merge_summary 
                (pmc_id, regex_count, llm_count, merged_count,
                 confidence_intervals, p_values, sample_sizes, effect_sizes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (pmc_id) DO UPDATE
                SET merged_count = EXCLUDED.merged_count,
                    merge_date = CURRENT_TIMESTAMP
            """, (
                pmc_id,
                len([s for s in merged.get('metadata', {}).get('sources', []) if 'regex' in s]),
                len([s for s in merged.get('metadata', {}).get('sources', []) if 'llm' in s]),
                sum(len(merged.get(k, [])) for k in ['confidence_intervals', 'p_values', 'sample_sizes', 'effect_sizes']),
                len(merged.get('confidence_intervals', [])),
                len(merged.get('p_values', [])),
                len(merged.get('sample_sizes', [])),
                len(merged.get('effect_sizes', []))
            ))
            
            self.conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            self.conn.rollback()
            
    def process_all(self):
        """Process all extractions and merge them"""
        logger.info("Starting merge process...")
        
        # Load LLM extractions
        llm_extractions = self.load_llm_extractions()
        
        # Process regex extractions in batches
        articles_processed = 0
        for regex_batch in self.load_regex_extractions(batch_size=1000):
            for pmc_id, regex_extraction in regex_batch.items():
                llm_extraction = llm_extractions.get(pmc_id)
                
                # Merge
                merged = self.merge_extractions(regex_extraction, llm_extraction)
                
                # Save
                self.save_merged_result(merged)
                if self.cursor:
                    self.save_to_database(merged)
                    
                articles_processed += 1
                if articles_processed % 100 == 0:
                    logger.info(f"Processed {articles_processed} articles")
                    
        # Process any remaining LLM-only extractions
        for pmc_id, llm_extraction in llm_extractions.items():
            if pmc_id not in self.merge_stats:  # Not yet processed
                merged = self.merge_extractions(None, llm_extraction)
                self.save_merged_result(merged)
                if self.cursor:
                    self.save_to_database(merged)
                    
        logger.info("Merge process completed")
        self.print_statistics()
        
    def print_statistics(self):
        """Print merge statistics"""
        print("\n" + "="*60)
        print("MERGE STATISTICS")
        print("="*60)
        print(f"Total articles processed: {self.merge_stats['total_articles']:,}")
        print(f"  - Regex only: {self.merge_stats['regex_only']:,}")
        print(f"  - LLM only: {self.merge_stats['llm_only']:,}")
        print(f"  - Both sources: {self.merge_stats['both']:,}")
        print("\nTotal statistics extracted:")
        for stat_type, count in self.merge_stats['total_statistics'].items():
            print(f"  - {stat_type}: {count:,}")
        print("\nStatistics by source:")
        for source, stats in self.merge_stats['source_statistics'].items():
            print(f"  {source}:")
            for stat_type, count in stats.items():
                print(f"    - {stat_type}: {count:,}")
        print("="*60)


def test_merger():
    """Test the merger"""
    # Create test data
    regex_extraction = {
        'pmc_id': 'PMC123456',
        'confidence_intervals': [
            {'lower': 1.2, 'upper': 3.4, 'level': 95},
            {'lower': 0.5, 'upper': 0.8, 'level': 95}
        ],
        'p_values': [
            {'value': 0.023},
            {'value': 0.001}
        ],
        'sample_sizes': [
            {'value': 100},
            {'value': 250}
        ]
    }
    
    llm_extraction = {
        'pmc_id': 'PMC123456',
        'confidence_intervals': [
            {'lower': 1.2, 'upper': 3.4, 'level': 95, 'location': 'table'},
            {'lower': 2.1, 'upper': 4.5, 'level': 95, 'location': 'figure'}
        ],
        'p_values': [
            {'value': 0.023, 'location': 'text'},
            {'value': 0.456, 'location': 'table'}
        ],
        'sample_sizes': [
            {'value': 100},
            {'value': 175}
        ],
        'effect_sizes': [
            {'type': 'cohens_d', 'value': 0.85}
        ]
    }
    
    merger = ExtractionMerger(regex_dir="test_regex", llm_output_dir="test_llm")
    
    # Test merge
    merged = merger.merge_extractions(regex_extraction, llm_extraction)
    
    print(f"Merged result for {merged['pmc_id']}:")
    print(f"  CIs: {len(merged['confidence_intervals'])}")
    print(f"  P-values: {len(merged['p_values'])}")
    print(f"  Sample sizes: {len(merged['sample_sizes'])}")
    print(f"  Effect sizes: {len(merged['effect_sizes'])}")
    print(f"  Sources: {merged['metadata']['sources']}")
    
    # Save test result
    merger.save_merged_result(merged)
    print(f"\nSaved to {merger.merged_dir}")


if __name__ == "__main__":
    test_merger()
