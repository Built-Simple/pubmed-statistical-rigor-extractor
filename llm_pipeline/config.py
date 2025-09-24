"""
Configuration file for PubMed Statistical Extraction Pipeline
"""

import os
from pathlib import Path

# Database Configuration
DATABASE_CONFIG = {
    'dbname': 'pmc_fulltext',
    'user': 'postgres',  # Update with your username
    'password': 'your_password',  # Update with your password
    'host': 'localhost',
    'port': 5432
}

# Ollama Configuration
OLLAMA_INSTANCES = [
    # First machine (104GB VRAM) - Main machine
    {'host': 'localhost', 'port': 11434, 'name': 'ollama-main-1'},
    {'host': 'localhost', 'port': 11435, 'name': 'ollama-main-2'},
    {'host': 'localhost', 'port': 11436, 'name': 'ollama-main-3'},
    
    # Second machine (48GB VRAM) - Update with actual IP
    # {'host': '192.168.1.100', 'port': 11437, 'name': 'ollama-secondary-1'},
    # {'host': '192.168.1.100', 'port': 11438, 'name': 'ollama-secondary-2'},
]

# LLM Model Configuration
LLM_MODEL = "qwen2.5:72b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 4096

# Batch Processing Configuration
BATCH_SIZE_LIMITS = {
    'short': {'min': 20, 'max': 50, 'char_limit': 10000},
    'medium': {'min': 10, 'max': 25, 'char_limit': 50000},
    'long': {'min': 5, 'max': 10, 'char_limit': 100000},
    'very_long': {'min': 1, 'max': 5, 'char_limit': 200000}
}

# Target tokens per batch (for optimal LLM processing)
TARGET_TOKENS_PER_BATCH = 8000

# Number of articles to fetch from database at once
ARTICLES_PER_FETCH = 500

# Processing Configuration
NUM_WORKERS = 3  # Number of parallel extraction workers
MAX_RETRIES = 3
RETRY_DELAY = 2  # Seconds

# Directory Configuration
BASE_DIR = Path("C:/Users/neely/pubmed-statistical-rigor-extractor/llm_pipeline")
CHECKPOINT_DIR = str(BASE_DIR / "checkpoints")
LLM_OUTPUT_DIR = str(BASE_DIR / "llm_extractions")
MERGED_OUTPUT_DIR = str(BASE_DIR / "merged_extractions")

# Existing regex extraction directory
REGEX_EXTRACTION_DIR = "D:/PubMed_Statistical_Analysis/extraction_20250923_210605/"

# Performance Configuration
MAX_MEMORY_MB = 64000  # 64GB max memory usage
MAX_CPU_PERCENT = 80
CHECKPOINT_INTERVAL = 300  # Save checkpoint every 5 minutes

# Monitoring Configuration
MONITORING_INTERVAL = 60  # Check performance every minute
PROGRESS_PRINT_INTERVAL = 100  # Print progress every 100 articles

# Validation Configuration
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.5,
    'low': 0.2
}

# Enhanced Extraction Prompts
EXTRACTION_PROMPTS = {
    'main': """
You are a statistical extraction expert. Extract ALL statistical information from these articles, especially from tables and figures.

Focus on:
1. Confidence intervals in ANY format
2. P-values (including significance symbols)
3. Sample sizes
4. Effect sizes (Cohen's d, odds ratios, etc.)
5. Statistical test results

For each finding, provide:
- Exact value
- Type of statistic
- Location (table/figure/text)
- Context
- Confidence score (0-1)

Return results as structured JSON.
""",
    
    'table_focused': """
Extract ALL numeric data from tables in these articles. Tables contain the most statistical information.

Look for:
- Column headers with CI, p-value, n
- Numeric values with ranges
- Footnotes with statistical notes
- Significance indicators (*, †, ‡)

Preserve exact formatting and precision.
""",
    
    'validation': """
Review these statistical extractions for accuracy. Flag any:
- Impossible values (p > 1, negative sample sizes)
- Inconsistent confidence intervals
- Mismatched statistical tests

Rate confidence in each extraction.
"""
}

# Article Priority Configuration
PRIORITY_PATTERNS = [
    'Table',
    'Figure',
    'CI',
    'confidence interval',
    '95%',
    'p-value',
    'p <',
    'n =',
    'effect size',
    'odds ratio',
    'hazard ratio'
]

# Quality Control Configuration
MIN_EXTRACTION_CONFIDENCE = 0.3
MAX_EXTRACTION_PER_ARTICLE = 500  # Prevent over-extraction

# Error Handling Configuration
ERROR_LOG_FILE = str(BASE_DIR / "error_log.json")
MAX_ERROR_LOG_SIZE = 1000  # Maximum number of errors to keep in memory

# Network Configuration (for distributed processing)
MASTER_HOST = 'localhost'
MASTER_PORT = 8080
WORKER_PORTS = [8081, 8082, 8083, 8084, 8085]

# Distributed Processing Configuration
ENABLE_DISTRIBUTED = False  # Set to True to enable multi-machine processing
SECONDARY_MACHINES = [
    # {'host': '192.168.1.100', 'ssh_user': 'user', 'ssh_key': '/path/to/key'}
]

# Output Format Configuration
OUTPUT_FORMAT = 'jsonl'  # 'json', 'jsonl', 'csv'
COMPRESS_OUTPUT = False  # Whether to gzip output files

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = str(BASE_DIR / "pipeline.log")
LOG_MAX_SIZE = 100 * 1024 * 1024  # 100MB
LOG_BACKUP_COUNT = 5

# Resumption Configuration
ENABLE_AUTO_RESUME = True
RESUME_CHECK_INTERVAL = 60  # Check for resume every minute

# Statistics Collection
COLLECT_DETAILED_STATS = True
STATS_OUTPUT_FILE = str(BASE_DIR / "extraction_stats.json")

# Testing Configuration
TEST_MODE = False
TEST_ARTICLE_LIMIT = 100
TEST_BATCH_SIZE = 5

# Performance Targets
TARGET_ARTICLES_PER_MINUTE = 100
TARGET_COMPLETION_DAYS = 30

# Email Notification Configuration (optional)
ENABLE_EMAIL_NOTIFICATIONS = False
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender': 'your_email@gmail.com',
    'password': 'your_app_password',
    'recipients': ['notify@example.com']
}

# Notification triggers
NOTIFY_ON_COMPLETION = True
NOTIFY_ON_ERROR = True
NOTIFY_ON_MILESTONE = True  # Every 10% progress

def load_environment_config():
    """Load configuration from environment variables if present"""
    # Database
    if os.getenv('DB_NAME'):
        DATABASE_CONFIG['dbname'] = os.getenv('DB_NAME')
    if os.getenv('DB_USER'):
        DATABASE_CONFIG['user'] = os.getenv('DB_USER')
    if os.getenv('DB_PASSWORD'):
        DATABASE_CONFIG['password'] = os.getenv('DB_PASSWORD')
    if os.getenv('DB_HOST'):
        DATABASE_CONFIG['host'] = os.getenv('DB_HOST')
    if os.getenv('DB_PORT'):
        DATABASE_CONFIG['port'] = int(os.getenv('DB_PORT'))
        
    # Ollama
    if os.getenv('OLLAMA_MODEL'):
        global LLM_MODEL
        LLM_MODEL = os.getenv('OLLAMA_MODEL')
        
    # Directories
    if os.getenv('CHECKPOINT_DIR'):
        global CHECKPOINT_DIR
        CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR')
    if os.getenv('OUTPUT_DIR'):
        global LLM_OUTPUT_DIR
        LLM_OUTPUT_DIR = os.getenv('OUTPUT_DIR')
        
    # Performance
    if os.getenv('MAX_MEMORY_MB'):
        global MAX_MEMORY_MB
        MAX_MEMORY_MB = int(os.getenv('MAX_MEMORY_MB'))
    if os.getenv('NUM_WORKERS'):
        global NUM_WORKERS
        NUM_WORKERS = int(os.getenv('NUM_WORKERS'))
        
def validate_config():
    """Validate configuration settings"""
    # Check directories exist or can be created
    for dir_path in [CHECKPOINT_DIR, LLM_OUTPUT_DIR, MERGED_OUTPUT_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
    # Check regex extraction directory exists
    if not Path(REGEX_EXTRACTION_DIR).exists():
        print(f"Warning: Regex extraction directory not found: {REGEX_EXTRACTION_DIR}")
        
    # Validate database connection
    try:
        import psycopg2
        conn = psycopg2.connect(**DATABASE_CONFIG)
        conn.close()
        print("✓ Database connection successful")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        print("Please update DATABASE_CONFIG with correct credentials")
        
    # Check Ollama instances
    import requests
    for instance in OLLAMA_INSTANCES:
        try:
            response = requests.get(f"http://{instance['host']}:{instance['port']}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Ollama instance {instance['name']} is accessible")
            else:
                print(f"✗ Ollama instance {instance['name']} returned status {response.status_code}")
        except Exception as e:
            print(f"✗ Ollama instance {instance['name']} is not accessible: {e}")
            
    print("\nConfiguration validation complete")
    
# Load environment variables on import
load_environment_config()

if __name__ == "__main__":
    # Validate configuration when run directly
    validate_config()
