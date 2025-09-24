# PubMed Statistical Extraction Pipeline - LLM Enhancement

## Overview
This pipeline enhances regex-based statistical extraction from 4.48 million PubMed articles using Large Language Models (LLMs), specifically targeting statistics missed in tables, figures, and non-standard formats.

### Key Features
- **Multi-model Ollama orchestration** across multiple GPUs and machines
- **Intelligent batching** based on article characteristics
- **Automatic checkpoint/resume** for 24/7 operation
- **Real-time validation** and quality control
- **Distributed processing** across multiple machines
- **Comprehensive progress tracking** with PostgreSQL integration

## System Requirements

### Hardware
- **Main Machine**: AMD Ryzen 9 7900 (24 cores), 128GB RAM, Dual RTX 3090 (48GB VRAM)
- **Secondary Machine(s)**: Additional GPU resources (optional)
- **Storage**: ~500GB for outputs and checkpoints

### Software
- Python 3.8+
- PostgreSQL 17
- Ollama with Qwen2.5:72b model
- CUDA drivers for GPU support

## Installation

### 1. Clone and Setup
```bash
cd C:\Users\neely\pubmed-statistical-rigor-extractor\llm_pipeline
pip install -r requirements.txt
```

### 2. Configure Database
Update `config.py` with your PostgreSQL credentials:
```python
DATABASE_CONFIG = {
    'dbname': 'pmc_fulltext',
    'user': 'your_username',
    'password': 'your_password',
    'host': 'localhost',
    'port': 5432
}
```

### 3. Setup Ollama Instances

#### On Main Machine:
```bash
# Start multiple Ollama instances on different ports
ollama serve --port 11434 &
ollama serve --port 11435 &
ollama serve --port 11436 &

# Pull the model (only needed once)
ollama pull qwen2.5:72b
```

#### On Secondary Machine(s):
```bash
# Update IP in config.py, then:
ollama serve --port 11437 &
ollama serve --port 11438 &
```

### 4. Verify Configuration
```bash
python config.py
```
This will validate database connection and Ollama instances.

## Usage

### Basic Usage - Process All Articles
```bash
python main_pipeline.py --start-from-beginning --use-all-articles
```

### Resume from Checkpoint
```bash
python main_pipeline.py --resume --run-id run_20250923_210605
```

### Test Run (100 articles)
```bash
python main_pipeline.py --limit 100 --single-instance
```

### Advanced Options
```bash
# Process with specific configurations
python main_pipeline.py \
    --workers 5 \
    --batch-size 30 \
    --priority-journals "Nature" "Science" "Cell" \
    --merge-on-complete \
    --checkpoint-interval 600
```

### Distributed Processing
```bash
# On master machine:
python distributed_runner.py --role master

# On worker machines:
python distributed_runner.py --role worker --master-host 192.168.1.100
```

## Pipeline Components

### 1. **Batch Processor** (`batch_processor.py`)
- Groups similar articles for optimal LLM processing
- Manages article retrieval from PostgreSQL
- Tracks processed articles to avoid duplicates
- Intelligent batching based on article length and content

### 2. **Ollama Manager** (`ollama_manager.py`)
- Load balances across multiple Ollama instances
- Handles failover and retry logic
- Monitors instance health and performance
- Supports distributed GPU resources

### 3. **Extraction Engine** (`extraction_engine.py`)
- Core statistical extraction logic
- Table and figure section identification
- LLM prompt optimization for statistics
- Combines regex and LLM approaches

### 4. **Progress Tracker** (`progress_tracker.py`)
- Checkpoint management for resume capability
- Real-time progress monitoring
- ETA calculation
- Database persistence of progress

### 5. **Validator** (`validator.py`)
- Statistical validation (CI bounds, p-value ranges)
- Quality scoring
- Comparison with regex baseline
- Error detection and logging

### 6. **Merger** (`merger.py`)
- Combines LLM and regex extractions
- Deduplication logic
- Output formatting
- Database storage of merged results

### 7. **Main Pipeline** (`main_pipeline.py`)
- Orchestrates all components
- Manages worker threads
- Performance monitoring
- Graceful shutdown handling

## Extraction Targets

The pipeline specifically targets:

1. **Confidence Intervals**
   - Table formats: (X to Y), [X, Y], X (Y-Z)
   - Narrative: "between X and Y"
   - Column-based in tables

2. **P-values**
   - Non-standard: "significant at 0.05 level"
   - Symbols: *, **, *** indicators
   - Table footnotes

3. **Sample Sizes**
   - n=X anywhere in text
   - "X participants/patients/subjects"
   - Table headers and footers

4. **Effect Sizes**
   - Cohen's d, eta squared, R squared
   - Odds ratios, hazard ratios
   - Beta coefficients

## Monitoring

### Real-time Progress
The pipeline provides continuous updates:
```
============================================================
EXTRACTION PROGRESS - Run: run_20250923_210605
============================================================
Status: running
Progress: 125,432/4,483,066 (2.8%)
Failed: 23
Current Batch: 2509
Elapsed Time: 2:15:30
Speed: 92.7 articles/minute
ETA: 2025-10-24 15:30:00

Extraction Counts:
  confidence_intervals: 1,245,678
  p_values: 892,345
  sample_sizes: 456,789
  effect_sizes: 234,567
============================================================
```

### Database Tables
Monitor via PostgreSQL:
```sql
-- Check overall progress
SELECT * FROM llm_extraction_progress;

-- View article-level status
SELECT status, COUNT(*) 
FROM llm_article_status 
GROUP BY status;

-- Check extraction quality
SELECT AVG(confidence_intervals) as avg_ci,
       AVG(p_values) as avg_p,
       AVG(sample_sizes) as avg_n
FROM merge_summary;
```

## Performance Optimization

### Batch Size Tuning
Adjust in `config.py` based on your hardware:
```python
BATCH_SIZE_LIMITS = {
    'short': {'min': 20, 'max': 50},    # More articles for short texts
    'medium': {'min': 10, 'max': 25},   # Balanced
    'long': {'min': 5, 'max': 10},      # Fewer for long articles
    'very_long': {'min': 1, 'max': 5}   # Individual processing
}
```

### Memory Management
```python
MAX_MEMORY_MB = 64000  # Adjust based on available RAM
```

### GPU Optimization
- Run 3 Qwen2.5:72b instances on 104GB VRAM machine
- Run 2 instances on 48GB VRAM machine
- Monitor GPU memory with `nvidia-smi`

## Expected Performance

- **Processing Speed**: 100+ articles/minute
- **Completion Time**: 30-40 days for 4.48M articles
- **Extraction Improvement**: 60%+ more CIs than regex alone
- **Accuracy**: 95%+ for table-based statistics

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch sizes in config.py
   - Decrease number of workers
   - Check for memory leaks with psutil

2. **Ollama Timeout**
   - Increase timeout in ollama_manager.py
   - Check GPU utilization
   - Reduce model temperature

3. **Database Connection Issues**
   - Verify PostgreSQL is running
   - Check connection parameters
   - Ensure sufficient connections available

4. **Slow Processing**
   - Check Ollama instance health
   - Verify GPU is being utilized
   - Adjust batch sizes

### Logs
Check `pipeline.log` for detailed debugging information.

## Output Files

### LLM Extractions
```
llm_extractions/
├── extractions_20250923.jsonl
├── extractions_20250924.jsonl
└── ...
```

### Merged Results
```
merged_extractions/
├── merged_20250923.jsonl
└── merged_summary.json
```

### Checkpoints
```
checkpoints/
├── latest_run_20250923.json
└── checkpoint_run_20250923_143022.json
```

## Validation Metrics

The pipeline tracks:
- **Validity Rate**: Percentage of valid extractions
- **Confidence Scores**: Per-extraction confidence levels
- **Error Patterns**: Common validation failures
- **Improvement Rates**: LLM vs regex performance

## Database Schema

```sql
-- Processing status
CREATE TABLE llm_article_status (
    pmc_id VARCHAR(20) PRIMARY KEY,
    processed_at TIMESTAMP,
    status VARCHAR(20),
    findings_count INTEGER,
    confidence_intervals INTEGER,
    p_values INTEGER,
    sample_sizes INTEGER,
    effect_sizes INTEGER
);

-- Merged statistics
CREATE TABLE merged_statistics (
    id SERIAL PRIMARY KEY,
    pmc_id VARCHAR(20),
    statistic_type VARCHAR(50),
    value JSONB,
    context TEXT,
    location VARCHAR(50),
    source VARCHAR(20),
    confidence FLOAT
);
```

## Contributing

To modify extraction patterns, edit:
1. `extraction_engine.py` - Core extraction logic
2. `config.py` - Extraction prompts
3. `validator.py` - Validation rules

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review pipeline.log for errors
3. Verify configuration with `python config.py`

## License

This pipeline is designed for academic research purposes.

---

## Quick Start Checklist

- [ ] PostgreSQL database running with 4.48M articles
- [ ] Ollama installed with Qwen2.5:72b model
- [ ] Python dependencies installed
- [ ] Database credentials configured
- [ ] Ollama instances started
- [ ] Configuration validated
- [ ] Test run successful with --limit 100
- [ ] Ready for full pipeline execution

## Command Reference

```bash
# Full processing
python main_pipeline.py --start-from-beginning --use-all-articles

# Resume after interruption
python main_pipeline.py --resume

# Test with small batch
python main_pipeline.py --limit 100 --debug

# Distributed master
python distributed_runner.py --role master

# Distributed worker
python distributed_runner.py --role worker --master-host <IP>

# Validate configuration
python config.py

# Merge results manually
python merger.py
```
