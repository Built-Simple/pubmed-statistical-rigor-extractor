# QUICK SETUP GUIDE - PubMed Statistical Extraction Pipeline

## 🚀 5-Minute Setup

### Step 1: Install Dependencies
```bash
cd C:\Users\neely\pubmed-statistical-rigor-extractor\llm_pipeline
pip install -r requirements.txt
```

### Step 2: Configure Database
Edit `config.py` and update these lines with your PostgreSQL credentials:
```python
DATABASE_CONFIG = {
    'dbname': 'pmc_fulltext',
    'user': 'your_username',     # <-- CHANGE THIS
    'password': 'your_password',   # <-- CHANGE THIS
    'host': 'localhost',
    'port': 5432
}
```

### Step 3: Install Ollama & Model
1. Download Ollama from https://ollama.ai/
2. Install and run:
```bash
ollama pull qwen2.5:72b
```
Note: This will download ~40GB model. Ensure sufficient disk space.

### Step 4: Test Configuration
```bash
python config.py
```
You should see:
- ✓ Database connection successful
- ✓ Ollama instance accessible

### Step 5: Run Test
```bash
python start_pipeline.py test
```
This processes 100 articles to verify everything works.

## 📊 Running the Full Pipeline

### Option A: Windows (Easy Mode)
Double-click `run_pipeline.bat` and select option 4 for full processing.

### Option B: Command Line
```bash
python main_pipeline.py --start-from-beginning --use-all-articles
```

### Option C: With Auto-Start
```bash
# This starts Ollama instances automatically
python start_pipeline.py full
```

## 🔄 Resume After Interruption
```bash
python main_pipeline.py --resume
```
Or use the batch file and select option 5.

## 📈 Monitor Progress

### Real-time Monitoring
```bash
python start_pipeline.py monitor
```

### Database Query
```sql
-- Connect to PostgreSQL and run:
SELECT 
    run_id,
    processed_articles,
    total_articles,
    ROUND(processed_articles::numeric / total_articles * 100, 2) as percent_complete,
    last_update
FROM llm_extraction_progress
WHERE status = 'running';
```

### Log File
```bash
tail -f pipeline.log
```

## ⚡ Performance Tips

### For Faster Processing:
1. **Increase Workers** (if you have RAM):
   ```bash
   python main_pipeline.py --workers 5
   ```

2. **Use More Ollama Instances**:
   ```bash
   python start_pipeline.py start-ollama --ollama-instances 4
   ```

3. **Optimize Batch Sizes** in `config.py`:
   ```python
   BATCH_SIZE_LIMITS = {
       'short': {'max': 75},  # Increase for more RAM
       'medium': {'max': 35},
       'long': {'max': 15}
   }
   ```

## 🛑 Common Issues & Fixes

### Issue: "Out of memory"
**Fix**: Reduce batch sizes or workers
```bash
python main_pipeline.py --workers 2 --batch-size 20
```

### Issue: "Ollama timeout"
**Fix**: Start fewer instances or reduce batch size
```bash
python start_pipeline.py test --single-instance
```

### Issue: "Database connection refused"
**Fix**: Ensure PostgreSQL is running
```bash
# Windows
net start postgresql-x64-17

# or check services.msc
```

### Issue: "No module named 'psycopg2'"
**Fix**: Install dependencies
```bash
pip install psycopg2-binary
```

## 📁 Output Locations

- **Extractions**: `llm_extractions/`
- **Merged Results**: `merged_extractions/`
- **Checkpoints**: `checkpoints/`
- **Logs**: `pipeline.log`

## 🎯 Expected Results

- **Speed**: ~100 articles/minute
- **Duration**: 30-40 days for all 4.48M articles
- **Improvement**: 60%+ more statistics than regex alone
- **Storage**: ~50GB for all outputs

## 💡 Quick Commands Reference

```bash
# Test run (100 articles)
python start_pipeline.py test

# Small batch (10,000 articles)
python start_pipeline.py small

# Full processing
python start_pipeline.py full

# Resume from checkpoint
python start_pipeline.py resume

# Check status
python start_pipeline.py monitor

# Validate setup
python config.py

# Start Ollama instances
python start_pipeline.py start-ollama

# Emergency stop
python start_pipeline.py stop-ollama
```

## ✅ Pre-Launch Checklist

- [ ] PostgreSQL running with 4.48M articles loaded
- [ ] Database credentials in config.py
- [ ] Ollama installed
- [ ] Qwen2.5:72b model downloaded
- [ ] Python dependencies installed
- [ ] Test run completed successfully
- [ ] At least 100GB free disk space
- [ ] GPU drivers installed (nvidia-smi works)

## 🚨 Emergency Stop

Press `Ctrl+C` twice, or run:
```bash
taskkill /F /IM python.exe /T
taskkill /F /IM ollama.exe /T
```

## 📞 Need Help?

1. Check `pipeline.log` for detailed errors
2. Run `python config.py` to validate setup
3. Try test mode first: `python start_pipeline.py test`

---

**Ready to go?** Start with the test run to ensure everything works, then launch the full pipeline!

```bash
# Your first command:
python start_pipeline.py test
```

Good luck! The pipeline will run 24/7 until all 4.48M articles are processed. 🚀
