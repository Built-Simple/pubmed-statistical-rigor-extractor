import sys
import os
import psutil
import json
from datetime import datetime
import subprocess

def run_diagnostic():
    diagnostic = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else "N/A",
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "percent": psutil.virtual_memory().percent
        },
        "disk_usage": {},
        "gpu_info": {},
        "environment": {
            k: v for k, v in os.environ.items()
            if any(x in k.upper() for x in ['CUDA', 'NVIDIA', 'OLLAMA', 'PATH'])
        },
        "working_directory": os.getcwd(),
        "database_check": check_database_connection(),
        "ollama_models": check_ollama_models(),
        "extraction_output": check_extraction_output()
    }

    # Disk usage for all drives
    for partition in psutil.disk_partitions():
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            diagnostic["disk_usage"][partition.device] = {
                "mountpoint": partition.mountpoint,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2),
                "percent": usage.percent
            }
        except:
            pass

    # GPU info
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(", ")
            diagnostic["gpu_info"] = {
                "name": gpu_data[0],
                "memory_total": gpu_data[1],
                "memory_free": gpu_data[2],
                "memory_used": gpu_data[3]
            }
    except:
        diagnostic["gpu_info"] = "No NVIDIA GPU detected or nvidia-smi not available"

    return diagnostic

def check_database_connection():
    try:
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            database='pmc_fulltext',
            user='postgres',
            password='Tapane2001!',
            port=5432
        )
        cursor = conn.cursor()

        # Get article count
        cursor.execute("SELECT COUNT(*) FROM articles")
        count = cursor.fetchone()[0]

        # Get sample article to check structure
        cursor.execute("SELECT id, pmc_id, title FROM articles LIMIT 1")
        sample = cursor.fetchone()

        # Get year range
        cursor.execute("SELECT MIN(pub_year), MAX(pub_year) FROM articles WHERE pub_year IS NOT NULL")
        year_range = cursor.fetchone()

        cursor.close()
        conn.close()

        return {
            "status": "connected",
            "article_count": count,
            "sample_article": {
                "id": sample[0],
                "pmc_id": sample[1],
                "title": sample[2][:50] + "..." if sample[2] else None
            },
            "year_range": f"{year_range[0]}-{year_range[1]}" if year_range[0] else "N/A"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def check_ollama_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Parse ollama list output
            models = []
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.split()
                    if parts:
                        models.append(parts[0])
            return models if models else ["No models installed"]
        return "Ollama command failed"
    except:
        return "Ollama not installed or not in PATH"

def check_extraction_output():
    """Check the extraction output directory"""
    try:
        import glob

        # Find extraction directories
        extraction_dirs = glob.glob("D:/PubMed_Statistical_Analysis/extraction_*")

        if extraction_dirs:
            latest_dir = max(extraction_dirs)
            jsonl_files = glob.glob(os.path.join(latest_dir, "*.jsonl"))

            # Calculate total size
            total_size = sum(os.path.getsize(f) for f in jsonl_files)

            return {
                "latest_extraction": os.path.basename(latest_dir),
                "file_count": len(jsonl_files),
                "total_size_gb": round(total_size / (1024**3), 2),
                "location": latest_dir
            }
        else:
            return "No extraction output found"
    except Exception as e:
        return f"Error checking extraction: {str(e)}"

# Run diagnostic
print("="*80)
print("PUBMED STATISTICAL EXTRACTION - SYSTEM DIAGNOSTIC")
print("="*80)
print(json.dumps(run_diagnostic(), indent=2))
print("="*80)