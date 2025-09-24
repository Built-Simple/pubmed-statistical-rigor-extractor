"""
Startup script for PubMed Statistical Extraction Pipeline
Provides easy launch options for common scenarios
"""

import subprocess
import sys
import os
import time
import psutil
from pathlib import Path
import argparse
import json

def check_requirements():
    """Check if all requirements are met"""
    print("Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print("✓ Python version OK")
    
    # Check PostgreSQL
    try:
        import psycopg2
        print("✓ PostgreSQL driver installed")
    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install -r requirements.txt")
        return False
    
    # Check Ollama
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'qwen2.5:72b' in result.stdout:
            print("✓ Qwen2.5:72b model available")
        else:
            print("⚠ Qwen2.5:72b not found. Run: ollama pull qwen2.5:72b")
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first")
        return False
    
    # Check GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print("✓ NVIDIA GPU detected")
    except FileNotFoundError:
        print("⚠ NVIDIA GPU tools not found")
    
    # Check memory
    mem = psutil.virtual_memory()
    if mem.total < 64 * 1024 * 1024 * 1024:  # 64GB
        print(f"⚠ Low memory: {mem.total / 1024**3:.1f}GB (recommended: 128GB)")
    else:
        print(f"✓ Memory: {mem.total / 1024**3:.1f}GB")
    
    return True

def start_ollama_instances(num_instances=3, base_port=11434):
    """Start multiple Ollama instances"""
    print(f"\nStarting {num_instances} Ollama instances...")
    
    processes = []
    for i in range(num_instances):
        port = base_port + i
        print(f"Starting Ollama on port {port}...")
        
        # Check if already running
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"  Ollama already running on port {port}")
                continue
        except:
            pass
        
        # Start new instance
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
        
        process = subprocess.Popen(
            ['ollama', 'serve'],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(process)
        print(f"  Started Ollama on port {port} (PID: {process.pid})")
        
    # Wait for instances to be ready
    time.sleep(5)
    
    # Verify all instances
    import requests
    for i in range(num_instances):
        port = base_port + i
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✓ Ollama instance on port {port} is ready")
            else:
                print(f"❌ Ollama instance on port {port} not responding")
        except Exception as e:
            print(f"❌ Cannot connect to Ollama on port {port}: {e}")
    
    return processes

def create_run_config(scenario):
    """Create configuration for different scenarios"""
    configs = {
        'test': {
            'args': ['--limit', '100', '--single-instance', '--debug'],
            'description': 'Test run with 100 articles using single Ollama instance'
        },
        'small': {
            'args': ['--limit', '10000', '--workers', '2'],
            'description': 'Small run with 10,000 articles'
        },
        'resume': {
            'args': ['--resume'],
            'description': 'Resume from last checkpoint'
        },
        'full': {
            'args': ['--start-from-beginning', '--use-all-articles', '--workers', '3'],
            'description': 'Full pipeline for all 4.48M articles'
        },
        'full-merge': {
            'args': ['--start-from-beginning', '--use-all-articles', '--merge-on-complete'],
            'description': 'Full pipeline with automatic merging'
        },
        'distributed-master': {
            'args': [],
            'script': 'distributed_runner.py',
            'extra_args': ['--role', 'master'],
            'description': 'Start distributed master coordinator'
        },
        'distributed-worker': {
            'args': [],
            'script': 'distributed_runner.py',
            'extra_args': ['--role', 'worker'],
            'description': 'Start distributed worker node'
        }
    }
    
    return configs.get(scenario, configs['test'])

def run_pipeline(scenario='test', custom_args=None):
    """Run the pipeline with specified configuration"""
    config = create_run_config(scenario)
    script = config.get('script', 'main_pipeline.py')
    
    print(f"\n{config['description']}")
    print("="*60)
    
    cmd = ['python', script]
    cmd.extend(config.get('args', []))
    cmd.extend(config.get('extra_args', []))
    
    if custom_args:
        cmd.extend(custom_args)
    
    print(f"Command: {' '.join(cmd)}")
    print("="*60)
    
    # Run the pipeline
    try:
        process = subprocess.run(cmd)
        return process.returncode
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running pipeline: {e}")
        return 1

def monitor_pipeline():
    """Open monitoring dashboard"""
    print("\nOpening monitoring dashboard...")
    
    # Check for running pipeline
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and 'main_pipeline.py' in str(proc.info['cmdline']):
                print(f"Found running pipeline (PID: {proc.info['pid']})")
                
                # Get latest checkpoint
                checkpoint_dir = Path("checkpoints")
                if checkpoint_dir.exists():
                    latest = max(checkpoint_dir.glob("latest_*.json"), key=os.path.getctime, default=None)
                    if latest:
                        with open(latest) as f:
                            data = json.load(f)
                        
                        print("\nCurrent Status:")
                        print(f"  Run ID: {data.get('run_id')}")
                        print(f"  Progress: {len(data.get('processed_articles', []))}/{data.get('total_articles')}")
                        print(f"  Batch: {data.get('current_batch')}")
                        
                        # Calculate stats
                        processed = len(data.get('processed_articles', []))
                        total = data.get('total_articles', 1)
                        percentage = (processed / total) * 100
                        print(f"  Completion: {percentage:.2f}%")
                
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    else:
        print("No running pipeline found")

def stop_ollama_instances():
    """Stop all Ollama instances"""
    print("\nStopping Ollama instances...")
    
    stopped = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'ollama' in proc.info['name'].lower():
                proc.terminate()
                stopped += 1
                print(f"  Stopped Ollama (PID: {proc.info['pid']})")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    print(f"Stopped {stopped} Ollama instances")

def main():
    """Main entry point for startup script"""
    parser = argparse.ArgumentParser(
        description='PubMed Pipeline Startup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scenarios:
  test              - Test run with 100 articles
  small             - Process 10,000 articles
  resume            - Resume from last checkpoint
  full              - Process all 4.48M articles
  full-merge        - Process all and merge results
  distributed-master - Start distributed master
  distributed-worker - Start distributed worker

Examples:
  python start_pipeline.py test
  python start_pipeline.py full --workers 5
  python start_pipeline.py resume --run-id run_20250923
        """
    )
    
    parser.add_argument('scenario', 
                       choices=['test', 'small', 'resume', 'full', 'full-merge', 
                               'distributed-master', 'distributed-worker', 'monitor',
                               'start-ollama', 'stop-ollama', 'check'],
                       help='Pipeline scenario to run')
    
    parser.add_argument('--ollama-instances', type=int, default=3,
                       help='Number of Ollama instances to start')
    
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip requirement checks')
    
    # Pass through additional arguments
    parser.add_argument('extra_args', nargs='*', 
                       help='Additional arguments to pass to pipeline')
    
    args = parser.parse_args()
    
    # Special commands
    if args.scenario == 'check':
        check_requirements()
        return
    
    if args.scenario == 'start-ollama':
        start_ollama_instances(args.ollama_instances)
        return
    
    if args.scenario == 'stop-ollama':
        stop_ollama_instances()
        return
    
    if args.scenario == 'monitor':
        monitor_pipeline()
        return
    
    # Regular pipeline execution
    if not args.skip_checks:
        if not check_requirements():
            print("\n❌ Requirements check failed. Fix issues and try again.")
            return 1
    
    # Start Ollama if needed
    if args.scenario in ['test', 'small', 'full', 'full-merge']:
        if args.scenario == 'test':
            ollama_procs = start_ollama_instances(1)
        else:
            ollama_procs = start_ollama_instances(args.ollama_instances)
    
    # Run pipeline
    exit_code = run_pipeline(args.scenario, args.extra_args)
    
    print(f"\nPipeline completed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
