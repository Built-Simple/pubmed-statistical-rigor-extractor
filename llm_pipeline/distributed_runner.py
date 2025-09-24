"""
Distributed Runner for Multi-Machine Processing
Coordinates pipeline execution across multiple machines
"""

import asyncio
import aiohttp
import json
import logging
import subprocess
import paramiko
import socket
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pickle
import psycopg2

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedCoordinator:
    """Coordinates distributed processing across multiple machines"""
    
    def __init__(self, role: str = 'master'):
        """
        Initialize distributed coordinator
        
        Args:
            role: 'master' or 'worker'
        """
        self.role = role
        self.workers = []
        self.ssh_clients = {}
        self.worker_status = {}
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        
        # Master server for coordination
        self.server = None
        self.clients = []
        
        # Performance metrics per worker
        self.worker_metrics = {}
        
    async def start_master(self):
        """Start the master coordinator"""
        logger.info("Starting distributed master coordinator...")
        
        # Start coordination server
        self.server = await asyncio.start_server(
            self._handle_worker_connection,
            config.MASTER_HOST,
            config.MASTER_PORT
        )
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f"Master coordinator listening on {addr}")
        
        # Connect to secondary machines
        await self._connect_to_workers()
        
        # Start monitoring task
        asyncio.create_task(self._monitor_workers())
        
        # Start load balancer
        asyncio.create_task(self._load_balancer())
        
    async def _handle_worker_connection(self, reader, writer):
        """Handle incoming worker connections"""
        addr = writer.get_extra_info('peername')
        logger.info(f"Worker connected from {addr}")
        
        worker_id = f"worker_{addr[0]}_{addr[1]}"
        self.workers.append({
            'id': worker_id,
            'reader': reader,
            'writer': writer,
            'status': 'connected',
            'last_heartbeat': time.time()
        })
        
        # Handle worker messages
        try:
            while True:
                data = await reader.read(4096)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                await self._process_worker_message(worker_id, message)
                
        except Exception as e:
            logger.error(f"Error handling worker {worker_id}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            self._remove_worker(worker_id)
            
    async def _connect_to_workers(self):
        """Connect to secondary machines via SSH"""
        for machine in config.SECONDARY_MACHINES:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                if 'ssh_key' in machine:
                    client.connect(
                        hostname=machine['host'],
                        username=machine['ssh_user'],
                        key_filename=machine['ssh_key']
                    )
                else:
                    client.connect(
                        hostname=machine['host'],
                        username=machine['ssh_user'],
                        password=machine.get('ssh_password')
                    )
                    
                self.ssh_clients[machine['host']] = client
                
                # Start worker process on remote machine
                await self._start_remote_worker(machine['host'], client)
                
                logger.info(f"Connected to secondary machine: {machine['host']}")
                
            except Exception as e:
                logger.error(f"Failed to connect to {machine['host']}: {e}")
                
    async def _start_remote_worker(self, host: str, ssh_client: paramiko.SSHClient):
        """Start worker process on remote machine"""
        command = f"""
        cd {config.BASE_DIR}
        python distributed_runner.py --role worker --master-host {config.MASTER_HOST} --master-port {config.MASTER_PORT}
        """
        
        stdin, stdout, stderr = ssh_client.exec_command(command)
        
        # Log output in background
        asyncio.create_task(self._log_remote_output(host, stdout, stderr))
        
    async def _log_remote_output(self, host: str, stdout, stderr):
        """Log output from remote worker"""
        for line in stdout:
            logger.info(f"[{host}] {line.strip()}")
        for line in stderr:
            logger.error(f"[{host}] {line.strip()}")
            
    async def _process_worker_message(self, worker_id: str, message: Dict[str, Any]):
        """Process message from worker"""
        msg_type = message.get('type')
        
        if msg_type == 'heartbeat':
            self.worker_status[worker_id] = {
                'last_heartbeat': time.time(),
                'status': message.get('status'),
                'metrics': message.get('metrics', {})
            }
            
        elif msg_type == 'task_complete':
            # Task completed by worker
            result = message.get('result')
            await self.result_queue.put(result)
            
            # Send acknowledgment
            worker = self._get_worker(worker_id)
            if worker:
                response = {'type': 'ack', 'status': 'received'}
                worker['writer'].write(json.dumps(response).encode())
                await worker['writer'].drain()
                
        elif msg_type == 'request_task':
            # Worker requesting new task
            if not self.task_queue.empty():
                task = await self.task_queue.get()
                await self._send_task_to_worker(worker_id, task)
                
        elif msg_type == 'error':
            logger.error(f"Worker {worker_id} reported error: {message.get('error')}")
            
    async def _send_task_to_worker(self, worker_id: str, task: Dict[str, Any]):
        """Send task to specific worker"""
        worker = self._get_worker(worker_id)
        if worker and worker['status'] == 'connected':
            message = {
                'type': 'task',
                'task_id': task.get('id'),
                'data': task
            }
            
            worker['writer'].write(json.dumps(message).encode())
            await worker['writer'].drain()
            
            worker['status'] = 'busy'
            logger.debug(f"Sent task {task.get('id')} to {worker_id}")
            
    def _get_worker(self, worker_id: str) -> Optional[Dict]:
        """Get worker by ID"""
        for worker in self.workers:
            if worker['id'] == worker_id:
                return worker
        return None
        
    def _remove_worker(self, worker_id: str):
        """Remove disconnected worker"""
        self.workers = [w for w in self.workers if w['id'] != worker_id]
        if worker_id in self.worker_status:
            del self.worker_status[worker_id]
        logger.info(f"Removed worker {worker_id}")
        
    async def _monitor_workers(self):
        """Monitor worker health and performance"""
        while True:
            await asyncio.sleep(30)
            
            current_time = time.time()
            for worker in self.workers:
                worker_id = worker['id']
                
                # Check heartbeat
                if worker_id in self.worker_status:
                    last_heartbeat = self.worker_status[worker_id]['last_heartbeat']
                    if current_time - last_heartbeat > 60:
                        logger.warning(f"Worker {worker_id} heartbeat timeout")
                        worker['status'] = 'timeout'
                        
                # Request status update
                if worker['status'] == 'connected':
                    message = {'type': 'status_request'}
                    try:
                        worker['writer'].write(json.dumps(message).encode())
                        await worker['writer'].drain()
                    except:
                        pass
                        
            # Log overall status
            active_workers = sum(1 for w in self.workers if w['status'] in ['connected', 'busy'])
            logger.info(f"Active workers: {active_workers}/{len(self.workers)}")
            
    async def _load_balancer(self):
        """Distribute tasks among workers based on performance"""
        while True:
            if not self.task_queue.empty():
                # Get best available worker
                best_worker = self._get_best_worker()
                
                if best_worker:
                    task = await self.task_queue.get()
                    await self._send_task_to_worker(best_worker['id'], task)
                    
            await asyncio.sleep(1)
            
    def _get_best_worker(self) -> Optional[Dict]:
        """Get the best available worker based on metrics"""
        available_workers = [w for w in self.workers if w['status'] == 'connected']
        
        if not available_workers:
            return None
            
        # Sort by performance metrics
        def worker_score(worker):
            metrics = self.worker_status.get(worker['id'], {}).get('metrics', {})
            # Lower CPU and memory usage is better
            cpu_score = 100 - metrics.get('cpu_percent', 50)
            mem_score = 100 - metrics.get('memory_percent', 50)
            # Higher processing speed is better
            speed_score = metrics.get('articles_per_minute', 1)
            
            return cpu_score * 0.3 + mem_score * 0.3 + speed_score * 0.4
            
        available_workers.sort(key=worker_score, reverse=True)
        return available_workers[0]
        
    async def distribute_articles(self, articles: List[Dict[str, Any]]):
        """Distribute articles across workers"""
        # Create tasks from articles
        for i, article_batch in enumerate(self._chunk_articles(articles, 50)):
            task = {
                'id': f"task_{i}_{datetime.now():%Y%m%d_%H%M%S}",
                'type': 'extract_statistics',
                'articles': article_batch
            }
            await self.task_queue.put(task)
            
        logger.info(f"Queued {self.task_queue.qsize()} tasks for distribution")
        
    def _chunk_articles(self, articles: List[Dict], chunk_size: int) -> List[List[Dict]]:
        """Split articles into chunks"""
        chunks = []
        for i in range(0, len(articles), chunk_size):
            chunks.append(articles[i:i + chunk_size])
        return chunks
        
    async def collect_results(self) -> List[Dict[str, Any]]:
        """Collect results from all workers"""
        results = []
        
        while not self.result_queue.empty():
            result = await self.result_queue.get()
            results.append(result)
            
        return results
        
    async def shutdown(self):
        """Gracefully shutdown distributed processing"""
        logger.info("Shutting down distributed coordinator...")
        
        # Send shutdown signal to all workers
        for worker in self.workers:
            if worker['status'] in ['connected', 'busy']:
                message = {'type': 'shutdown'}
                try:
                    worker['writer'].write(json.dumps(message).encode())
                    await worker['writer'].drain()
                    worker['writer'].close()
                except:
                    pass
                    
        # Close SSH connections
        for client in self.ssh_clients.values():
            client.close()
            
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        logger.info("Distributed coordinator shutdown complete")


class DistributedWorker:
    """Worker node for distributed processing"""
    
    def __init__(self, master_host: str, master_port: int):
        """
        Initialize distributed worker
        
        Args:
            master_host: Master coordinator hostname
            master_port: Master coordinator port
        """
        self.master_host = master_host
        self.master_port = master_port
        self.reader = None
        self.writer = None
        self.running = False
        
        # Initialize local extraction components
        from extraction_engine import ExtractionEngine
        from validator import Validator
        
        self.extraction_engine = ExtractionEngine()
        self.validator = Validator()
        
    async def connect_to_master(self):
        """Connect to master coordinator"""
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.master_host,
                self.master_port
            )
            
            logger.info(f"Connected to master at {self.master_host}:{self.master_port}")
            
            # Send initial registration
            message = {
                'type': 'register',
                'worker_info': {
                    'hostname': socket.gethostname(),
                    'capabilities': ['extraction', 'validation']
                }
            }
            
            self.writer.write(json.dumps(message).encode())
            await self.writer.drain()
            
            self.running = True
            
        except Exception as e:
            logger.error(f"Failed to connect to master: {e}")
            raise
            
    async def run(self):
        """Main worker loop"""
        # Start heartbeat task
        asyncio.create_task(self._send_heartbeat())
        
        # Request initial task
        await self._request_task()
        
        # Process messages from master
        while self.running:
            try:
                data = await self.reader.read(4096)
                if not data:
                    break
                    
                message = json.loads(data.decode())
                await self._process_master_message(message)
                
            except Exception as e:
                logger.error(f"Error processing master message: {e}")
                await asyncio.sleep(5)
                
    async def _process_master_message(self, message: Dict[str, Any]):
        """Process message from master"""
        msg_type = message.get('type')
        
        if msg_type == 'task':
            # Process task
            task_id = message.get('task_id')
            task_data = message.get('data')
            
            logger.info(f"Received task {task_id}")
            
            result = await self._process_task(task_data)
            
            # Send result back
            response = {
                'type': 'task_complete',
                'task_id': task_id,
                'result': result
            }
            
            self.writer.write(json.dumps(response).encode())
            await self.writer.drain()
            
            # Request next task
            await self._request_task()
            
        elif msg_type == 'status_request':
            # Send status update
            await self._send_status()
            
        elif msg_type == 'shutdown':
            logger.info("Received shutdown signal")
            self.running = False
            
    async def _process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process extraction task"""
        task_type = task.get('type')
        
        if task_type == 'extract_statistics':
            articles = task.get('articles', [])
            results = []
            
            for article in articles:
                try:
                    # Extract statistics
                    findings = self.extraction_engine._regex_extraction(article)
                    
                    # Validate
                    validated = self.extraction_engine.validate_findings(findings)
                    
                    # Convert to result format
                    result = {
                        'pmc_id': article['pmc_id'],
                        'findings': [f.to_dict() for f in validated],
                        'status': 'completed'
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing article {article.get('pmc_id')}: {e}")
                    results.append({
                        'pmc_id': article.get('pmc_id'),
                        'error': str(e),
                        'status': 'failed'
                    })
                    
            return {'results': results}
            
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {'error': 'Unknown task type'}
            
    async def _request_task(self):
        """Request new task from master"""
        message = {'type': 'request_task'}
        self.writer.write(json.dumps(message).encode())
        await self.writer.drain()
        
    async def _send_heartbeat(self):
        """Send periodic heartbeat to master"""
        while self.running:
            await asyncio.sleep(30)
            
            if self.writer:
                try:
                    # Get system metrics
                    import psutil
                    process = psutil.Process()
                    
                    message = {
                        'type': 'heartbeat',
                        'status': 'alive',
                        'metrics': {
                            'cpu_percent': process.cpu_percent(),
                            'memory_percent': process.memory_percent(),
                            'articles_per_minute': 0  # TODO: Calculate actual rate
                        }
                    }
                    
                    self.writer.write(json.dumps(message).encode())
                    await self.writer.drain()
                    
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                    
    async def _send_status(self):
        """Send detailed status to master"""
        import psutil
        
        message = {
            'type': 'status',
            'hostname': socket.gethostname(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'disk_usage': psutil.disk_usage('/').percent
        }
        
        self.writer.write(json.dumps(message).encode())
        await self.writer.drain()
        
    async def shutdown(self):
        """Shutdown worker gracefully"""
        self.running = False
        
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            
        logger.info("Worker shutdown complete")


async def run_master():
    """Run as master coordinator"""
    coordinator = DistributedCoordinator(role='master')
    await coordinator.start_master()
    
    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        await coordinator.shutdown()
        

async def run_worker(master_host: str, master_port: int):
    """Run as worker node"""
    worker = DistributedWorker(master_host, master_port)
    
    try:
        await worker.connect_to_master()
        await worker.run()
    except KeyboardInterrupt:
        await worker.shutdown()
        

def main():
    """Main entry point for distributed runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed runner for extraction pipeline')
    parser.add_argument('--role', choices=['master', 'worker'], default='master',
                       help='Role: master or worker')
    parser.add_argument('--master-host', default=config.MASTER_HOST,
                       help='Master coordinator hostname')
    parser.add_argument('--master-port', type=int, default=config.MASTER_PORT,
                       help='Master coordinator port')
    
    args = parser.parse_args()
    
    if args.role == 'master':
        asyncio.run(run_master())
    else:
        asyncio.run(run_worker(args.master_host, args.master_port))
        

if __name__ == "__main__":
    main()
