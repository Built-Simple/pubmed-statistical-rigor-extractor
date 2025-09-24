"""
Ollama Multi-Instance Manager
Handles load balancing across multiple Ollama instances on multiple machines
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import random
from concurrent.futures import ThreadPoolExecutor
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OllamaInstance:
    """Represents a single Ollama instance"""
    host: str
    port: int
    name: str
    is_available: bool = True
    current_load: int = 0
    total_processed: int = 0
    total_errors: int = 0
    last_error_time: float = 0
    average_response_time: float = 0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def health_score(self) -> float:
        """Calculate health score for load balancing"""
        error_penalty = min(self.total_errors * 0.1, 0.5)
        load_penalty = self.current_load * 0.2
        return max(0, 1.0 - error_penalty - load_penalty)


class OllamaManager:
    """Manages multiple Ollama instances with load balancing"""
    
    def __init__(self, instances_config: List[Dict[str, Any]]):
        """
        Initialize with instance configurations
        instances_config: List of dicts with 'host', 'port', 'name'
        """
        self.instances = []
        for config in instances_config:
            instance = OllamaInstance(
                host=config['host'],
                port=config['port'],
                name=config['name']
            )
            self.instances.append(instance)
        
        self.request_queue = asyncio.Queue()
        self.response_times = deque(maxlen=100)
        self.lock = threading.Lock()
        
        # Start health check task
        self.health_check_task = None
        
    async def start(self):
        """Start the manager and health checks"""
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Ollama Manager started with {len(self.instances)} instances")
        
    async def stop(self):
        """Stop the manager"""
        if self.health_check_task:
            self.health_check_task.cancel()
            
    async def _health_check_loop(self):
        """Continuously check health of instances"""
        while True:
            try:
                await self._check_all_instances()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                
    async def _check_all_instances(self):
        """Check health of all instances"""
        tasks = [self._check_instance(instance) for instance in self.instances]
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _check_instance(self, instance: OllamaInstance):
        """Check if an instance is responding"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{instance.url}/api/tags", timeout=5) as response:
                    if response.status == 200:
                        instance.is_available = True
                        return True
        except:
            instance.is_available = False
            instance.last_error_time = time.time()
        return False
        
    def get_best_instance(self) -> Optional[OllamaInstance]:
        """Get the best available instance based on health score"""
        with self.lock:
            available_instances = [i for i in self.instances if i.is_available]
            if not available_instances:
                return None
                
            # Sort by health score and current load
            available_instances.sort(key=lambda x: (x.health_score, -x.current_load), reverse=True)
            return available_instances[0]
            
    async def process_batch(self, 
                           articles: List[Dict[str, Any]], 
                           prompt_template: str,
                           model: str = "qwen2.5:72b",
                           temperature: float = 0.1,
                           max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Process a batch of articles using the best available instance
        """
        instance = self.get_best_instance()
        if not instance:
            logger.error("No available Ollama instances")
            return []
            
        instance.current_load += 1
        start_time = time.time()
        
        try:
            # Prepare batch prompt
            batch_prompt = self._prepare_batch_prompt(articles, prompt_template)
            
            # Make request
            result = await self._make_request(
                instance, 
                batch_prompt, 
                model, 
                temperature,
                max_retries
            )
            
            # Update statistics
            response_time = time.time() - start_time
            instance.average_response_time = (
                (instance.average_response_time * instance.total_processed + response_time) /
                (instance.total_processed + 1)
            )
            instance.total_processed += 1
            
            return self._parse_batch_response(result, articles)
            
        except Exception as e:
            instance.total_errors += 1
            instance.last_error_time = time.time()
            logger.error(f"Error processing batch on {instance.name}: {e}")
            
            # Try another instance
            return await self._fallback_processing(articles, prompt_template, model, temperature)
            
        finally:
            instance.current_load -= 1
            
    def _prepare_batch_prompt(self, articles: List[Dict[str, Any]], template: str) -> str:
        """Prepare a batch prompt for multiple articles"""
        batch_text = "Process the following articles and extract ALL statistical information:\n\n"
        
        for idx, article in enumerate(articles, 1):
            batch_text += f"===== ARTICLE {idx} (PMC_ID: {article['pmc_id']}) =====\n"
            batch_text += f"Title: {article.get('title', 'N/A')}\n"
            batch_text += f"Abstract: {article.get('abstract', '')[:500]}...\n"
            batch_text += f"Body (excerpt): {article.get('body', '')[:2000]}...\n"
            batch_text += "="*50 + "\n\n"
            
        batch_text += "\n" + template
        return batch_text
        
    async def _make_request(self, 
                           instance: OllamaInstance,
                           prompt: str,
                           model: str,
                           temperature: float,
                           max_retries: int) -> Dict[str, Any]:
        """Make a request to an Ollama instance with retries"""
        url = f"{instance.url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
            "format": "json"
        }
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=300)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            return json.loads(result.get('response', '{}'))
                        else:
                            logger.warning(f"Instance {instance.name} returned status {response.status}")
                            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on instance {instance.name}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Request error on {instance.name}: {e}")
                
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
        raise Exception(f"Failed after {max_retries} retries")
        
    def _parse_batch_response(self, 
                             response: Dict[str, Any], 
                             articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the batch response and map to individual articles"""
        results = []
        
        # Handle different response formats
        if 'articles' in response:
            # Response has article-level results
            for idx, article in enumerate(articles):
                if idx < len(response['articles']):
                    article_result = response['articles'][idx]
                    article_result['pmc_id'] = article['pmc_id']
                    results.append(article_result)
                else:
                    results.append({'pmc_id': article['pmc_id'], 'error': 'No extraction'})
        else:
            # Single response for all articles
            for article in articles:
                results.append({
                    'pmc_id': article['pmc_id'],
                    'statistics': response.get('statistics', {}),
                    'confidence_intervals': response.get('confidence_intervals', []),
                    'p_values': response.get('p_values', []),
                    'sample_sizes': response.get('sample_sizes', []),
                    'effect_sizes': response.get('effect_sizes', [])
                })
                
        return results
        
    async def _fallback_processing(self,
                                  articles: List[Dict[str, Any]],
                                  prompt_template: str,
                                  model: str,
                                  temperature: float) -> List[Dict[str, Any]]:
        """Fallback to another instance if primary fails"""
        # Get next best instance
        for instance in self.instances:
            if instance.is_available and instance.current_load < 3:
                instance.current_load += 1
                try:
                    return await self.process_batch(
                        articles, 
                        prompt_template, 
                        model, 
                        temperature, 
                        max_retries=1
                    )
                finally:
                    instance.current_load -= 1
                    
        logger.error("All fallback attempts failed")
        return [{'pmc_id': a['pmc_id'], 'error': 'All instances failed'} for a in articles]
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all instances"""
        status = {
            'instances': [],
            'total_processed': sum(i.total_processed for i in self.instances),
            'total_errors': sum(i.total_errors for i in self.instances),
            'available_instances': sum(1 for i in self.instances if i.is_available)
        }
        
        for instance in self.instances:
            status['instances'].append({
                'name': instance.name,
                'url': instance.url,
                'is_available': instance.is_available,
                'current_load': instance.current_load,
                'total_processed': instance.total_processed,
                'total_errors': instance.total_errors,
                'health_score': instance.health_score,
                'average_response_time': instance.average_response_time
            })
            
        return status


# Configuration for multiple machines
DEFAULT_INSTANCES = [
    # First machine (104GB VRAM)
    {'host': 'localhost', 'port': 11434, 'name': 'ollama-main-1'},
    {'host': 'localhost', 'port': 11435, 'name': 'ollama-main-2'},
    {'host': 'localhost', 'port': 11436, 'name': 'ollama-main-3'},
    
    # Second machine (48GB VRAM) - Update host IP as needed
    {'host': '192.168.1.100', 'port': 11437, 'name': 'ollama-secondary-1'},
    {'host': '192.168.1.100', 'port': 11438, 'name': 'ollama-secondary-2'},
]


async def test_manager():
    """Test the Ollama manager"""
    manager = OllamaManager(DEFAULT_INSTANCES[:1])  # Test with first instance only
    await manager.start()
    
    # Test articles
    test_articles = [
        {'pmc_id': 'PMC123', 'title': 'Test Article 1', 'body': 'Sample content...'},
        {'pmc_id': 'PMC124', 'title': 'Test Article 2', 'body': 'More content...'}
    ]
    
    prompt = "Extract all statistics from the articles above."
    
    try:
        results = await manager.process_batch(test_articles, prompt)
        print(f"Processed {len(results)} articles")
        print(f"Status: {manager.get_status()}")
    finally:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(test_manager())
