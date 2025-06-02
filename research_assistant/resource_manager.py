"""
Resource management for production safety.

Controls concurrency, monitors memory usage, and enforces limits.
"""

import asyncio
import threading
import psutil
import time
import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps
from contextlib import asynccontextmanager, contextmanager
import os

logger = logging.getLogger(__name__)


class ResourceExhaustedError(Exception):
    """Raised when resource limits are exceeded."""
    pass


class ResourceManager:
    """Manages system resources to prevent exhaustion."""
    
    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_concurrent_embeddings: int = 3,
        max_memory_percent: float = 80.0,
        max_request_size: int = 10 * 1024 * 1024,  # 10MB
        max_response_size: int = 50 * 1024 * 1024,  # 50MB
        default_timeout: float = 30.0
    ):
        """
        Initialize resource manager.
        
        Args:
            max_concurrent_requests: Maximum concurrent API requests
            max_concurrent_embeddings: Maximum concurrent embedding operations
            max_memory_percent: Maximum memory usage percentage
            max_request_size: Maximum request size in bytes
            max_response_size: Maximum response size in bytes
            default_timeout: Default operation timeout in seconds
        """
        # Store limits
        self.max_concurrent_requests = max_concurrent_requests
        self.max_concurrent_embeddings = max_concurrent_embeddings
        
        # Semaphores for concurrency control
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._embedding_semaphore = asyncio.Semaphore(max_concurrent_embeddings)
        self._sync_request_semaphore = threading.Semaphore(max_concurrent_requests)
        self._sync_embedding_semaphore = threading.Semaphore(max_concurrent_embeddings)
        
        # Resource limits
        self.max_memory_percent = max_memory_percent
        self.max_request_size = max_request_size
        self.max_response_size = max_response_size
        self.default_timeout = default_timeout
        
        # Monitoring
        self._start_time = time.time()
        self._request_count = 0
        self._error_count = 0
        self._last_memory_check = 0
        self._memory_check_interval = 1.0  # seconds
        
    def check_memory(self) -> None:
        """
        Check current memory usage.
        
        Raises:
            ResourceExhaustedError: If memory usage exceeds limit
        """
        current_time = time.time()
        if current_time - self._last_memory_check < self._memory_check_interval:
            return
            
        self._last_memory_check = current_time
        
        try:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.max_memory_percent:
                raise ResourceExhaustedError(
                    f"Memory usage ({memory_percent:.1f}%) exceeds limit ({self.max_memory_percent}%)"
                )
        except Exception as e:
            logger.warning(f"Failed to check memory: {e}")
            
    def check_request_size(self, size: int) -> None:
        """
        Check if request size is within limits.
        
        Args:
            size: Request size in bytes
            
        Raises:
            ResourceExhaustedError: If size exceeds limit
        """
        if size > self.max_request_size:
            raise ResourceExhaustedError(
                f"Request size ({size} bytes) exceeds limit ({self.max_request_size} bytes)"
            )
            
    def check_response_size(self, size: int) -> None:
        """
        Check if response size is within limits.
        
        Args:
            size: Response size in bytes
            
        Raises:
            ResourceExhaustedError: If size exceeds limit
        """
        if size > self.max_response_size:
            raise ResourceExhaustedError(
                f"Response size ({size} bytes) exceeds limit ({self.max_response_size} bytes)"
            )
            
    @asynccontextmanager
    async def limit_api_request(self, timeout: Optional[float] = None):
        """
        Async context manager for API request limiting.
        
        Args:
            timeout: Optional timeout override
            
        Yields:
            None
            
        Raises:
            ResourceExhaustedError: If resources exhausted
            asyncio.TimeoutError: If operation times out
        """
        self.check_memory()
        timeout = timeout or self.default_timeout
        
        try:
            async with asyncio.timeout(timeout):
                async with self._request_semaphore:
                    self._request_count += 1
                    logger.debug(f"API request started (total: {self._request_count})")
                    yield
        except asyncio.TimeoutError:
            self._error_count += 1
            logger.error(f"API request timed out after {timeout}s")
            raise
        except Exception:
            self._error_count += 1
            raise
            
    @asynccontextmanager
    async def limit_embedding_operation(self, timeout: Optional[float] = None):
        """
        Async context manager for embedding operation limiting.
        
        Args:
            timeout: Optional timeout override
            
        Yields:
            None
            
        Raises:
            ResourceExhaustedError: If resources exhausted
            asyncio.TimeoutError: If operation times out
        """
        self.check_memory()
        timeout = timeout or self.default_timeout * 2  # Embeddings take longer
        
        try:
            async with asyncio.timeout(timeout):
                async with self._embedding_semaphore:
                    logger.debug("Embedding operation started")
                    yield
        except asyncio.TimeoutError:
            self._error_count += 1
            logger.error(f"Embedding operation timed out after {timeout}s")
            raise
        except Exception:
            self._error_count += 1
            raise
            
    @contextmanager
    def limit_sync_operation(self, operation_type: str = "request", timeout: Optional[float] = None):
        """
        Sync context manager for resource limiting.
        
        Args:
            operation_type: Type of operation (request, embedding)
            timeout: Optional timeout override
            
        Yields:
            None
            
        Raises:
            ResourceExhaustedError: If resources exhausted
        """
        self.check_memory()
        
        semaphore = (self._sync_embedding_semaphore if operation_type == "embedding" 
                    else self._sync_request_semaphore)
        timeout = timeout or self.default_timeout
        
        acquired = semaphore.acquire(timeout=timeout)
        if not acquired:
            raise ResourceExhaustedError(f"Failed to acquire {operation_type} semaphore")
            
        try:
            logger.debug(f"Sync {operation_type} operation started")
            yield
        finally:
            semaphore.release()
            
    def with_timeout(self, timeout: Optional[float] = None):
        """
        Decorator for adding timeout to async functions.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                timeout_val = timeout or self.default_timeout
                try:
                    async with asyncio.timeout(timeout_val):
                        return await func(*args, **kwargs)
                except asyncio.TimeoutError:
                    logger.error(f"{func.__name__} timed out after {timeout_val}s")
                    raise
            return wrapper
        return decorator
        
    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        uptime = time.time() - self._start_time
        
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except:
            memory = None
            cpu_percent = None
            
        return {
            "uptime_seconds": uptime,
            "total_requests": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "memory_percent": memory.percent if memory else None,
            "memory_available_mb": memory.available / 1024 / 1024 if memory else None,
            "cpu_percent": cpu_percent,
            "max_concurrent_requests": self.max_concurrent_requests,
            "max_concurrent_embeddings": self.max_concurrent_embeddings,
        }
        

class RequestLimiter:
    """Rate limiter with resource awareness."""
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        rate_limiter: Any  # RateLimiter instance
    ):
        """
        Initialize request limiter.
        
        Args:
            resource_manager: ResourceManager instance
            rate_limiter: RateLimiter instance
        """
        self.resource_manager = resource_manager
        self.rate_limiter = rate_limiter
        
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire both rate limit token and resource limit.
        
        Args:
            timeout: Optional timeout
            
        Yields:
            None
        """
        async with self.resource_manager.limit_api_request(timeout):
            await self.rate_limiter.acquire()
            yield
            

# Global resource manager instance
_resource_manager = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance."""
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
    

def configure_resource_limits(
    max_concurrent_requests: int = 10,
    max_concurrent_embeddings: int = 3,
    max_memory_percent: float = 80.0,
    max_request_size: int = 10 * 1024 * 1024,
    max_response_size: int = 50 * 1024 * 1024,
    default_timeout: float = 30.0
) -> ResourceManager:
    """
    Configure global resource limits.
    
    Returns:
        Configured ResourceManager instance
    """
    global _resource_manager
    _resource_manager = ResourceManager(
        max_concurrent_requests=max_concurrent_requests,
        max_concurrent_embeddings=max_concurrent_embeddings,
        max_memory_percent=max_memory_percent,
        max_request_size=max_request_size,
        max_response_size=max_response_size,
        default_timeout=default_timeout
    )
    return _resource_manager