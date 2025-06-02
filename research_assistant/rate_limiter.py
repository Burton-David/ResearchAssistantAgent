"""
Rate limiting utilities for API calls.

Implements token bucket algorithm for rate limiting with async support.
Based on best practices from "Rate Limiting Strategies and Techniques" (Kong, 2021).
"""

import asyncio
import time
from typing import Optional, Callable, Dict, Any, Tuple
import logging
from dataclasses import dataclass, field
from enum import Enum
import random

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    
    This implementation allows bursts while maintaining average rate limits,
    which is ideal for API interactions that may have sporadic usage patterns.
    """
    
    def __init__(
        self,
        max_calls: int,
        time_window: float,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time window
            time_window: Time window in seconds
            burst_size: Maximum burst size (defaults to max_calls)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.burst_size = burst_size or max_calls
        
        # Token bucket state
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        
        # Calculate token generation rate
        self.token_rate = max_calls / time_window
        
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket, waiting if necessary.
        
        Args:
            tokens: Number of tokens to acquire (default 1)
            
        Raises:
            ValueError: If requesting more tokens than burst size
        """
        if tokens > self.burst_size:
            raise ValueError(f"Cannot acquire {tokens} tokens, burst size is {self.burst_size}")
            
        async with self.lock:
            # Update token count based on time elapsed
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.burst_size, self.tokens + elapsed * self.token_rate)
            self.last_update = now
            
            # Wait if not enough tokens
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.token_rate
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                
                # Update tokens after wait
                self.tokens = min(self.burst_size, self.tokens + wait_time * self.token_rate)
                self.last_update = time.monotonic()
                
            # Consume tokens
            self.tokens -= tokens
            
    async def __aenter__(self):
        """Context manager entry."""
        await self.acquire()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
        
    @property
    def available_tokens(self) -> float:
        """Get current number of available tokens."""
        now = time.monotonic()
        elapsed = now - self.last_update
        return min(self.burst_size, self.tokens + elapsed * self.token_rate)
        
    def reset(self) -> None:
        """Reset the rate limiter to full capacity."""
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()


class MultiServiceRateLimiter:
    """
    Manages rate limiting for multiple services with different limits.
    
    Useful when your application interacts with multiple APIs, each with
    their own rate limiting requirements.
    """
    
    def __init__(self):
        """Initialize multi-service rate limiter."""
        self.limiters: dict[str, RateLimiter] = {}
        
    def add_service(
        self,
        service_name: str,
        max_calls: int,
        time_window: float,
        burst_size: Optional[int] = None
    ) -> None:
        """
        Add a service with its rate limiting configuration.
        
        Args:
            service_name: Unique identifier for the service
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
            burst_size: Maximum burst size
        """
        self.limiters[service_name] = RateLimiter(max_calls, time_window, burst_size)
        logger.info(f"Added rate limiter for {service_name}: {max_calls} calls per {time_window}s")
        
    async def acquire(self, service_name: str, tokens: int = 1) -> None:
        """
        Acquire tokens for a specific service.
        
        Args:
            service_name: Service to acquire tokens for
            tokens: Number of tokens to acquire
            
        Raises:
            KeyError: If service not registered
        """
        if service_name not in self.limiters:
            raise KeyError(f"Service '{service_name}' not registered")
            
        await self.limiters[service_name].acquire(tokens)
        
    def get_limiter(self, service_name: str) -> RateLimiter:
        """
        Get rate limiter for a specific service.
        
        Args:
            service_name: Service name
            
        Returns:
            RateLimiter instance for the service
            
        Raises:
            KeyError: If service not registered
        """
        if service_name not in self.limiters:
            raise KeyError(f"Service '{service_name}' not registered")
            
        return self.limiters[service_name]


class AdaptiveRateLimiter(RateLimiter):
    """
    Rate limiter that adapts based on server responses.
    
    Implements exponential backoff when rate limit errors are encountered,
    and gradually increases rate when successful.
    """
    
    def __init__(
        self,
        initial_rate: float,
        time_window: float = 1.0,
        min_rate: float = 0.1,
        max_rate: Optional[float] = None,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1
    ):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_rate: Initial requests per time window
            time_window: Time window in seconds
            min_rate: Minimum rate (won't go below this)
            max_rate: Maximum rate (won't go above this)
            backoff_factor: Multiplier when rate limit hit (< 1.0)
            recovery_factor: Multiplier for successful requests (> 1.0)
        """
        super().__init__(int(initial_rate), time_window, int(initial_rate))
        
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate or initial_rate * 10
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        
    def on_rate_limit_error(self) -> None:
        """Called when a rate limit error is encountered."""
        old_rate = self.current_rate
        self.current_rate = max(self.min_rate, self.current_rate * self.backoff_factor)
        
        # Update the parent class parameters
        self.max_calls = int(self.current_rate)
        self.burst_size = int(self.current_rate)
        self.token_rate = self.current_rate / self.time_window
        
        logger.warning(f"Rate limit hit, reducing rate from {old_rate:.2f} to {self.current_rate:.2f}")
        
    def on_success(self) -> None:
        """Called on successful request."""
        if self.current_rate < self.max_rate:
            old_rate = self.current_rate
            self.current_rate = min(self.max_rate, self.current_rate * self.recovery_factor)
            
            # Update the parent class parameters
            self.max_calls = int(self.current_rate)
            self.burst_size = int(self.current_rate)
            self.token_rate = self.current_rate / self.time_window
            
            logger.debug(f"Increasing rate from {old_rate:.2f} to {self.current_rate:.2f}")


class APIType(Enum):
    """Supported API types."""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    CROSSREF = "crossref"


@dataclass
class APIRateLimitConfig:
    """Configuration for API rate limits."""
    requests_per_second: float
    requests_per_window: Optional[int] = None
    window_seconds: Optional[float] = None
    burst_size: Optional[int] = None
    requires_api_key: bool = False
    
    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = int(self.requests_per_second * 2)


class UnifiedRateLimiter:
    """
    Unified rate limiter for all API services with exponential backoff.
    
    Handles different rate limits for each service and implements
    exponential backoff when rate limits are hit.
    """
    
    # Default configurations for each API
    DEFAULT_CONFIGS = {
        APIType.ARXIV: APIRateLimitConfig(
            requests_per_second=0.33,  # 1 request per 3 seconds
            burst_size=1
        ),
        APIType.SEMANTIC_SCHOLAR: {
            "with_key": APIRateLimitConfig(
                requests_per_second=1.0,
                burst_size=5,
                requires_api_key=True
            ),
            "without_key": APIRateLimitConfig(
                requests_per_second=0.33,  # 100 per 5 min = 20 per min = 0.33 per sec
                requests_per_window=100,
                window_seconds=300,
                burst_size=5
            )
        },
        APIType.PUBMED: {
            "with_key": APIRateLimitConfig(
                requests_per_second=10.0,
                burst_size=20,
                requires_api_key=True
            ),
            "without_key": APIRateLimitConfig(
                requests_per_second=3.0,
                burst_size=10
            )
        },
        APIType.CROSSREF: APIRateLimitConfig(
            requests_per_second=50.0,
            burst_size=100
        )
    }
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize unified rate limiter.
        
        Args:
            api_keys: Dictionary of API keys by service name
        """
        self.api_keys = api_keys or {}
        self.limiters: Dict[APIType, RateLimiter] = {}
        self.backoff_state: Dict[APIType, Dict[str, Any]] = {}
        self._init_limiters()
        
    def _init_limiters(self):
        """Initialize rate limiters for each API service."""
        for api_type in APIType:
            config = self._get_config(api_type)
            
            if config.requests_per_window and config.window_seconds:
                # Use window-based limiting
                limiter = RateLimiter(
                    max_calls=config.requests_per_window,
                    time_window=config.window_seconds,
                    burst_size=config.burst_size
                )
            else:
                # Use rate-based limiting
                limiter = AdaptiveRateLimiter(
                    initial_rate=config.requests_per_second,
                    time_window=1.0,
                    min_rate=0.1,
                    max_rate=config.requests_per_second * 2,
                    backoff_factor=0.5,
                    recovery_factor=1.1
                )
            
            self.limiters[api_type] = limiter
            self.backoff_state[api_type] = {
                "consecutive_errors": 0,
                "last_error_time": 0,
                "backoff_until": 0
            }
            
    def _get_config(self, api_type: APIType) -> APIRateLimitConfig:
        """Get configuration for API type considering API key availability."""
        config = self.DEFAULT_CONFIGS[api_type]
        
        if isinstance(config, dict):
            # Service has different limits with/without API key
            has_key = api_type.value in self.api_keys
            return config["with_key" if has_key else "without_key"]
        else:
            return config
            
    async def acquire(self, api_type: APIType, tokens: int = 1) -> None:
        """
        Acquire rate limit tokens with exponential backoff support.
        
        Args:
            api_type: API service type
            tokens: Number of tokens to acquire
            
        Raises:
            ValueError: If API type not supported
        """
        if api_type not in self.limiters:
            raise ValueError(f"Unsupported API type: {api_type}")
            
        # Check if we're in backoff period
        backoff = self.backoff_state[api_type]
        if backoff["backoff_until"] > time.time():
            wait_time = backoff["backoff_until"] - time.time()
            logger.info(f"In backoff period for {api_type.value}, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
        # Acquire from rate limiter
        await self.limiters[api_type].acquire(tokens)
        
    def on_success(self, api_type: APIType):
        """
        Called on successful API request.
        
        Args:
            api_type: API service type
        """
        # Reset consecutive errors
        self.backoff_state[api_type]["consecutive_errors"] = 0
        
        # If using adaptive rate limiter, notify success
        limiter = self.limiters[api_type]
        if isinstance(limiter, AdaptiveRateLimiter):
            limiter.on_success()
            
    def on_rate_limit_error(self, api_type: APIType, retry_after: Optional[int] = None):
        """
        Called when rate limit error (429) is encountered.
        
        Args:
            api_type: API service type
            retry_after: Optional retry-after header value in seconds
        """
        backoff = self.backoff_state[api_type]
        backoff["consecutive_errors"] += 1
        backoff["last_error_time"] = time.time()
        
        # Calculate backoff time
        if retry_after:
            # Use server-provided retry time
            wait_time = retry_after
        else:
            # Exponential backoff: 2^n seconds with jitter
            base_wait = min(2 ** backoff["consecutive_errors"], 300)  # Max 5 minutes
            jitter = random.uniform(0, base_wait * 0.1)  # 10% jitter
            wait_time = base_wait + jitter
            
        backoff["backoff_until"] = time.time() + wait_time
        
        logger.warning(
            f"Rate limit hit for {api_type.value} "
            f"(attempt {backoff['consecutive_errors']}), "
            f"backing off for {wait_time:.2f}s"
        )
        
        # If using adaptive rate limiter, notify error
        limiter = self.limiters[api_type]
        if isinstance(limiter, AdaptiveRateLimiter):
            limiter.on_rate_limit_error()
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics for all services."""
        metrics = {}
        
        for api_type in APIType:
            limiter = self.limiters[api_type]
            backoff = self.backoff_state[api_type]
            
            metrics[api_type.value] = {
                "available_tokens": limiter.available_tokens,
                "consecutive_errors": backoff["consecutive_errors"],
                "in_backoff": backoff["backoff_until"] > time.time(),
                "backoff_remaining": max(0, backoff["backoff_until"] - time.time())
            }
            
            if isinstance(limiter, AdaptiveRateLimiter):
                metrics[api_type.value]["current_rate"] = limiter.current_rate
                
        return metrics
        
    def reset(self, api_type: Optional[APIType] = None):
        """
        Reset rate limiter state.
        
        Args:
            api_type: Specific API to reset, or None for all
        """
        if api_type:
            self.limiters[api_type].reset()
            self.backoff_state[api_type] = {
                "consecutive_errors": 0,
                "last_error_time": 0,
                "backoff_until": 0
            }
        else:
            for api in APIType:
                self.reset(api)


def rate_limited(
    max_calls: int,
    time_window: float,
    limiter_attr: str = "_rate_limiter"
) -> Callable:
    """
    Decorator for rate limiting async methods.
    
    Args:
        max_calls: Maximum calls in time window
        time_window: Time window in seconds
        limiter_attr: Attribute name to store limiter on instance
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        async def wrapper(self, *args, **kwargs):
            # Get or create rate limiter for this instance
            if not hasattr(self, limiter_attr):
                setattr(self, limiter_attr, RateLimiter(max_calls, time_window))
                
            limiter = getattr(self, limiter_attr)
            
            # Acquire token before calling function
            async with limiter:
                return await func(self, *args, **kwargs)
                
        return wrapper
    return decorator