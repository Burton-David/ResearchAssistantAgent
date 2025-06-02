"""Tests for the RateLimiter class."""

import asyncio
import time
import pytest
from research_assistant.rate_limiter import RateLimiter


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiting functionality."""
        # 2 calls per second with burst of 2
        limiter = RateLimiter(max_calls=2, time_window=1.0, burst_size=2)
        
        # Should allow first two requests immediately
        start = time.time()
        await limiter.acquire()
        await limiter.acquire()
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be nearly instant
        
        # Third request should be delayed
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        assert elapsed >= 0.4  # Should wait ~0.5 seconds
    
    @pytest.mark.asyncio
    async def test_rate_limiter_refill(self):
        """Test token bucket refill mechanism."""
        # 10 calls per second with burst of 2
        limiter = RateLimiter(max_calls=10, time_window=1.0, burst_size=2)
        
        # Use all tokens
        await limiter.acquire()
        await limiter.acquire()
        
        # Wait for partial refill
        await asyncio.sleep(0.05)  # 0.5 tokens should refill
        
        # This should still need to wait
        start = time.time()
        await limiter.acquire()
        elapsed = time.time() - start
        
        assert elapsed >= 0.04  # Should wait for remaining 0.5 tokens
    
    def test_rate_limiter_configuration(self):
        """Test rate limiter configuration."""
        limiter = RateLimiter(max_calls=5, time_window=1.0, burst_size=10)
        
        assert limiter.max_calls == 5
        assert limiter.burst_size == 10
        assert limiter.tokens == 10.0