"""
Custom exception hierarchy for better error handling and recovery.
"""

from typing import Optional, Dict, Any, List
import logging
import asyncio

logger = logging.getLogger(__name__)


class ResearchAssistantError(Exception):
    """Base exception for all Research Assistant errors."""
    
    def __init__(
        self, 
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Optional error code for categorization
            details: Optional details dictionary
            recoverable: Whether error is recoverable
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.recoverable = recoverable
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable
        }


# Configuration Errors
class ConfigurationError(ResearchAssistantError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            details={"config_key": config_key} if config_key else {},
            recoverable=False
        )


class MissingAPIKeyError(ConfigurationError):
    """API key not found."""
    
    def __init__(self, service: str):
        super().__init__(
            message=f"API key for {service} not found. Run 'research-assistant configure'",
            config_key=f"{service}_api_key"
        )
        self.service = service


# Validation Errors
class ValidationError(ResearchAssistantError):
    """Input validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={
                "field": field,
                "value": str(value)[:100] if value else None
            },
            recoverable=True
        )


# API Errors
class APIError(ResearchAssistantError):
    """External API errors."""
    
    def __init__(
        self, 
        message: str,
        service: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None
    ):
        super().__init__(
            message=message,
            error_code="API_ERROR",
            details={
                "service": service,
                "status_code": status_code,
                "response_snippet": response_body[:200] if response_body else None
            },
            recoverable=status_code not in [401, 403] if status_code else True
        )
        self.service = service
        self.status_code = status_code


class RateLimitError(APIError):
    """API rate limit exceeded."""
    
    def __init__(self, service: str, retry_after: Optional[int] = None):
        super().__init__(
            message=f"Rate limit exceeded for {service}",
            service=service,
            status_code=429
        )
        self.retry_after = retry_after
        self.details["retry_after"] = retry_after


class APITimeoutError(APIError):
    """API request timeout."""
    
    def __init__(self, service: str, timeout: float):
        super().__init__(
            message=f"Request to {service} timed out after {timeout}s",
            service=service
        )
        self.timeout = timeout


# Resource Errors
class ResourceError(ResearchAssistantError):
    """Resource-related errors."""
    
    def __init__(self, message: str, resource_type: str, limit: Any = None):
        super().__init__(
            message=message,
            error_code="RESOURCE_ERROR",
            details={
                "resource_type": resource_type,
                "limit": limit
            },
            recoverable=True
        )


class MemoryLimitError(ResourceError):
    """Memory limit exceeded."""
    
    def __init__(self, current_usage: float, limit: float):
        super().__init__(
            message=f"Memory usage ({current_usage:.1f}%) exceeds limit ({limit:.1f}%)",
            resource_type="memory",
            limit=limit
        )


class ConcurrencyLimitError(ResourceError):
    """Concurrency limit exceeded."""
    
    def __init__(self, operation: str, limit: int):
        super().__init__(
            message=f"Too many concurrent {operation} operations (limit: {limit})",
            resource_type="concurrency",
            limit=limit
        )


# Storage Errors
class StorageError(ResearchAssistantError):
    """Storage-related errors."""
    
    def __init__(self, message: str, path: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            details={"path": path} if path else {},
            recoverable=False
        )


class CorruptedDataError(StorageError):
    """Data corruption detected."""
    
    def __init__(self, path: str, reason: str):
        super().__init__(
            message=f"Corrupted data at {path}: {reason}",
            path=path
        )


# Processing Errors
class ProcessingError(ResearchAssistantError):
    """Data processing errors."""
    
    def __init__(self, message: str, operation: str, input_data: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="PROCESSING_ERROR",
            details={
                "operation": operation,
                "input_snippet": input_data[:100] if input_data else None
            },
            recoverable=True
        )


class EmbeddingError(ProcessingError):
    """Embedding generation errors."""
    
    def __init__(self, message: str, text_length: Optional[int] = None):
        super().__init__(
            message=message,
            operation="embedding"
        )
        if text_length:
            self.details["text_length"] = text_length


class ChunkingError(ProcessingError):
    """Text chunking errors."""
    
    def __init__(self, message: str, document_size: Optional[int] = None):
        super().__init__(
            message=message,
            operation="chunking"
        )
        if document_size:
            self.details["document_size"] = document_size


# Error Handlers
class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    @staticmethod
    def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Handle error with appropriate logging and recovery.
        
        Args:
            error: Exception to handle
            context: Optional context information
        """
        if isinstance(error, ResearchAssistantError):
            if error.recoverable:
                logger.warning(f"{error.error_code}: {error.message}", extra=error.details)
            else:
                logger.error(f"{error.error_code}: {error.message}", extra=error.details)
        else:
            logger.exception("Unexpected error", extra={"context": context})
            
    @staticmethod
    def get_recovery_strategy(error: Exception) -> Optional[str]:
        """
        Get recovery strategy for error.
        
        Args:
            error: Exception to analyze
            
        Returns:
            Recovery strategy name or None
        """
        if isinstance(error, RateLimitError):
            return "exponential_backoff"
        elif isinstance(error, APITimeoutError):
            return "retry_with_timeout"
        elif isinstance(error, MemoryLimitError):
            return "reduce_batch_size"
        elif isinstance(error, ConcurrencyLimitError):
            return "queue_operation"
        elif isinstance(error, ValidationError):
            return "request_valid_input"
        elif isinstance(error, MissingAPIKeyError):
            return "configure_api_key"
        elif isinstance(error, ResearchAssistantError) and error.recoverable:
            return "retry_operation"
        else:
            return None
            
    @staticmethod
    def create_user_message(error: Exception) -> str:
        """
        Create user-friendly error message.
        
        Args:
            error: Exception to describe
            
        Returns:
            User-friendly message
        """
        if isinstance(error, MissingAPIKeyError):
            return f"Please configure your {error.service} API key"
        elif isinstance(error, ValidationError):
            return f"Invalid input: {error.message}"
        elif isinstance(error, RateLimitError):
            retry_msg = f" Retry after {error.retry_after}s" if error.retry_after else ""
            return f"Rate limit reached for {error.service}.{retry_msg}"
        elif isinstance(error, MemoryLimitError):
            return "System memory is running low. Please try again with smaller batches"
        elif isinstance(error, APITimeoutError):
            return f"Request timed out. The {error.service} service may be slow"
        elif isinstance(error, StorageError):
            return f"Storage error: {error.message}"
        elif isinstance(error, ResearchAssistantError):
            return error.message
        else:
            return "An unexpected error occurred. Please check the logs"


def ensure_no_silent_failures(func):
    """
    Decorator to ensure no silent failures.
    
    Logs all exceptions and re-raises them.
    """
    from functools import wraps
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.handle_error(e, {"function": func.__name__, "args": str(args)[:100]})
            raise
            
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            ErrorHandler.handle_error(e, {"function": func.__name__, "args": str(args)[:100]})
            raise
            
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper