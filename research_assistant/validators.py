"""
Input validation and sanitization for production safety.

Prevents injection attacks, directory traversal, and resource exhaustion.
"""

import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import unicodedata
import logging

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class Validators:
    """Centralized input validation for security and safety."""
    
    # Limits
    MAX_QUERY_LENGTH = 1000
    MAX_PATH_LENGTH = 4096
    MAX_FIELD_LENGTH = 500
    MAX_METADATA_SIZE = 10240  # 10KB
    MAX_ARRAY_LENGTH = 1000
    MAX_RESULTS_LIMIT = 100
    
    # Patterns
    ARXIV_ID_PATTERN = re.compile(r'^[0-9]{4}\.[0-9]{4,5}(v[0-9]+)?$')
    SEMANTIC_SCHOLAR_ID_PATTERN = re.compile(r'^[a-f0-9]{40}$')
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    # Dangerous patterns for queries
    INJECTION_PATTERNS = [
        re.compile(r'[<>]'),  # HTML tags
        re.compile(r'javascript:', re.I),  # JS injection
        re.compile(r'(script|eval|exec)\s*\(', re.I),  # Code execution
        re.compile(r'\$\{.*\}'),  # Template injection
        re.compile(r'{{.*}}'),  # Template injection
        re.compile(r'%\{.*\}'),  # Template injection
        re.compile(r"[';]", re.I),  # SQL injection
        re.compile(r'(\bDROP\b|\bDELETE\b|\bINSERT\b|\bUPDATE\b|\bUNION\b|\bSELECT\b)', re.I),  # SQL keywords
        re.compile(r'--\s*$'),  # SQL comment
        re.compile(r'&&|\|\|'),  # Command chaining
        re.compile(r'\\0|\\x00'),  # Null bytes
    ]
    
    @classmethod
    def validate_query(cls, query: str, max_length: Optional[int] = None) -> str:
        """
        Validate and sanitize search query.
        
        Args:
            query: User search query
            max_length: Maximum allowed length
            
        Returns:
            Sanitized query
            
        Raises:
            ValidationError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise ValidationError("Query must be a non-empty string")
            
        # Normalize unicode
        query = unicodedata.normalize('NFKC', query)
        
        # Strip whitespace
        query = query.strip()
        
        # Check length
        max_len = max_length or cls.MAX_QUERY_LENGTH
        if len(query) > max_len:
            raise ValidationError(f"Query exceeds maximum length of {max_len} characters")
            
        # Check for empty after stripping
        if not query:
            raise ValidationError("Query cannot be empty or just whitespace")
            
        # Check for injection patterns
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(query):
                raise ValidationError("Query contains potentially dangerous characters")
                
        # Remove multiple spaces
        query = ' '.join(query.split())
        
        return query
        
    @classmethod
    def validate_path(cls, path: Union[str, Path], must_exist: bool = False,
                     allow_create: bool = False, base_path: Optional[Path] = None) -> Path:
        """
        Validate file path to prevent directory traversal attacks.
        
        Args:
            path: Path to validate
            must_exist: Whether path must already exist
            allow_create: Whether path can be created
            base_path: Optional base path for restriction
            
        Returns:
            Validated Path object
            
        Raises:
            ValidationError: If path is invalid or unsafe
        """
        if not path:
            raise ValidationError("Path cannot be empty")
            
        try:
            path = Path(path)
        except Exception as e:
            raise ValidationError(f"Invalid path: {e}")
            
        # Resolve to absolute path
        try:
            path = path.resolve()
        except Exception as e:
            raise ValidationError(f"Cannot resolve path: {e}")
            
        # Check length
        if len(str(path)) > cls.MAX_PATH_LENGTH:
            raise ValidationError(f"Path exceeds maximum length of {cls.MAX_PATH_LENGTH}")
            
        # Check for null bytes
        if '\x00' in str(path):
            raise ValidationError("Path contains null bytes")
            
        # Prevent directory traversal
        if '..' in path.parts:
            raise ValidationError("Path contains directory traversal attempts")
            
        # Check against base path if provided
        if base_path:
            base_path = Path(base_path).resolve()
            try:
                path.relative_to(base_path)
            except ValueError:
                raise ValidationError(f"Path must be within {base_path}")
                
        # Check existence
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")
            
        if not allow_create and not must_exist:
            # Check parent exists at least
            if not path.parent.exists():
                raise ValidationError(f"Parent directory does not exist: {path.parent}")
                
        return path
        
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata dictionary for safety.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Validated metadata
            
        Raises:
            ValidationError: If metadata is invalid
        """
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")
            
        # Check total size
        import json
        try:
            size = len(json.dumps(metadata))
            if size > cls.MAX_METADATA_SIZE:
                raise ValidationError(f"Metadata exceeds size limit of {cls.MAX_METADATA_SIZE} bytes")
        except Exception as e:
            raise ValidationError(f"Invalid metadata structure: {e}")
            
        validated = {}
        
        for key, value in metadata.items():
            # Validate key
            if not isinstance(key, str):
                raise ValidationError(f"Metadata key must be string, got {type(key)}")
                
            if len(key) > cls.MAX_FIELD_LENGTH:
                raise ValidationError(f"Metadata key '{key[:20]}...' too long")
                
            # Check for dangerous key names
            if key.startswith('__') or key.startswith('$'):
                raise ValidationError(f"Metadata key '{key}' uses reserved prefix")
                
            # Validate value
            validated[key] = cls._validate_metadata_value(value, f"metadata['{key}']")
            
        return validated
        
    @classmethod
    def _validate_metadata_value(cls, value: Any, path: str) -> Any:
        """Recursively validate metadata values."""
        if value is None:
            return value
            
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and len(value) > cls.MAX_FIELD_LENGTH:
                raise ValidationError(f"{path} string value too long")
            return value
            
        if isinstance(value, list):
            if len(value) > cls.MAX_ARRAY_LENGTH:
                raise ValidationError(f"{path} array too long ({len(value)} items)")
            return [cls._validate_metadata_value(item, f"{path}[{i}]") 
                   for i, item in enumerate(value)]
                   
        if isinstance(value, dict):
            if len(value) > cls.MAX_ARRAY_LENGTH:
                raise ValidationError(f"{path} object has too many keys ({len(value)})")
            return {k: cls._validate_metadata_value(v, f"{path}.{k}") 
                   for k, v in value.items()}
                   
        raise ValidationError(f"{path} has unsupported type: {type(value)}")
        
    @classmethod
    def validate_limit(cls, limit: Any, max_limit: Optional[int] = None) -> int:
        """
        Validate a limit parameter.
        
        Args:
            limit: Limit value to validate
            max_limit: Maximum allowed limit
            
        Returns:
            Validated integer limit
            
        Raises:
            ValidationError: If limit is invalid
        """
        max_limit = max_limit or cls.MAX_RESULTS_LIMIT
        
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise ValidationError("Limit must be an integer")
            
        if limit <= 0:
            raise ValidationError("Limit must be positive")
            
        if limit > max_limit:
            raise ValidationError(f"Limit exceeds maximum of {max_limit}")
            
        return limit
        
    @classmethod
    def validate_paper_id(cls, paper_id: str, source: str) -> str:
        """
        Validate paper ID based on source.
        
        Args:
            paper_id: Paper identifier
            source: Source system (arxiv, semantic_scholar)
            
        Returns:
            Validated paper ID
            
        Raises:
            ValidationError: If paper ID is invalid
        """
        if not paper_id or not isinstance(paper_id, str):
            raise ValidationError("Paper ID must be a non-empty string")
            
        paper_id = paper_id.strip()
        
        if source == "arxiv":
            if not cls.ARXIV_ID_PATTERN.match(paper_id):
                raise ValidationError(f"Invalid ArXiv ID format: {paper_id}")
                
        elif source == "semantic_scholar":
            if not cls.SEMANTIC_SCHOLAR_ID_PATTERN.match(paper_id):
                raise ValidationError(f"Invalid Semantic Scholar ID format: {paper_id}")
                
        else:
            # Generic validation
            if len(paper_id) > 100:
                raise ValidationError("Paper ID too long")
                
            if not re.match(r'^[a-zA-Z0-9._-]+$', paper_id):
                raise ValidationError("Paper ID contains invalid characters")
                
        return paper_id
        
    @classmethod
    def validate_url(cls, url: str, allowed_schemes: Optional[List[str]] = None) -> str:
        """
        Validate URL for safety.
        
        Args:
            url: URL to validate
            allowed_schemes: Allowed URL schemes
            
        Returns:
            Validated URL
            
        Raises:
            ValidationError: If URL is invalid or unsafe
        """
        if not url or not isinstance(url, str):
            raise ValidationError("URL must be a non-empty string")
            
        allowed_schemes = allowed_schemes or ['http', 'https']
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL: {e}")
            
        if parsed.scheme not in allowed_schemes:
            raise ValidationError(f"URL scheme must be one of: {allowed_schemes}")
            
        if not parsed.netloc:
            raise ValidationError("URL must have a valid domain")
            
        # Check for localhost/internal IPs (SSRF prevention)
        if parsed.hostname in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValidationError("URL points to localhost")
            
        # Check for private IP ranges
        import ipaddress
        try:
            ip = ipaddress.ip_address(parsed.hostname)
            if ip.is_private or ip.is_reserved or ip.is_loopback:
                raise ValidationError("URL points to private/reserved IP")
        except ValueError:
            # Not an IP address, hostname is OK
            pass
            
        return url
        
    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
            
        Raises:
            ValidationError: If filename cannot be sanitized
        """
        if not filename or not isinstance(filename, str):
            raise ValidationError("Filename must be a non-empty string")
            
        # Get base name without path
        filename = os.path.basename(filename)
        
        # Replace dangerous characters
        filename = re.sub(r'[^\w\s.-]', '_', filename)
        filename = re.sub(r'\.+', '.', filename)  # Multiple dots
        filename = re.sub(r'^\.+', '', filename)  # Leading dots
        filename = filename.strip()
        
        if not filename:
            raise ValidationError("Filename empty after sanitization")
            
        if len(filename) > 255:
            raise ValidationError("Filename too long")
            
        # Check against reserved names
        reserved = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        name_without_ext = filename.split('.')[0].upper()
        if name_without_ext in reserved:
            raise ValidationError(f"Filename '{filename}' is reserved")
            
        return filename