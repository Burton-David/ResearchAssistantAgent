# Security Design Decisions - Research Assistant Agent

## Overview
This document explains the security decisions made during the hardening of the Research Assistant Agent project. Each decision was made based on security best practices and the OWASP guidelines.

## 1. Data Serialization: JSON vs Pickle

### Decision: Replace Pickle with JSON
**File**: `vector_store.py`

### Rationale:
- **Security Risk**: Python's pickle module can execute arbitrary code during deserialization
- **Attack Vector**: Malicious pickle files could compromise the entire system
- **Solution**: JSON is safe by design - it can only deserialize data, not code

### Implementation:
```python
# Before (UNSAFE):
with open(path / "documents.pkl", "rb") as f:
    data = pickle.load(f)  # Can execute arbitrary code!

# After (SAFE):
with open(path / "documents.json", "r") as f:
    data = json.load(f)  # Only deserializes data
```

### Trade-offs:
- ✅ Eliminates code execution vulnerability
- ✅ Human-readable format aids debugging
- ❌ Slightly larger file sizes
- ❌ Cannot serialize complex Python objects (mitigated with to_dict/from_dict methods)

## 2. XML Parsing: defusedxml vs Standard Library

### Decision: Use defusedxml for all XML parsing
**File**: `arxiv_collector.py`

### Rationale:
- **XXE Attacks**: Standard XML parsers vulnerable to XML External Entity attacks
- **Billion Laughs**: Exponential entity expansion can cause DoS
- **Resource Exhaustion**: Malicious XML can consume excessive memory

### Implementation:
```python
# Safe XML parsing setup
try:
    import defusedxml.ElementTree as ET
except ImportError:
    from xml.etree import ElementTree as ET
    # Configure ET to be more secure
    ET.XMLParser = ET.XMLParser(
        resolve_entities=False,
        forbid_dtd=True,
        forbid_entities=True,
        forbid_external=True
    )
```

### Additional Protections:
- Size validation (10MB limit)
- Input type checking
- Proper error handling

## 3. API Key Storage: Encryption at Rest

### Decision: Encrypt API keys using cryptography library
**File**: `security.py`

### Rationale:
- **Compliance**: Many regulations require encryption of sensitive data at rest
- **Defense in Depth**: Even if config files are exposed, keys remain protected
- **User Trust**: Demonstrates commitment to security

### Implementation:
- Machine-specific key derivation using PBKDF2
- Fernet symmetric encryption
- Restrictive file permissions (0600)
- No hardcoded keys or salts

### Key Derivation:
```python
# Combines multiple sources for uniqueness
sources = [
    str(Path.home()),           # User home directory
    os.getenv("USER", "default"), # Username
    machine_id                   # Platform-specific ID
]
```

## 4. Input Validation Strategy

### Decision: Comprehensive validation for all user inputs
**File**: `validators.py`

### Rationale:
- **Injection Prevention**: Block SQL injection, XSS, template injection
- **Path Traversal**: Prevent directory traversal attacks
- **Resource Protection**: Enforce size and length limits

### Validation Rules:
1. **Queries**: Max 1000 chars, no HTML/JS, no template syntax
2. **Paths**: No "..", must be within base path, no null bytes
3. **Metadata**: Max 10KB total, recursive validation
4. **URLs**: No localhost/private IPs (SSRF prevention)

### Example Patterns Blocked:
```python
INJECTION_PATTERNS = [
    re.compile(r'[<>]'),              # HTML tags
    re.compile(r'javascript:', re.I),  # JS injection
    re.compile(r'\$\{.*\}'),          # Template injection
    re.compile(r'{{.*}}'),            # Jinja2 templates
]
```

## 5. Resource Management

### Decision: Implement comprehensive resource limits
**File**: `resource_manager.py`

### Rationale:
- **DoS Prevention**: Prevent memory/CPU exhaustion
- **Fair Usage**: Ensure system remains responsive
- **Cost Control**: Limit API usage
- **Monitoring**: Track resource usage

### Limits Implemented:
- Max concurrent requests: 10
- Max concurrent embeddings: 3
- Memory usage limit: 80%
- Request size: 10MB
- Response size: 50MB
- Default timeout: 30s

### Monitoring:
```python
# Real-time resource tracking
- Memory usage percentage
- CPU usage
- Active operations count
- Error rates
```

## 6. Error Handling Philosophy

### Decision: No silent failures, structured exceptions
**File**: `exceptions.py`

### Rationale:
- **Debugging**: Silent failures hide problems
- **Recovery**: Structured errors enable recovery strategies
- **User Experience**: Clear error messages help users

### Exception Hierarchy:
```
ResearchAssistantError (base)
├── ConfigurationError
│   └── MissingAPIKeyError
├── ValidationError
├── APIError
│   ├── RateLimitError
│   └── APITimeoutError
├── ResourceError
│   ├── MemoryLimitError
│   └── ConcurrencyLimitError
└── StorageError
    └── CorruptedDataError
```

## 7. Rate Limiting Design

### Decision: Token bucket algorithm with burst support
**File**: `rate_limiter.py`

### Rationale:
- **API Compliance**: Respect service rate limits
- **Burst Support**: Allow short bursts of activity
- **Fair Usage**: Smooth out request patterns

### Implementation:
- ArXiv: 3 requests/second
- Semantic Scholar: 10 requests/second
- Exponential backoff on 429 errors

## 8. Secure Defaults

### Decision: Security by default configuration

### Examples:
- Embeddings use local models by default (no API keys required)
- Strict input validation enabled
- Resource limits enforced
- Secure file permissions
- HTTPS only for API calls

## 9. Logging and Monitoring

### Decision: Structured logging without sensitive data

### Rules:
- Never log API keys or tokens
- Truncate long inputs in logs
- Include correlation IDs
- Log security events (validation failures, rate limits)

## 10. Future Security Considerations

### Recommended Additions:
1. **API Gateway**: Add authentication/authorization layer
2. **Audit Logging**: Track all data access
3. **Encryption in Transit**: TLS for all communications
4. **Key Rotation**: Automated API key rotation
5. **Security Headers**: If web interface added
6. **CORS Policy**: Restrict cross-origin requests
7. **Input Fuzzing**: Automated security testing

## Conclusion

These security decisions transform the Research Assistant Agent from a prototype into a production-ready system. The principle of "defense in depth" was applied throughout, ensuring multiple layers of protection against various attack vectors.

The security enhancements do not significantly impact performance or usability, while dramatically improving the system's resilience against both accidental and malicious inputs.