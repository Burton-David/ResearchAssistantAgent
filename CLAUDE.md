# CLAUDE.md - Project Configuration for ResearchAssistantAgent

## Python Environment Standards

- **Always use the virtual environment**: Check for `venv/` or `.venv/` and activate it before running Python commands
- **Python version**: Use Python 3.11+ features (type hints, async/await, dataclasses)
- **Package management**: 
  - Always update requirements.txt when adding new dependencies
  - Use `pip install -e .` for development installation
  - Include dev dependencies in requirements-dev.txt

## Code Style Guidelines

- Use type hints for all function parameters and returns
- Follow PEP 8 with 88-character line length (Black formatter style)
- Write self-documenting code with meaningful variable and function names
- Add docstrings for public APIs, but keep them concise
- Only comment non-obvious logic or important gotchas

## Professional Engineering Practices

### What TO Do:
- Write robust error handling with specific exception types
- Use design patterns appropriately (don't force them)
- Structure code for testability (dependency injection, pure functions)
- Implement proper retry logic with exponential backoff
- Use logging instead of print statements
- Cache expensive operations intelligently

### What NOT to Do:
- Don't add obvious comments like "# Initialize the logger"
- Avoid fake-sounding performance metrics in comments
- Don't over-engineer simple solutions
- Skip the "tutorial voice" in documentation
- No placeholder names like "data", "result", "temp"

### API-Specific Knowledge:
- Semantic Scholar: Rate limit is 100 requests per 5 minutes
- ArXiv API: No official rate limit but be respectful (3 req/s max)
- Use exponential backoff starting at 1s for 429 errors
- Both APIs can have sporadic 503s - retry up to 3 times

## Project Structure

```
ResearchAssistantAgent/
├── src/
│   └── research_assistant/
│       ├── __init__.py
│       ├── collectors/
│       │   ├── __init__.py
│       │   ├── arxiv_collector.py
│       │   └── semantic_scholar_collector.py
│       ├── analyzers/
│       │   ├── __init__.py
│       │   └── paper_analyzer.py
│       ├── vector_store/
│       │   ├── __init__.py
│       │   └── faiss_store.py
│       └── utils/
│           ├── __init__.py
│           └── rate_limiter.py
├── tests/
├── notebooks/
├── requirements.txt
├── requirements-dev.txt
├── setup.py
└── README.md
```

## Virtual Environment Commands

When working on this project, Claude should:
1. Check if virtual environment exists: `test -d venv || python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Ensure pip is updated: `pip install --upgrade pip`
4. Install dependencies: `pip install -r requirements.txt`
