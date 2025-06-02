# Research Assistant Agent

An AI-powered research assistant for collecting and analyzing academic papers from ArXiv and Semantic Scholar. Built with async Python, FAISS vector search, and LLM integration for intelligent paper analysis.

## Features

- 🔍 **Multi-source paper collection** from ArXiv and Semantic Scholar APIs
- ⚡ **Async/await architecture** for efficient concurrent API calls  
- 🚦 **Intelligent rate limiting** with adaptive backoff strategies
- 🧠 **LLM-powered analysis** for extracting insights from papers
- 📊 **Vector similarity search** using FAISS for finding related papers
- 🖥️ **Rich CLI interface** with colorful tables and progress tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/davidburton/ResearchAssistantAgent.git
cd ResearchAssistantAgent

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

## Quick Start

### Command Line Interface

Search for papers across both ArXiv and Semantic Scholar:

```bash
# Basic search
research-assistant search "transformer neural networks"

# Search only ArXiv
research-assistant search "quantum computing" --source arxiv --limit 20

# Search by author
research-assistant advanced-search --author "Yoshua Bengio" --limit 10

# Search by category (ArXiv)
research-assistant advanced-search --category cs.AI --limit 15

# Store results in vector database (requires OpenAI API key for embeddings)
research-assistant search "large language models" --store
```

### Python API

```python
import asyncio
from research_assistant import ArxivCollector, SemanticScholarCollector

async def search_papers():
    # Search ArXiv
    async with ArxivCollector() as arxiv:
        papers = await arxiv.search("cat:cs.LG transformer", max_results=5)
        for paper in papers:
            print(f"{paper.title} - {paper.arxiv_id}")
    
    # Search Semantic Scholar  
    async with SemanticScholarCollector() as s2:
        papers = await s2.search("deep learning", limit=5)
        for paper in papers:
            print(f"{paper.title} - Citations: {paper.citation_count}")

asyncio.run(search_papers())
```

## Architecture

The project follows a modular architecture:

```
src/research_assistant/
├── collectors/          # API clients for paper sources
│   ├── arxiv_collector.py
│   └── semantic_scholar_collector.py
├── analyzers/          # LLM-based paper analysis
│   └── paper_analyzer.py
├── vector_store/       # FAISS similarity search
│   └── faiss_store.py
└── utils/             # Rate limiting and helpers
    └── rate_limiter.py
```

## Configuration

Set environment variables for API keys:

```bash
export OPENAI_API_KEY="your-api-key"  # For paper analysis and embeddings
export SEMANTIC_SCHOLAR_API_KEY="your-key"  # Optional, for higher rate limits
```

## Development

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/
```

## API Rate Limits

The tool respects API rate limits:
- **ArXiv**: Max 3 requests/second (configurable)
- **Semantic Scholar**: 100 requests per 5 minutes (anonymous)

## Advanced Usage

### Using the Rate Limiter

```python
from research_assistant import RateLimiter, AdaptiveRateLimiter

# Fixed rate limiting
limiter = RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute

# Adaptive rate limiting (adjusts based on server responses)
adaptive = AdaptiveRateLimiter(
    initial_rate=10.0,
    min_rate=1.0,
    max_rate=50.0,
    backoff_factor=0.5
)

# Use with async context manager
async with limiter:
    # Your API call here
    pass
```

### Paper Analysis with LLMs

```python
from research_assistant import PaperAnalyzer, AnalysisType

analyzer = PaperAnalyzer(api_key="your-openai-key")

# Analyze a paper
analysis = await analyzer.analyze_paper(
    paper_text="Paper abstract or full text...",
    paper_id="arxiv.2301.00001",
    paper_title="Attention Is All You Need",
    analysis_type=AnalysisType.METHODOLOGY
)

print(analysis.methodology)
print(analysis.key_contributions)
```

### Vector Store Operations

```python
from research_assistant import FAISSVectorStore, Document

# Initialize vector store
store = FAISSVectorStore(dimension=1536, index_type="flat")

# Add documents
doc = Document(
    id="paper_001",
    text="Paper content...",
    metadata={"title": "Paper Title", "authors": ["Author 1"]},
    embedding=[0.1, 0.2, ...]  # 1536-dimensional vector
)
store.add_documents([doc])

# Search similar documents
results = store.search(query_embedding, k=10)

# Save and load
store.save("./my_index")
loaded_store = FAISSVectorStore.load("./my_index")
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=research_assistant tests/

# Run specific test file
pytest tests/unit/utils/test_rate_limiter.py
```

## Project Status

This is an actively developed research tool. Current focus areas:
- ✅ Core API collectors (ArXiv, Semantic Scholar)
- ✅ Rate limiting and async architecture
- ✅ FAISS vector store integration
- ✅ CLI interface
- 🚧 Full paper content extraction
- 🚧 Advanced LLM analysis pipelines
- 📋 Web UI dashboard
- 📋 Citation graph analysis

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with:
- [aiohttp](https://docs.aiohttp.org/) for async HTTP
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Click](https://click.palletsprojects.com/) for CLI
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output