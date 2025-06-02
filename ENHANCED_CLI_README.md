# Research Assistant Pro - Enhanced CLI

A beautiful, powerful command-line interface for academic research with intelligent citation finding, quality scoring, and batch processing capabilities.

## Features

### 🎨 Beautiful Terminal UI
- Rich formatting with colors, panels, and progress bars
- Interactive paper browsing with detailed views
- Session statistics tracking
- Clear, organized output

### 🔍 Multi-Database Search
- Simultaneous search across ArXiv, Semantic Scholar, and PubMed
- Real-time progress tracking
- Automatic deduplication
- Consensus scoring across databases

### 🎯 Intelligent Citation Finding
- Automatic claim extraction from text
- NLP-powered relevance matching
- Citation quality scoring
- Detailed explanations for recommendations

### 📊 Advanced Quality Scoring
- Field-aware scoring (CS, Biology, Medicine, etc.)
- Venue reputation analysis
- Citation impact metrics
- Self-citation detection
- Suspicious pattern identification

### 🚀 Batch Processing
- Process multiple queries or documents
- CSV and JSON input/output
- Comprehensive reports
- Efficient parallel processing

### 📝 Export Capabilities
- JSON format for programmatic use
- BibTeX format for LaTeX integration
- Formatted reports

## Installation

```bash
# Clone the repository
git clone https://github.com/yourname/ResearchAssistantAgent.git
cd ResearchAssistantAgent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Download spaCy model (required for claim extraction)
python -m spacy download en_core_web_sm
```

## Quick Start

### Basic Search
```bash
# Search across all databases
research-assistant-pro search -q "transformer neural networks"

# Search specific databases
research-assistant-pro search -q "CRISPR gene editing" -d pubmed -d semantic_scholar

# Interactive mode with paper details
research-assistant-pro search -q "quantum computing" -i

# Export results
research-assistant-pro search -q "climate change" -e json
```

### Citation Finding
```bash
# Find citations for text
research-assistant-pro cite -t "Transformers have revolutionized NLP by enabling parallel processing of sequences."

# Find citations for a file
research-assistant-pro cite -f manuscript.txt

# Interactive mode (type text, press Ctrl+D when done)
research-assistant-pro cite
```

### Batch Processing
```bash
# Process multiple queries from JSON
research-assistant-pro batch -i examples/batch_search.json

# Process from CSV with CSV output
research-assistant-pro batch -i queries.csv -o csv

# View example batch files
cat examples/batch_search.json
cat examples/batch_search.csv
```

### Citation Scoring Demo
```bash
# Run interactive scoring demo
research-assistant-pro score-demo -f cs  # Computer Science
research-assistant-pro score-demo -f med  # Medicine
research-assistant-pro score-demo -f math  # Mathematics
```

### Configuration
```bash
# Set up API keys securely
research-assistant-pro configure
```

## Advanced Usage

### Python API
```python
from research_assistant.cli_enhanced import EnhancedResearchAssistant
import asyncio

async def main():
    assistant = EnhancedResearchAssistant()
    
    # Search for papers
    results = await assistant.search_with_progress(
        "deep learning attention mechanisms",
        ["arxiv", "semantic_scholar"],
        max_results=10
    )
    
    # Display results
    assistant.display_search_results(results["papers"])
    
    # Find citations for text
    await assistant.find_citations_interactive(
        "Recent advances in transformers have improved NLP tasks."
    )

asyncio.run(main())
```

### Batch File Format

**JSON Format:**
```json
[
  {
    "type": "search",
    "data": {
      "query": "machine learning fairness",
      "databases": ["arxiv", "semantic_scholar"],
      "limit": 5
    }
  },
  {
    "type": "cite",
    "data": {
      "text": "Neural networks can exhibit biased behavior when trained on imbalanced datasets."
    }
  }
]
```

**CSV Format:**
```csv
type,query,databases,limit,text
search,"quantum error correction","arxiv",10,
cite,,,,"Quantum computers require error correction to achieve fault tolerance."
```

## Features in Detail

### Citation Explanations
When finding citations, the system provides detailed explanations including:
- Why the citation is relevant
- Quality assessment breakdown
- Confidence scores
- Potential warnings or issues
- Recommendations (strong/moderate/weak)

### Field-Aware Scoring
The scoring system adapts to different academic fields:
- **Computer Science**: Values recent papers, conference venues
- **Mathematics**: Values timeless results, theoretical rigor
- **Medicine**: Values clinical trials, high-impact journals
- **Physics**: Accommodates large collaborations

### Smart Deduplication
Papers found across multiple databases are:
- Automatically deduplicated by DOI/title
- Given higher consensus scores
- Merged with combined metadata

## Tips and Tricks

1. **Use interactive mode** (`-i`) to explore papers in detail
2. **Export to BibTeX** for direct use in LaTeX documents
3. **Batch process** literature reviews to save time
4. **Check explanations** to understand citation recommendations
5. **Configure field** in scoring demo to see field-specific differences

## Troubleshooting

### No results found
- Try broader search terms
- Check your internet connection
- Ensure API keys are configured (if needed)

### Slow performance
- Reduce the number of results (`-l 5`)
- Search fewer databases at once
- Check rate limits in configuration

### Installation issues
- Ensure Python 3.11+ is installed
- Try upgrading pip: `pip install --upgrade pip`
- Install in a clean virtual environment

## Examples

See the `examples/` directory for:
- `cli_demo.py` - Interactive demo of all features
- `batch_search.json` - Sample batch processing file
- `batch_search.csv` - CSV format example
- `scoring_demo.py` - Citation quality scoring demonstration

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Open an issue on GitHub
- Check the documentation
- Run `research-assistant-pro --help` for command options

---

Made with ❤️ for researchers who deserve better tools