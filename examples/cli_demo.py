#!/usr/bin/env python3
"""
Demo script showing enhanced CLI capabilities.

This showcases the beautiful interface and powerful features
of the Research Assistant Pro CLI.
"""

import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()


def demo_search():
    """Demo the search functionality."""
    console.print("\n[bold blue]Demo: Multi-Database Search[/bold blue]\n")
    
    query = "transformer neural networks attention mechanism"
    console.print(f"Searching for: [cyan]{query}[/cyan]\n")
    
    # Run the search command
    cmd = [
        sys.executable, "-m", "research_assistant",
        "search", "-q", query,
        "-d", "all",
        "-l", "10",
        "--interactive"
    ]
    
    subprocess.run(cmd)


def demo_citation_finding():
    """Demo the citation finding functionality."""
    console.print("\n[bold blue]Demo: Citation Finding[/bold blue]\n")
    
    sample_text = """
    Recent advances in transformer architectures have revolutionized natural language processing.
    The self-attention mechanism allows models to capture long-range dependencies more effectively
    than traditional RNNs. Studies show that transformer models achieve state-of-the-art results
    on many NLP benchmarks. The computational efficiency of transformers scales better with
    sequence length compared to recurrent architectures.
    """
    
    console.print("Sample text with claims:")
    console.print(Panel(sample_text, border_style="dim"))
    
    # Run the cite command
    cmd = [
        sys.executable, "-m", "research_assistant",
        "cite", "-t", sample_text
    ]
    
    subprocess.run(cmd)


def demo_scoring():
    """Demo the citation scoring system."""
    console.print("\n[bold blue]Demo: Citation Quality Scoring[/bold blue]\n")
    
    # Run the score demo
    cmd = [
        sys.executable, "-m", "research_assistant",
        "score-demo", "-f", "cs"
    ]
    
    subprocess.run(cmd)


def main():
    """Run the enhanced CLI demo."""
    console.print("""
[bold green]Research Assistant Pro - CLI Demo[/bold green]

This demo showcases the enhanced features of Research Assistant Pro:
- Beautiful Rich terminal UI
- Multi-database search with progress tracking
- Interactive paper browsing
- Intelligent claim extraction and citation finding
- Citation quality scoring with field awareness
- Export capabilities (JSON/BibTeX)

Let's explore these features!
""")
    
    if Confirm.ask("\nRun search demo?", default=True):
        demo_search()
        
    if Confirm.ask("\nRun citation finding demo?", default=True):
        demo_citation_finding()
        
    if Confirm.ask("\nRun scoring system demo?", default=True):
        demo_scoring()
        
    console.print("""
[bold green]Demo Complete![/bold green]

To use Research Assistant Pro in your own projects:

1. Install the package:
   pip install -e .

2. Run the enhanced CLI:
   research-assistant-pro search -q "your query"
   research-assistant-pro cite -f your_paper.txt
   research-assistant-pro score-demo

3. Or use as a module:
   python -m research_assistant search -q "your query"

Happy researching! 🚀
""")


if __name__ == "__main__":
    main()