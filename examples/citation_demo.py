#!/usr/bin/env python3
"""
Demo of intelligent claim extraction and citation finding.

Shows how the system identifies different types of claims in research text
and finds appropriate citations from multiple sources.
"""

import asyncio
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from research_assistant.claim_extractor import ClaimExtractor
from research_assistant.citation_finder import CitationFinder
from research_assistant.arxiv_collector import ArxivCollector
from research_assistant.semantic_scholar_collector import SemanticScholarCollector
from research_assistant.config import Config

console = Console()


SAMPLE_TEXT = """
Recent advances in transformer-based models have revolutionized natural language processing.
Our proposed method, TransformerXL, achieves 92.4% accuracy on the GLUE benchmark, which
represents a 15% improvement over the baseline BERT model (p < 0.001). 

We employed a novel attention mechanism that incorporates positional encodings in a 
hierarchical manner. This approach outperforms standard self-attention by reducing
computational complexity from O(n²) to O(n log n), making it more efficient than
existing methods.

The results suggest that the improvement is due to the model's ability to capture
long-range dependencies more effectively. This is consistent with the theory of
emergent complexity in deep neural networks proposed by Bengio et al.

Furthermore, our ablation studies indicate that the hierarchical structure causes
a significant reduction in training time, leading to 3x faster convergence compared
to vanilla transformers. The correlation between model depth and performance was
r=0.87 (p<0.01), demonstrating a strong positive relationship.
"""


async def demo_claim_extraction():
    """Demonstrate claim extraction capabilities."""
    console.print("\n[bold blue]Intelligent Claim Extraction Demo[/bold blue]\n")
    
    # Initialize claim extractor
    extractor = ClaimExtractor()
    
    # Show sample text
    console.print(Panel(
        SAMPLE_TEXT,
        title="Sample Research Text",
        border_style="cyan"
    ))
    
    # Extract claims
    console.print("\n[yellow]Extracting claims...[/yellow]")
    claims = extractor.extract_claims(SAMPLE_TEXT)
    
    # Display claims by type
    claim_types = {}
    for claim in claims:
        if claim.claim_type not in claim_types:
            claim_types[claim.claim_type] = []
        claim_types[claim.claim_type].append(claim)
    
    # Create table for claims
    table = Table(title="Extracted Claims", show_header=True, header_style="bold magenta")
    table.add_column("Type", style="cyan", width=15)
    table.add_column("Claim", style="white", width=50)
    table.add_column("Confidence", justify="right", style="green")
    table.add_column("Keywords", style="yellow")
    
    for claim_type, type_claims in claim_types.items():
        for claim in type_claims:
            table.add_row(
                claim_type.value.title(),
                claim.text[:50] + "..." if len(claim.text) > 50 else claim.text,
                f"{claim.confidence:.2f}",
                ", ".join(claim.keywords[:3])
            )
    
    console.print(table)
    
    # Show analysis
    analysis = extractor.analyze_citation_needs(claims)
    console.print(f"\n[bold]Citation Analysis:[/bold]")
    console.print(f"  Total claims found: {analysis['total_claims']}")
    console.print(f"  High priority citations needed: {analysis['suggested_citations_needed']}")
    console.print(f"  Breakdown by type:")
    for claim_type, count in analysis['by_type'].items():
        if count > 0:
            console.print(f"    - {claim_type}: {count}")
    
    return claims


async def demo_citation_finding(claims):
    """Demonstrate citation finding capabilities."""
    console.print("\n[bold blue]Citation Finding Demo[/bold blue]\n")
    
    # Initialize components
    config = Config()
    
    async with ArxivCollector() as arxiv, \
               SemanticScholarCollector() as s2:
        
        finder = CitationFinder(
            config=config,
            arxiv_collector=arxiv,
            semantic_scholar_collector=s2
        )
        
        # Find citations for high-priority claims
        console.print("[yellow]Searching for citations...[/yellow]\n")
        
        # Get top 3 claims
        top_claims = sorted(claims, key=lambda c: c.confidence, reverse=True)[:3]
        
        for i, claim in enumerate(top_claims, 1):
            console.print(f"[bold]Claim {i}:[/bold] {claim.text}")
            console.print(f"[dim]Type: {claim.claim_type.value}, Confidence: {claim.confidence:.2f}[/dim]")
            
            # Find citations
            with console.status(f"Searching databases for claim {i}..."):
                recommendation = await finder.find_citations_for_claim(
                    claim,
                    max_citations=3
                )
            
            # Display results
            if recommendation.candidates:
                console.print(f"\n[green]Found {len(recommendation.candidates)} citations:[/green]")
                
                for j, candidate in enumerate(recommendation.candidates, 1):
                    console.print(f"\n  [{j}] [bold]{candidate.title}[/bold]")
                    console.print(f"      Authors: {', '.join(candidate.authors[:3])}")
                    console.print(f"      Year: {candidate.year}, Venue: {candidate.venue}")
                    console.print(f"      Relevance: {candidate.relevance_score:.2f}")
                    if candidate.citation_count:
                        console.print(f"      Citations: {candidate.citation_count}")
                    console.print(f"      Source: {candidate.source_database}")
                    
            else:
                console.print("[red]No suitable citations found[/red]")
                
            console.print(f"\n[dim]{recommendation.explanation}[/dim]")
            console.print("-" * 80)
            
            # Small delay to respect rate limits
            await asyncio.sleep(1)


async def main():
    """Run the complete demo."""
    console.print("""
[bold green]Research Assistant - Intelligent Citation System Demo[/bold green]

This demo shows how the system:
1. Extracts different types of claims from research text
2. Identifies which claims need citations
3. Searches multiple databases to find relevant papers
4. Ranks citations by quality and relevance
""")
    
    try:
        # Extract claims
        claims = await demo_claim_extraction()
        
        # Find citations
        if claims:
            await demo_citation_finding(claims)
        else:
            console.print("[red]No claims found to cite![/red]")
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())