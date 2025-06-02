#!/usr/bin/env python3
"""
Demo of multi-database search and verification.

Shows how the system searches across ArXiv, Semantic Scholar, and PubMed
simultaneously to find and verify papers with consensus scoring.
"""

import asyncio
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from research_assistant.multi_database_search import MultiDatabaseSearch, PaperSource
from research_assistant.config import Config

console = Console()


async def search_demo():
    """Demonstrate multi-database search capabilities."""
    console.print("\n[bold blue]Multi-Database Search Demo[/bold blue]\n")
    
    # Get search query from user or use default
    query = "transformer neural networks attention mechanism"
    console.print(f"[cyan]Search Query:[/cyan] {query}\n")
    
    # Initialize multi-database search
    config = Config()
    
    # Check for PubMed credentials
    pubmed_api_key = os.getenv("NCBI_API_KEY")
    pubmed_email = os.getenv("PUBMED_EMAIL", "demo@example.com")
    
    search_engine = MultiDatabaseSearch(
        config=config,
        enable_arxiv=True,
        enable_semantic_scholar=True,
        enable_pubmed=True,
        enable_vector_store=False,  # Disable for demo
        pubmed_api_key=pubmed_api_key,
        pubmed_email=pubmed_email
    )
    
    async with search_engine:
        # Show which databases are enabled
        console.print("[yellow]Enabled databases:[/yellow]")
        if search_engine.arxiv:
            console.print("  ✓ ArXiv")
        if search_engine.semantic_scholar:
            console.print("  ✓ Semantic Scholar")  
        if search_engine.pubmed:
            console.print("  ✓ PubMed")
        console.print()
        
        # Perform search with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Searching databases...", total=None)
            
            results = await search_engine.search(
                query,
                max_results_per_db=10,
                deduplicate=True,
                require_consensus=1
            )
            
            progress.update(task, completed=True)
        
        # Display results
        console.print(f"\n[green]Found {len(results)} unique papers[/green]\n")
        
        # Create results table
        table = Table(title="Search Results", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", width=50)
        table.add_column("Authors", style="white", width=30)
        table.add_column("Year", justify="center", width=6)
        table.add_column("Sources", style="yellow", width=15)
        table.add_column("Citations", justify="right", style="green", width=10)
        table.add_column("Score", justify="right", style="blue", width=6)
        
        for i, paper in enumerate(results[:15], 1):  # Show top 15
            # Format sources
            sources = []
            if "arxiv" in paper.source_ids:
                sources.append("arXiv")
            if "s2" in paper.source_ids:
                sources.append("S2")
            if "pubmed" in paper.source_ids:
                sources.append("PM")
                
            # Format authors
            authors = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors += f" +{len(paper.authors)-2}"
                
            # Calculate simple score for display
            score = paper.consensus_score * 0.5 + min(paper.venue_score, 1.0) * 0.5
            
            table.add_row(
                str(i),
                paper.title[:50] + "..." if len(paper.title) > 50 else paper.title,
                authors,
                str(paper.year),
                ", ".join(sources),
                str(paper.citation_count or "-"),
                f"{score:.2f}"
            )
            
        console.print(table)
        
        # Show consensus analysis
        console.print("\n[bold]Consensus Analysis:[/bold]")
        consensus_counts = {}
        for paper in results:
            count = paper.consensus_score
            consensus_counts[count] = consensus_counts.get(count, 0) + 1
            
        for count, num_papers in sorted(consensus_counts.items(), reverse=True):
            console.print(f"  Papers in {count} database(s): {num_papers}")
            
        # Show papers with high consensus
        high_consensus = [p for p in results if p.consensus_score >= 2]
        if high_consensus:
            console.print(f"\n[green]Found {len(high_consensus)} papers in multiple databases:[/green]")
            
            for paper in high_consensus[:5]:
                console.print(f"\n[bold]{paper.title}[/bold]")
                console.print(f"  Sources: {', '.join(paper.source_ids.keys())}")
                if paper.doi:
                    console.print(f"  DOI: {paper.doi}")
                console.print(f"  Citations: {paper.citation_count or 'N/A'}")
                
        return results


async def verification_demo(papers):
    """Demonstrate paper verification across databases."""
    console.print("\n[bold blue]Cross-Database Verification Demo[/bold blue]\n")
    
    if not papers:
        console.print("[red]No papers to verify![/red]")
        return
        
    # Select a paper with high consensus
    high_consensus_papers = [p for p in papers if p.consensus_score >= 2]
    
    if high_consensus_papers:
        paper = high_consensus_papers[0]
    else:
        paper = papers[0]
        
    console.print(f"[cyan]Verifying paper:[/cyan] {paper.title}\n")
    
    # Show verification results
    console.print("[yellow]Database Coverage:[/yellow]")
    
    verification_table = Table(show_header=True, header_style="bold")
    verification_table.add_column("Database", style="cyan")
    verification_table.add_column("ID", style="white")
    verification_table.add_column("Status", style="green")
    
    for source, source_id in paper.source_ids.items():
        status = "✓ Found"
        verification_table.add_row(source.upper(), source_id[:20] + "...", status)
        
    # Add missing databases
    all_sources = ["arxiv", "s2", "pubmed", "doi"]
    for source in all_sources:
        if source not in paper.source_ids:
            verification_table.add_row(source.upper(), "-", "[red]Not found[/red]")
            
    console.print(verification_table)
    
    # Show paper details
    console.print("\n[yellow]Paper Details:[/yellow]")
    details = Panel(
        f"[bold]Title:[/bold] {paper.title}\n"
        f"[bold]Authors:[/bold] {', '.join(paper.authors[:5])}\n"
        f"[bold]Year:[/bold] {paper.year}\n"
        f"[bold]Venue:[/bold] {paper.venue or 'N/A'}\n"
        f"[bold]Citations:[/bold] {paper.citation_count or 'N/A'}\n"
        f"[bold]DOI:[/bold] {paper.doi or 'N/A'}\n"
        f"[bold]Abstract:[/bold] {paper.abstract[:200]}...",
        title="Verified Paper Information",
        border_style="green"
    )
    console.print(details)
    
    # Show consensus strength
    console.print(f"\n[bold]Verification Strength:[/bold]")
    if paper.consensus_score >= 3:
        console.print("  [green]★★★ High confidence - Found in 3+ databases[/green]")
    elif paper.consensus_score == 2:
        console.print("  [yellow]★★☆ Medium confidence - Found in 2 databases[/yellow]")
    else:
        console.print("  [dim]★☆☆ Low confidence - Found in 1 database[/dim]")
        
    if paper.doi:
        console.print("  [green]+ Has DOI (persistent identifier)[/green]")
    if paper.citation_count and paper.citation_count > 10:
        console.print(f"  [green]+ Well-cited ({paper.citation_count} citations)[/green]")


async def main():
    """Run the complete demo."""
    console.print("""
[bold green]Multi-Database Search & Verification Demo[/bold green]

This demo shows how the system:
1. Searches multiple databases simultaneously (ArXiv, Semantic Scholar, PubMed)
2. Deduplicates results based on DOI and title matching
3. Calculates consensus scores when papers appear in multiple databases
4. Verifies paper information across sources
5. Ranks results by quality metrics

Note: For full PubMed access, set NCBI_API_KEY and PUBMED_EMAIL environment variables.
""")
    
    try:
        # Run search demo
        papers = await search_demo()
        
        # Run verification demo
        if papers:
            await verification_demo(papers)
            
        # Show example of finding contradicting evidence
        console.print("\n[bold blue]Finding Contradicting Evidence[/bold blue]\n")
        console.print("The system can also search for papers that disagree with claims...")
        console.print("For example, searching for 'transformer limitations problems' would find:")
        console.print("  - Papers discussing computational complexity issues")
        console.print("  - Studies showing where transformers fail")
        console.print("  - Analyses of transformer weaknesses")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())