#!/usr/bin/env python3
"""
Enhanced command-line interface for Research Assistant Agent.

Features a beautiful, interactive UI with Rich components for
an exceptional user experience.
"""

import asyncio
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import time

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.text import Text
from rich.columns import Columns
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.logging import RichHandler

from .multi_database_search import MultiDatabaseSearch, PaperSource
from .claim_extractor import ClaimExtractor
from .citation_finder import CitationFinder
from .citation_scorer import CitationScorer, FieldOfStudy
from .citation_explainer import CitationExplainer
from .vector_store import FAISSVectorStore
from .config import Config
from .arxiv_collector import ArxivCollector
from .semantic_scholar_collector import SemanticScholarCollector
from .pubmed_collector import PubMedCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)
logger = logging.getLogger(__name__)
console = Console()


class EnhancedResearchAssistant:
    """Enhanced research assistant with beautiful CLI."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.search_history: List[Dict[str, Any]] = []
        self.current_session = {
            "start_time": datetime.now(),
            "searches": 0,
            "papers_found": 0,
            "citations_generated": 0
        }
        
    def _create_header(self) -> Panel:
        """Create a beautiful header panel."""
        header_content = """[bold cyan]Research Assistant[/bold cyan] - Your AI Citation Expert
        
[dim]Find, verify, and score academic citations with confidence[/dim]
        
🔍 Multi-database search  📊 Quality scoring  🎯 Smart recommendations"""
        
        return Panel(
            header_content,
            title="[bold green]Welcome[/bold green]",
            border_style="bright_blue",
            padding=(1, 2)
        )
        
    def _create_search_panel(self, query: str, databases: List[str]) -> Panel:
        """Create search information panel."""
        db_icons = {
            "arxiv": "📚",
            "semantic_scholar": "🎓",
            "pubmed": "🏥",
            "vector_store": "💾"
        }
        
        db_display = " ".join([f"{db_icons.get(db, '📄')} {db.title()}" for db in databases])
        
        content = f"""[bold]Query:[/bold] {query}
[bold]Databases:[/bold] {db_display}
[bold]Time:[/bold] {datetime.now().strftime('%H:%M:%S')}"""
        
        return Panel(content, title="[cyan]Search Parameters[/cyan]", border_style="cyan")
        
    async def search_with_progress(
        self,
        query: str,
        databases: List[str],
        max_results: int = 20
    ) -> Dict[str, Any]:
        """Search with beautiful progress display."""
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True
        ) as progress:
            
            # Create tasks for each database
            tasks = {}
            for db in databases:
                task_name = f"Searching {db.replace('_', ' ').title()}..."
                tasks[db] = progress.add_task(task_name, total=100)
            
            # Initialize search engine
            search_engine = MultiDatabaseSearch(
                config=self.config,
                enable_arxiv="arxiv" in databases,
                enable_semantic_scholar="semantic_scholar" in databases,
                enable_pubmed="pubmed" in databases,
                enable_vector_store="vector_store" in databases
            )
            
            async with search_engine:
                # Simulate progress updates
                for db, task_id in tasks.items():
                    progress.update(task_id, advance=30)
                
                # Perform actual search
                papers = await search_engine.search(
                    query,
                    max_results_per_db=max_results,
                    deduplicate=True
                )
                
                # Complete all tasks
                for task_id in tasks.values():
                    progress.update(task_id, completed=100)
                
                results["papers"] = papers
                results["total"] = len(papers)
                
        return results
        
    def display_search_results(self, papers: List[Any]) -> None:
        """Display search results with rich formatting."""
        if not papers:
            console.print("[yellow]No papers found. Try different search terms.[/yellow]")
            return
            
        # Create results table
        table = Table(
            title=f"[bold]Search Results[/bold] ({len(papers)} papers)",
            show_header=True,
            header_style="bold magenta",
            show_lines=True
        )
        
        table.add_column("#", style="dim", width=3)
        table.add_column("Title", style="cyan", width=50)
        table.add_column("Authors", style="white", width=25)
        table.add_column("Year", justify="center", width=6)
        table.add_column("Sources", style="yellow", width=12)
        table.add_column("Citations", justify="right", style="green", width=8)
        table.add_column("Score", justify="right", style="blue", width=6)
        
        for i, paper in enumerate(papers[:20], 1):
            # Format authors
            authors = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors += f" +{len(paper.authors)-2}"
                
            # Format sources
            sources = []
            for source in paper.source_ids.keys():
                if source == "arxiv":
                    sources.append("[red]arXiv[/red]")
                elif source == "s2":
                    sources.append("[green]S2[/green]")
                elif source == "pubmed":
                    sources.append("[blue]PM[/blue]")
                    
            # Calculate simple score
            score = min(paper.consensus_score * 30 + paper.venue_score * 70, 100)
            
            # Add row with conditional formatting
            title_style = "bold" if paper.consensus_score >= 2 else ""
            table.add_row(
                str(i),
                Text(paper.title[:50] + "..." if len(paper.title) > 50 else paper.title, style=title_style),
                authors,
                str(paper.year),
                " ".join(sources),
                str(paper.citation_count or "-"),
                f"{score:.0f}"
            )
            
        console.print(table)
        
    def display_paper_details(self, paper: Any) -> None:
        """Display detailed information about a paper."""
        # Create detail panels
        info_panel = Panel(
            f"""[bold]Title:[/bold] {paper.title}
[bold]Authors:[/bold] {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}
[bold]Year:[/bold] {paper.year}
[bold]Venue:[/bold] {paper.venue or 'N/A'}
[bold]DOI:[/bold] {paper.doi or 'N/A'}
[bold]Citations:[/bold] {paper.citation_count or 'N/A'}""",
            title="[cyan]Paper Information[/cyan]",
            border_style="cyan"
        )
        
        abstract_panel = Panel(
            Markdown(paper.abstract[:500] + "..." if len(paper.abstract) > 500 else paper.abstract),
            title="[green]Abstract[/green]",
            border_style="green"
        )
        
        # Score the paper
        scorer = CitationScorer()
        paper_dict = {
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "venue": paper.venue,
            "citation_count": paper.citation_count,
            "database_appearances": paper.consensus_score
        }
        score = scorer.score_citation(paper_dict)
        
        score_panel = Panel(
            f"""[bold]Total Score:[/bold] {score.total_score:.1f}/100
[bold]Venue Quality:[/bold] {'⭐' * int(score.venue_score * 5)}
[bold]Citation Impact:[/bold] {'⭐' * int(score.impact_score * 5)}
[bold]Consensus:[/bold] Found in {paper.consensus_score} database(s)

[dim]{score.explanation}[/dim]""",
            title="[yellow]Quality Score[/yellow]",
            border_style="yellow"
        )
        
        # Display all panels
        console.print(info_panel)
        console.print(abstract_panel)
        console.print(score_panel)
        
    async def find_citations_interactive(self, text: str) -> None:
        """Interactively find citations for text."""
        console.print("\n[bold blue]Citation Finder[/bold blue]\n")
        
        # Extract claims
        with console.status("[yellow]Extracting claims from text...[/yellow]"):
            extractor = ClaimExtractor()
            claims = extractor.extract_claims(text)
            
        if not claims:
            console.print("[yellow]No claims requiring citations found.[/yellow]")
            return
            
        # Display claims
        console.print(f"[green]Found {len(claims)} claims:[/green]\n")
        
        claims_table = Table(show_header=True, header_style="bold")
        claims_table.add_column("#", width=3)
        claims_table.add_column("Type", style="cyan", width=15)
        claims_table.add_column("Claim", width=60)
        claims_table.add_column("Confidence", justify="right", style="green")
        
        for i, claim in enumerate(claims, 1):
            claims_table.add_row(
                str(i),
                claim.claim_type.value.title(),
                claim.text[:60] + "..." if len(claim.text) > 60 else claim.text,
                f"{claim.confidence:.2f}"
            )
            
        console.print(claims_table)
        
        # Ask which claims to find citations for
        selected = IntPrompt.ask(
            "\nSelect claim number to find citations (0 for all)",
            default=0,
            choices=[str(i) for i in range(len(claims) + 1)]
        )
        
        if selected == 0:
            selected_claims = claims
        else:
            selected_claims = [claims[selected - 1]]
            
        # Find citations
        console.print("\n[yellow]Searching for citations...[/yellow]\n")
        
        async with ArxivCollector() as arxiv, \
                   SemanticScholarCollector() as s2:
            
            finder = CitationFinder(
                config=self.config,
                arxiv_collector=arxiv,
                semantic_scholar_collector=s2
            )
            
            for claim in selected_claims:
                with console.status(f"Finding citations for: {claim.text[:50]}..."):
                    recommendation = await finder.find_citations_for_claim(claim)
                    
                # Display recommendations
                if recommendation.candidates:
                    panel = Panel(
                        f"[bold]{claim.text}[/bold]\n\n" +
                        "\n".join([
                            f"{i}. [cyan]{c.title}[/cyan] ({c.year}) - Score: {c.relevance_score:.2f}"
                            for i, c in enumerate(recommendation.candidates[:3], 1)
                        ]),
                        title=f"[green]Citations for {claim.claim_type.value} claim[/green]",
                        border_style="green"
                    )
                    console.print(panel)
                    
                    # Ask if user wants detailed explanations
                    if Confirm.ask("\nWould you like detailed explanations for these citations?"):
                        explainer = CitationExplainer(field=FieldOfStudy.GENERAL)
                        scorer = CitationScorer(field=FieldOfStudy.GENERAL)
                        
                        for i, candidate in enumerate(recommendation.candidates[:3], 1):
                            console.print(f"\n[bold]Citation {i}:[/bold]")
                            
                            # Score the citation
                            paper_dict = {
                                'title': candidate.title,
                                'authors': candidate.authors,
                                'year': candidate.year,
                                'venue': candidate.venue,
                                'citation_count': candidate.citation_count,
                                'abstract': candidate.abstract
                            }
                            quality_score = scorer.score_citation(paper_dict)
                            
                            # Get explanation
                            explanation = explainer.explain_citation(
                                claim,
                                paper_dict,
                                candidate.relevance_score,
                                quality_score,
                                recommendation.matched_terms
                            )
                            
                            # Display explanation
                            explainer.display_explanation(explanation)
                            
                            if i < len(recommendation.candidates[:3]):
                                if not Confirm.ask("\nContinue to next citation?", default=True):
                                    break
                else:
                    console.print(f"[red]No citations found for: {claim.text[:50]}...[/red]")
                    
    def show_session_stats(self) -> None:
        """Display session statistics."""
        duration = datetime.now() - self.current_session["start_time"]
        
        stats_tree = Tree("[bold]Session Statistics[/bold]")
        stats_tree.add(f"Duration: {duration}")
        stats_tree.add(f"Searches performed: {self.current_session['searches']}")
        stats_tree.add(f"Papers found: {self.current_session['papers_found']}")
        stats_tree.add(f"Citations generated: {self.current_session['citations_generated']}")
        
        console.print(Panel(stats_tree, border_style="blue"))
        
    def export_results(self, papers: List[Any], format: str = "json") -> Path:
        """Export search results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_results_{timestamp}.{format}"
        
        if format == "json":
            data = []
            for paper in papers:
                data.append({
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "venue": paper.venue,
                    "doi": paper.doi,
                    "abstract": paper.abstract,
                    "citation_count": paper.citation_count,
                    "sources": list(paper.source_ids.keys()),
                    "url": paper.url
                })
                
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
                
        elif format == "bibtex":
            with open(filename, "w") as f:
                for i, paper in enumerate(papers):
                    # Generate BibTeX entry
                    entry_type = "@article" if paper.venue else "@misc"
                    cite_key = f"{paper.authors[0].split(',')[0].lower()}{paper.year}"
                    
                    f.write(f"{entry_type}{{{cite_key},\n")
                    f.write(f"  title = {{{paper.title}}},\n")
                    f.write(f"  author = {{{' and '.join(paper.authors)}}},\n")
                    f.write(f"  year = {{{paper.year}}},\n")
                    if paper.venue:
                        f.write(f"  journal = {{{paper.venue}}},\n")
                    if paper.doi:
                        f.write(f"  doi = {{{paper.doi}}},\n")
                    f.write("}\n\n")
                    
        console.print(f"[green]Results exported to {filename}[/green]")
        return Path(filename)


@click.group()
def cli():
    """Research Assistant - Your AI-powered citation expert."""
    pass


@cli.command()
@click.option('--query', '-q', prompt=True, help='Search query')
@click.option('--databases', '-d', multiple=True, 
              type=click.Choice(['arxiv', 'semantic_scholar', 'pubmed', 'all']),
              default=['all'], help='Databases to search')
@click.option('--limit', '-l', default=20, help='Maximum results per database')
@click.option('--export', '-e', type=click.Choice(['json', 'bibtex']), 
              help='Export format')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def search(query: str, databases: Tuple[str], limit: int, export: Optional[str], 
          interactive: bool):
    """Search for academic papers across multiple databases."""
    assistant = EnhancedResearchAssistant()
    
    # Show header
    console.print(assistant._create_header())
    
    # Process database selection
    if 'all' in databases:
        selected_dbs = ['arxiv', 'semantic_scholar', 'pubmed']
    else:
        selected_dbs = list(databases)
        
    # Show search panel
    console.print(assistant._create_search_panel(query, selected_dbs))
    
    async def run_search():
        # Perform search
        results = await assistant.search_with_progress(query, selected_dbs, limit)
        papers = results.get("papers", [])
        
        # Update session stats
        assistant.current_session["searches"] += 1
        assistant.current_session["papers_found"] += len(papers)
        
        # Display results
        assistant.display_search_results(papers)
        
        if papers and interactive:
            # Interactive mode - allow detailed viewing
            while True:
                choice = IntPrompt.ask(
                    "\nView paper details (enter number, 0 to exit)",
                    default=0,
                    choices=[str(i) for i in range(len(papers) + 1)]
                )
                
                if choice == 0:
                    break
                    
                assistant.display_paper_details(papers[choice - 1])
                
        if papers and export:
            assistant.export_results(papers, export)
            
        # Show session stats
        assistant.show_session_stats()
        
    asyncio.run(run_search())


@cli.command()
@click.option('--text', '-t', help='Text to find citations for')
@click.option('--file', '-f', type=click.Path(exists=True), 
              help='File containing text')
def cite(text: Optional[str], file: Optional[str]):
    """Find citations for claims in your text."""
    assistant = EnhancedResearchAssistant()
    
    # Get text
    if file:
        with open(file, 'r') as f:
            content = f.read()
    elif text:
        content = text
    else:
        console.print("[yellow]Enter your text (press Ctrl+D when done):[/yellow]")
        content = sys.stdin.read()
        
    if not content.strip():
        console.print("[red]No text provided![/red]")
        return
        
    # Show header
    console.print(assistant._create_header())
    
    # Find citations
    asyncio.run(assistant.find_citations_interactive(content))
    
    # Update stats
    assistant.current_session["citations_generated"] += 1
    assistant.show_session_stats()


@cli.command()
@click.option('--field', '-f', 
              type=click.Choice(['cs', 'bio', 'med', 'phys', 'chem', 'math', 'general']),
              default='general', help='Academic field')
def score_demo(field: str):
    """Interactive demo of the citation scoring system."""
    # Import scoring demo from examples
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent / "examples"))
    from scoring_demo import main as scoring_demo_main
    
    # Map field choice to enum
    field_map = {
        'cs': FieldOfStudy.COMPUTER_SCIENCE,
        'bio': FieldOfStudy.BIOLOGY,
        'med': FieldOfStudy.MEDICINE,
        'phys': FieldOfStudy.PHYSICS,
        'chem': FieldOfStudy.CHEMISTRY,
        'math': FieldOfStudy.MATHEMATICS,
        'general': FieldOfStudy.GENERAL
    }
    
    console.print("[bold green]Citation Quality Scoring Demo[/bold green]\n")
    console.print(f"Field: {field_map[field].name.replace('_', ' ').title()}")
    
    # Run the scoring demo
    scoring_demo_main()


@cli.command()
def configure():
    """Configure API keys and settings."""
    from .security import get_secure_config
    
    console.print("[bold]Research Assistant Configuration[/bold]\n")
    
    try:
        secure_config = get_secure_config()
        secure_config.configure_interactive()
        console.print("\n[green]Configuration saved successfully![/green]")
    except Exception as e:
        console.print(f"[red]Configuration failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), 
              required=True, help='Input batch file (CSV or JSON)')
@click.option('--output-format', '-o', type=click.Choice(['json', 'csv']), 
              default='json', help='Output format')
@click.option('--parallel', '-p', default=1, help='Number of parallel jobs')
def batch(input: str, output_format: str, parallel: int):
    """Process multiple queries or documents in batch mode."""
    from .batch_processor import process_batch_file
    
    console.print("[bold]Batch Processing Mode[/bold]\n")
    
    async def run_batch():
        try:
            output_file = await process_batch_file(
                Path(input),
                config=Config(),
                output_format=output_format
            )
            console.print(f"\n[bold green]Batch processing complete![/bold green]")
            console.print(f"Results saved to: {output_file}")
        except Exception as e:
            console.print(f"[red]Batch processing failed: {e}[/red]")
            sys.exit(1)
            
    asyncio.run(run_batch())


if __name__ == "__main__":
    cli()