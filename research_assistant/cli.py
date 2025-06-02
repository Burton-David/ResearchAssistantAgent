#!/usr/bin/env python3
"""
Command-line interface for Research Assistant Agent.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any

import click
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

from .arxiv_collector import ArxivCollector, ArxivPaper
from .semantic_scholar_collector import SemanticScholarCollector, SemanticScholarPaper
from .vector_store import FAISSVectorStore, Document
from .rate_limiter import RateLimiter
from .config import get_config
from .validators import Validators, ValidationError
from .exceptions import ErrorHandler, ResearchAssistantError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()


class ResearchAssistant:
    """Main class coordinating paper collection and storage."""
    
    def __init__(self, vector_store_path: Optional[Path] = None):
        self.vector_store_path = vector_store_path or Path.home() / ".research_assistant" / "vector_store"
        self.vector_store = None
        self.arxiv_collector = ArxivCollector()
        self.semantic_scholar_collector = SemanticScholarCollector()
        
    def _init_vector_store(self):
        """Initialize or load vector store."""
        if self.vector_store_path.exists() and (self.vector_store_path / "index.faiss").exists():
            logger.info(f"Loading existing vector store from {self.vector_store_path}")
            self.vector_store = FAISSVectorStore.load(self.vector_store_path)
        else:
            logger.info("Creating new vector store")
            config = get_config()
            self.vector_store = FAISSVectorStore(config=config)
            
    async def search_arxiv(self, query: str, max_results: int = 10) -> List[ArxivPaper]:
        """Search ArXiv for papers."""
        async with self.arxiv_collector as collector:
            papers = await collector.search(query, max_results=max_results)
            return papers
            
    async def search_semantic_scholar(self, query: str, limit: int = 10) -> List[SemanticScholarPaper]:
        """Search Semantic Scholar for papers."""
        async with self.semantic_scholar_collector as collector:
            papers = await collector.search(query, limit=limit)
            return papers
            
    def display_arxiv_results(self, papers: List[ArxivPaper]):
        """Display ArXiv search results in a table."""
        if not papers:
            console.print("[yellow]No papers found.[/yellow]")
            return
            
        table = Table(title=f"ArXiv Results ({len(papers)} papers)")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="white", max_width=50)
        table.add_column("Authors", style="green", max_width=30)
        table.add_column("Date", style="blue")
        table.add_column("Categories", style="magenta")
        
        for paper in papers:
            authors = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors += f" et al. ({len(paper.authors)} total)"
                
            table.add_row(
                paper.arxiv_id,
                paper.title[:100] + "..." if len(paper.title) > 100 else paper.title,
                authors,
                paper.published_date.strftime("%Y-%m-%d"),
                ", ".join(paper.categories[:3])
            )
            
        console.print(table)
        
    def display_semantic_scholar_results(self, papers: List[SemanticScholarPaper]):
        """Display Semantic Scholar search results in a table."""
        if not papers:
            console.print("[yellow]No papers found.[/yellow]")
            return
            
        table = Table(title=f"Semantic Scholar Results ({len(papers)} papers)")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="white", max_width=50)
        table.add_column("Year", style="blue")
        table.add_column("Citations", style="green")
        table.add_column("Venue", style="magenta", max_width=30)
        
        for paper in papers:
            paper_id = paper.paper_id[:12] + "..." if len(paper.paper_id) > 12 else paper.paper_id
            
            table.add_row(
                paper_id,
                paper.title[:100] + "..." if len(paper.title) > 100 else paper.title,
                str(paper.year) if paper.year else "N/A",
                str(paper.citation_count),
                paper.venue[:30] if paper.venue else "N/A"
            )
            
        console.print(table)
        
    async def store_papers(self, papers: List[Any], source: str):
        """Store papers in vector database with embeddings."""
        if not self.vector_store:
            self._init_vector_store()
            
        stored_count = 0
        console.print(f"\n[bold green]Storing {len(papers)} papers with embeddings...[/bold green]")
        
        for paper in papers:
            try:
                if isinstance(paper, ArxivPaper):
                    paper_id = f"arxiv_{paper.arxiv_id}"
                    paper_text = f"{paper.title}\n\n{paper.abstract}"
                    metadata = {
                        "source": "arxiv",
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "published_date": paper.published_date.isoformat(),
                        "categories": paper.categories,
                        "pdf_url": paper.pdf_url
                    }
                elif isinstance(paper, SemanticScholarPaper):
                    paper_id = f"s2_{paper.paper_id}"
                    paper_text = f"{paper.title}\n\n{paper.abstract or 'No abstract available'}"
                    metadata = {
                        "source": "semantic_scholar",
                        "paper_id": paper.paper_id,
                        "title": paper.title,
                        "year": paper.year,
                        "citation_count": paper.citation_count,
                        "venue": paper.venue,
                        "fields_of_study": paper.fields_of_study
                    }
                else:
                    continue
                    
                # Add paper with embedding generation
                self.vector_store.add_paper(
                    paper_id=paper_id,
                    paper_text=paper_text,
                    paper_metadata=metadata,
                    chunk_paper=False,  # Don't chunk abstracts
                    generate_embedding=True,
                    show_progress=len(papers) == 1  # Show progress for single papers
                )
                stored_count += 1
                
            except Exception as e:
                logger.error(f"Failed to store paper {paper_id}: {e}", exc_info=True)
                
        if stored_count > 0:
            console.print(f"[green]Successfully stored {stored_count} papers with embeddings[/green]")
            
            # Save vector store
            self.vector_store.save(self.vector_store_path)
            console.print(f"[dim]Vector store saved to {self.vector_store_path}[/dim]")
        else:
            console.print("[yellow]No papers were stored[/yellow]")


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug):
    """Research Assistant - AI-powered paper collection and analysis."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument('query')
@click.option('--source', type=click.Choice(['arxiv', 'semantic-scholar', 'both']), default='both')
@click.option('--limit', default=10, help='Maximum number of results')
@click.option('--store', is_flag=True, help='Store results in vector database')
@click.option('--store-path', type=click.Path(), help='Path to vector store')
def search(query: str, source: str, limit: int, store: bool, store_path: Optional[str]):
    """Search for research papers."""
    assistant = ResearchAssistant(vector_store_path=Path(store_path) if store_path else None)
    
    async def run_search():
        results = {}
        
        if source in ['arxiv', 'both']:
            console.print(f"\n[bold blue]Searching ArXiv for: {query}[/bold blue]")
            try:
                arxiv_papers = await assistant.search_arxiv(query, max_results=limit)
                results['arxiv'] = arxiv_papers
                assistant.display_arxiv_results(arxiv_papers)
            except Exception as e:
                logger.error(f"ArXiv search failed: {e}")
                
        if source in ['semantic-scholar', 'both']:
            console.print(f"\n[bold blue]Searching Semantic Scholar for: {query}[/bold blue]")
            try:
                s2_papers = await assistant.search_semantic_scholar(query, limit=limit)
                results['semantic_scholar'] = s2_papers
                assistant.display_semantic_scholar_results(s2_papers)
            except Exception as e:
                logger.error(f"Semantic Scholar search failed: {e}")
                
        if store and results:
            console.print("\n[bold green]Storing results in vector database...[/bold green]")
            for source_name, papers in results.items():
                if papers:
                    await assistant.store_papers(papers, source_name)
                    
    asyncio.run(run_search())


@cli.command()
def configure():
    """Configure API keys securely."""
    from .security import get_secure_config
    
    try:
        secure_config = get_secure_config()
        secure_config.configure_interactive()
    except Exception as e:
        console.print(f"[red]Configuration failed: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--store-path', type=click.Path(), help='Path to vector store')
def stats(store_path: Optional[str]):
    """Show vector store statistics."""
    assistant = ResearchAssistant(vector_store_path=Path(store_path) if store_path else None)
    
    if assistant.vector_store_path.exists() and (assistant.vector_store_path / "index.faiss").exists():
        assistant._init_vector_store()
        stats = assistant.vector_store.get_statistics()
        
        table = Table(title="Vector Store Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))
            
        console.print(table)
    else:
        console.print("[yellow]No vector store found. Run a search with --store flag first.[/yellow]")


@cli.command()
@click.argument('query')
@click.option('--limit', default=10, help='Number of similar papers to return')
@click.option('--source', type=click.Choice(['all', 'arxiv', 'semantic-scholar']), default='all')
@click.option('--store-path', type=click.Path(), help='Path to vector store')
def similarity_search(query: str, limit: int, source: str, store_path: Optional[str]):
    """Find papers similar to your query using vector similarity search."""
    assistant = ResearchAssistant(vector_store_path=Path(store_path) if store_path else None)
    
    if not (assistant.vector_store_path.exists() and (assistant.vector_store_path / "index.faiss").exists()):
        console.print("[red]No vector store found. Run searches with --store flag first.[/red]")
        return
        
    assistant._init_vector_store()
    
    console.print(f"\n[bold blue]Searching for papers similar to: {query}[/bold blue]")
    
    try:
        # Prepare filter
        filter_metadata = None
        if source != 'all':
            filter_metadata = {'source': source.replace('-', '_')}
            
        # Search
        results = assistant.vector_store.search(
            query=query,
            k=limit,
            filter_metadata=filter_metadata
        )
        
        if not results:
            console.print("[yellow]No similar papers found.[/yellow]")
            return
            
        # Display results
        table = Table(title=f"Similar Papers (Top {len(results)})")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Title", style="white", max_width=60)
        table.add_column("Source", style="green")
        table.add_column("Score", style="yellow")
        table.add_column("Year/Date", style="blue")
        
        for i, (doc, score) in enumerate(results, 1):
            # Extract title
            title = doc.metadata.get('title', 'Unknown')
            if len(title) > 60:
                title = title[:57] + "..."
                
            # Extract source
            source_name = doc.metadata.get('source', 'unknown')
            
            # Extract date/year
            if 'year' in doc.metadata:
                date_str = str(doc.metadata['year'])
            elif 'published_date' in doc.metadata:
                date_str = doc.metadata['published_date'][:10]
            else:
                date_str = "N/A"
                
            table.add_row(
                str(i),
                title,
                source_name,
                f"{score:.3f}",
                date_str
            )
            
        console.print(table)
        
        # Show details of top result
        if results:
            top_doc, top_score = results[0]
            console.print("\n[bold]Top Match Details:[/bold]")
            console.print(f"Title: {top_doc.metadata.get('title', 'Unknown')}")
            
            if 'authors' in top_doc.metadata:
                authors = top_doc.metadata['authors']
                if isinstance(authors, list):
                    console.print(f"Authors: {', '.join(authors[:3])}")
                    
            if 'categories' in top_doc.metadata:
                console.print(f"Categories: {', '.join(top_doc.metadata['categories'][:3])}")
                
            if 'citation_count' in top_doc.metadata:
                console.print(f"Citations: {top_doc.metadata['citation_count']}")
                
            # Show text preview
            text_preview = top_doc.text[:300]
            if len(top_doc.text) > 300:
                text_preview += "..."
            console.print(f"\n[dim]Preview: {text_preview}[/dim]")
            
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--author', help='Search by author name')
@click.option('--category', help='Search by ArXiv category (e.g., cs.AI)')
@click.option('--limit', default=10, help='Maximum number of results')
def advanced_search(author: Optional[str], category: Optional[str], limit: int):
    """Advanced search with specific filters."""
    assistant = ResearchAssistant()
    
    async def run_advanced_search():
        if author:
            console.print(f"\n[bold blue]Searching for papers by: {author}[/bold blue]")
            try:
                papers = await assistant.arxiv_collector.search_by_author(author, max_results=limit)
                assistant.display_arxiv_results(papers)
            except Exception as e:
                logger.error(f"Author search failed: {e}")
                
        if category:
            console.print(f"\n[bold blue]Searching for papers in category: {category}[/bold blue]")
            try:
                papers = await assistant.arxiv_collector.search_by_category(category, max_results=limit)
                assistant.display_arxiv_results(papers)
            except Exception as e:
                logger.error(f"Category search failed: {e}")
                
    if author or category:
        asyncio.run(run_advanced_search())
    else:
        console.print("[yellow]Please specify --author or --category[/yellow]")


def main():
    """Main entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted by user[/red]")
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()