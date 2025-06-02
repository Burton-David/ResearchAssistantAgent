#!/usr/bin/env python3
"""
Demo of citation quality scoring system.

Shows how citations are scored based on venue quality, impact,
author credibility, and other factors.
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn
from rich.columns import Columns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from research_assistant.citation_scorer import (
    CitationScorer, FieldOfStudy, VenueMetrics, AuthorMetrics
)

console = Console()


# Sample papers for demonstration
SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": ["Vaswani, A.", "Shazeer, N.", "Parmar, N.", "Uszkoreit, J.", 
                   "Jones, L.", "Gomez, A.N.", "Kaiser, Ł.", "Polosukhin, I."],
        "venue": "NeurIPS 2017",
        "year": 2017,
        "citation_count": 50000,
        "database_appearances": 3,
        "abstract": "The dominant sequence transduction models...",
        "venue_metrics": VenueMetrics(
            name="NeurIPS",
            impact_factor=10.5,
            tier="A*",
            field=FieldOfStudy.COMPUTER_SCIENCE
        ),
        "author_metrics": [
            AuthorMetrics(name="Vaswani, A.", h_index=45),
            AuthorMetrics(name="Shazeer, N.", h_index=60)
        ]
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors": ["Devlin, J.", "Chang, M.W.", "Lee, K.", "Toutanova, K."],
        "venue": "NAACL 2019",
        "year": 2019,
        "citation_count": 35000,
        "database_appearances": 3,
        "venue_metrics": VenueMetrics(
            name="NAACL",
            impact_factor=8.2,
            tier="A",
            field=FieldOfStudy.COMPUTER_SCIENCE
        )
    },
    {
        "title": "A Recent ArXiv Preprint on Transformers",
        "authors": ["Unknown, A.", "Student, G."],
        "venue": "arXiv cs.LG",
        "year": 2024,
        "citation_count": 5,
        "database_appearances": 1
    },
    {
        "title": "Questionable Results in Deep Learning",
        "authors": ["Suspicious, A."] * 15,  # Many authors
        "venue": "Unknown Workshop",
        "year": 2023,
        "citation_count": 500,  # Suspiciously high for new paper
        "database_appearances": 1,
        "is_retracted": False
    },
    {
        "title": "Classical Statistical Learning Theory",
        "authors": ["Vapnik, V."],
        "venue": "Annals of Statistics",
        "year": 1995,
        "citation_count": 15000,
        "database_appearances": 2,
        "venue_metrics": VenueMetrics(
            name="Annals of Statistics",
            impact_factor=4.5,
            field=FieldOfStudy.MATHEMATICS
        )
    }
]


def demo_basic_scoring():
    """Demonstrate basic citation scoring."""
    console.print("\n[bold blue]Citation Quality Scoring Demo[/bold blue]\n")
    
    # Create scorer for CS field
    scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
    
    # Score all sample papers
    console.print("[yellow]Scoring sample papers...[/yellow]\n")
    
    scored_papers = []
    for paper in SAMPLE_PAPERS:
        score = scorer.score_citation(paper, current_year=2024)
        scored_papers.append((paper, score))
    
    # Sort by score
    scored_papers.sort(key=lambda x: x[1].total_score, reverse=True)
    
    # Display results table
    table = Table(title="Citation Quality Scores", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Paper", style="cyan", width=40)
    table.add_column("Year", justify="center", width=6)
    table.add_column("Citations", justify="right", style="yellow", width=10)
    table.add_column("Total Score", justify="right", style="green", width=12)
    table.add_column("Grade", justify="center", width=8)
    
    for i, (paper, score) in enumerate(scored_papers, 1):
        # Determine grade
        if score.total_score >= 80:
            grade = "[green]A[/green]"
        elif score.total_score >= 70:
            grade = "[yellow]B[/yellow]"
        elif score.total_score >= 60:
            grade = "[orange3]C[/orange3]"
        else:
            grade = "[red]D[/red]"
            
        table.add_row(
            str(i),
            paper["title"][:40] + "..." if len(paper["title"]) > 40 else paper["title"],
            str(paper["year"]),
            f"{paper['citation_count']:,}",
            f"{score.total_score:.1f}",
            grade
        )
    
    console.print(table)
    
    return scored_papers


def demo_detailed_breakdown(scored_papers):
    """Show detailed score breakdown for top papers."""
    console.print("\n[bold blue]Detailed Score Breakdown[/bold blue]\n")
    
    # Show top 3 papers
    for i, (paper, score) in enumerate(scored_papers[:3], 1):
        console.print(f"\n[bold]#{i} {paper['title']}[/bold]")
        console.print(f"[dim]Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}[/dim]")
        
        # Create score breakdown
        breakdown = Table(show_header=False, box=None, padding=(0, 2))
        breakdown.add_column("Component", style="white")
        breakdown.add_column("Score", justify="right", style="cyan")
        breakdown.add_column("Weight", justify="right", style="dim")
        breakdown.add_column("Contribution", justify="right", style="yellow")
        
        # Calculate contributions
        scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
        contributions = {
            "Venue Quality": (score.venue_score, scorer.weights["venue"]),
            "Citation Impact": (score.impact_score, scorer.weights["impact"]),
            "Author Credibility": (score.author_score, scorer.weights["author"]),
            "Recency": (score.recency_score, scorer.weights["recency"]),
            "Database Consensus": (score.consensus_score, scorer.weights["consensus"])
        }
        
        for component, (comp_score, weight) in contributions.items():
            contribution = comp_score * weight * 100
            breakdown.add_row(
                component,
                f"{comp_score:.2f}",
                f"{weight:.0%}",
                f"{contribution:.1f}"
            )
        
        console.print(breakdown)
        
        # Show penalties if any
        if score.penalties:
            console.print("\n[red]Penalties:[/red]")
            for penalty, value in score.penalties.items():
                console.print(f"  - {penalty}: -{value:.0%}")
        
        # Show warnings if any
        if score.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in score.warnings:
                console.print(f"  ⚠ {warning}")
        
        # Show explanation
        console.print(f"\n[dim]Summary: {score.explanation}[/dim]")
        console.print("-" * 80)


def demo_field_comparison():
    """Compare scoring across different fields."""
    console.print("\n[bold blue]Field-Specific Scoring Comparison[/bold blue]\n")
    
    # Create a paper that will be scored differently by field
    test_paper = {
        "title": "A Mathematical Theory of Communication",
        "authors": ["Shannon, C.E."],
        "venue": "Bell System Technical Journal",
        "year": 1948,
        "citation_count": 100000,
        "database_appearances": 2
    }
    
    fields = [
        FieldOfStudy.COMPUTER_SCIENCE,
        FieldOfStudy.MATHEMATICS,
        FieldOfStudy.ENGINEERING
    ]
    
    console.print(f"[cyan]Paper:[/cyan] {test_paper['title']} ({test_paper['year']})")
    console.print(f"[cyan]Citations:[/cyan] {test_paper['citation_count']:,}\n")
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Field", style="cyan")
    table.add_column("Total Score", justify="right", style="green")
    table.add_column("Recency Score", justify="right")
    table.add_column("Impact Score", justify="right")
    table.add_column("Notes", style="dim")
    
    for field in fields:
        scorer = CitationScorer(field=field)
        score = scorer.score_citation(test_paper, current_year=2024)
        
        notes = []
        if field == FieldOfStudy.MATHEMATICS:
            notes.append("Values timeless work")
        elif field == FieldOfStudy.COMPUTER_SCIENCE:
            notes.append("Prefers recent papers")
            
        table.add_row(
            field.name.replace("_", " ").title(),
            f"{score.total_score:.1f}",
            f"{score.recency_score:.2f}",
            f"{score.impact_score:.2f}",
            ", ".join(notes)
        )
    
    console.print(table)


def demo_self_citation_detection():
    """Demonstrate self-citation detection."""
    console.print("\n[bold blue]Self-Citation Detection Demo[/bold blue]\n")
    
    paper = {
        "title": "Our Previous Work Extended",
        "authors": ["Smith, John", "Doe, Jane", "Johnson, Bob", "Williams, Alice"],
        "venue": "Conference 2023",
        "year": 2023,
        "citation_count": 50
    }
    
    scenarios = [
        {
            "name": "No self-citation",
            "citing_authors": ["Brown, Charlie", "Davis, Eve"]
        },
        {
            "name": "Partial self-citation",
            "citing_authors": ["Smith, John", "Brown, Charlie", "Davis, Eve"]
        },
        {
            "name": "High self-citation",
            "citing_authors": ["Smith, John", "Doe, Jane", "Johnson, Bob"]
        }
    ]
    
    scorer = CitationScorer()
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Scenario", style="cyan")
    table.add_column("Self-Citation Ratio", justify="right")
    table.add_column("Penalty", justify="right", style="red")
    table.add_column("Total Score", justify="right", style="green")
    table.add_column("Warning", style="yellow")
    
    for scenario in scenarios:
        score = scorer.score_citation(paper, citing_authors=scenario["citing_authors"])
        
        penalty = score.penalties.get("self_citation", 0)
        warning = "Yes" if any("self-citation" in w for w in score.warnings) else "No"
        
        table.add_row(
            scenario["name"],
            f"{score.self_citation_ratio:.2f}",
            f"{penalty:.0%}" if penalty > 0 else "-",
            f"{score.total_score:.1f}",
            warning
        )
    
    console.print(table)


def demo_suspicious_patterns():
    """Demonstrate detection of suspicious citation patterns."""
    console.print("\n[bold blue]Suspicious Pattern Detection[/bold blue]\n")
    
    papers = [
        {
            "name": "Normal paper",
            "paper": {
                "title": "Regular Research Paper",
                "year": 2020,
                "citation_count": 100,
                "venue": "Good Conference"
            }
        },
        {
            "name": "Suspiciously high citations",
            "paper": {
                "title": "Too Good To Be True",
                "year": 2023,
                "citation_count": 5000,  # Way too high for 1 year
                "venue": "Unknown Venue"
            }
        },
        {
            "name": "Predatory venue",
            "paper": {
                "title": "Published Anywhere",
                "year": 2023,
                "citation_count": 10,
                "venue": "Predatory Journal",
                "venue_metrics": VenueMetrics(
                    name="Predatory Journal",
                    is_predatory=True
                )
            }
        },
        {
            "name": "Retracted paper",
            "paper": {
                "title": "Fraudulent Results",
                "year": 2021,
                "citation_count": 500,
                "venue": "Nature",
                "is_retracted": True
            }
        }
    ]
    
    scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
    
    for item in papers:
        score = scorer.score_citation(item["paper"], current_year=2024)
        
        panel_content = f"[bold]Score:[/bold] {score.total_score:.1f}/100\n"
        
        if score.warnings:
            panel_content += "\n[yellow]Warnings:[/yellow]\n"
            for warning in score.warnings:
                panel_content += f"  ⚠ {warning}\n"
                
        if score.penalties:
            panel_content += "\n[red]Penalties:[/red]\n"
            for penalty, value in score.penalties.items():
                panel_content += f"  - {penalty}: -{value:.0%}\n"
                
        panel = Panel(
            panel_content,
            title=f"[cyan]{item['name']}[/cyan]",
            border_style="red" if score.total_score < 50 else "green"
        )
        console.print(panel)


def main():
    """Run all demos."""
    console.print("""
[bold green]Citation Quality Scoring System Demo[/bold green]

This demo shows how the system evaluates citation quality using:
- Venue reputation and impact factor
- Citation count and velocity
- Author credibility metrics
- Recency (field-dependent)
- Cross-database verification
- Self-citation detection
- Suspicious pattern identification
""")
    
    # Run demos
    scored_papers = demo_basic_scoring()
    demo_detailed_breakdown(scored_papers)
    demo_field_comparison()
    demo_self_citation_detection()
    demo_suspicious_patterns()
    
    # Summary
    console.print("\n[bold green]Summary[/bold green]")
    console.print("""
The citation scorer provides nuanced quality assessment that:
- Adapts to different academic fields
- Detects problematic citation patterns
- Rewards high-quality, well-verified sources
- Penalizes suspicious or low-quality citations
- Provides transparent scoring explanations

This helps researchers identify the most trustworthy and relevant citations
for their work, improving the overall quality of academic discourse.
""")


if __name__ == "__main__":
    main()