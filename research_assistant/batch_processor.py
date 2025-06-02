"""
Batch processing capabilities for Research Assistant.

Handles multiple queries, files, and citation requests efficiently.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import csv

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

from .multi_database_search import MultiDatabaseSearch
from .claim_extractor import ClaimExtractor
from .citation_finder import CitationFinder
from .citation_scorer import CitationScorer, FieldOfStudy
from .config import Config
from .arxiv_collector import ArxivCollector
from .semantic_scholar_collector import SemanticScholarCollector

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class BatchJob:
    """Represents a single batch processing job."""
    id: str
    type: str  # 'search', 'cite', 'score'
    input_data: Dict[str, Any]
    status: str = 'pending'  # pending, processing, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get job duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class BatchProcessor:
    """
    Processes multiple research tasks in batch mode.
    
    Supports:
    - Multiple search queries
    - Multiple documents for citation finding
    - Bulk paper scoring
    - CSV/JSON input and output
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.jobs: List[BatchJob] = []
        self.results_dir = Path("batch_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_batch_file(self, file_path: Path) -> List[BatchJob]:
        """Load batch jobs from file (CSV or JSON)."""
        jobs = []
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for i, item in enumerate(data):
                job = BatchJob(
                    id=f"job_{i:04d}",
                    type=item['type'],
                    input_data=item['data']
                )
                jobs.append(job)
                
        elif file_path.suffix.lower() == '.csv':
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    job_type = row.get('type', 'search')
                    
                    if job_type == 'search':
                        input_data = {
                            'query': row['query'],
                            'databases': row.get('databases', 'all').split(','),
                            'limit': int(row.get('limit', 20))
                        }
                    elif job_type == 'cite':
                        input_data = {
                            'text': row.get('text', ''),
                            'file': row.get('file', '')
                        }
                    else:
                        input_data = row
                        
                    job = BatchJob(
                        id=f"job_{i:04d}",
                        type=job_type,
                        input_data=input_data
                    )
                    jobs.append(job)
                    
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
        self.jobs.extend(jobs)
        return jobs
        
    async def process_search_job(self, job: BatchJob) -> Dict[str, Any]:
        """Process a search job."""
        query = job.input_data['query']
        databases = job.input_data.get('databases', ['all'])
        limit = job.input_data.get('limit', 20)
        
        if 'all' in databases:
            databases = ['arxiv', 'semantic_scholar', 'pubmed']
            
        search_engine = MultiDatabaseSearch(
            config=self.config,
            enable_arxiv='arxiv' in databases,
            enable_semantic_scholar='semantic_scholar' in databases,
            enable_pubmed='pubmed' in databases
        )
        
        async with search_engine:
            papers = await search_engine.search(
                query,
                max_results_per_db=limit,
                deduplicate=True
            )
            
        # Score each paper
        scorer = CitationScorer()
        scored_papers = []
        
        for paper in papers:
            paper_dict = {
                'title': paper.title,
                'authors': paper.authors,
                'year': paper.year,
                'venue': paper.venue,
                'citation_count': paper.citation_count,
                'database_appearances': paper.consensus_score
            }
            score = scorer.score_citation(paper_dict)
            
            scored_papers.append({
                'paper': paper_dict,
                'score': score.total_score,
                'explanation': score.explanation
            })
            
        return {
            'query': query,
            'total_results': len(papers),
            'papers': scored_papers[:limit]
        }
        
    async def process_citation_job(self, job: BatchJob) -> Dict[str, Any]:
        """Process a citation finding job."""
        text = job.input_data.get('text', '')
        file_path = job.input_data.get('file', '')
        
        if file_path and Path(file_path).exists():
            with open(file_path, 'r') as f:
                text = f.read()
                
        if not text:
            raise ValueError("No text provided for citation finding")
            
        # Extract claims
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(text)
        
        if not claims:
            return {
                'text_preview': text[:200],
                'claims_found': 0,
                'citations': []
            }
            
        # Find citations
        citations = []
        async with ArxivCollector() as arxiv, \
                   SemanticScholarCollector() as s2:
            
            finder = CitationFinder(
                config=self.config,
                arxiv_collector=arxiv,
                semantic_scholar_collector=s2
            )
            
            for claim in claims:
                recommendation = await finder.find_citations_for_claim(claim)
                
                if recommendation.candidates:
                    citations.append({
                        'claim': claim.text,
                        'claim_type': claim.claim_type.value,
                        'confidence': claim.confidence,
                        'citations': [
                            {
                                'title': c.title,
                                'authors': c.authors[:3],
                                'year': c.year,
                                'score': c.relevance_score
                            }
                            for c in recommendation.candidates[:3]
                        ]
                    })
                    
        return {
            'text_preview': text[:200],
            'claims_found': len(claims),
            'citations': citations
        }
        
    async def process_jobs(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[BatchJob]:
        """Process all jobs with optional progress callback."""
        total_jobs = len(self.jobs)
        completed = 0
        
        for job in self.jobs:
            job.status = 'processing'
            job.start_time = datetime.now()
            
            try:
                if job.type == 'search':
                    job.result = await self.process_search_job(job)
                elif job.type == 'cite':
                    job.result = await self.process_citation_job(job)
                else:
                    raise ValueError(f"Unknown job type: {job.type}")
                    
                job.status = 'completed'
                
            except Exception as e:
                logger.error(f"Job {job.id} failed: {e}")
                job.status = 'failed'
                job.error = str(e)
                
            finally:
                job.end_time = datetime.now()
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total_jobs)
                    
        return self.jobs
        
    def save_results(
        self,
        output_format: str = 'json',
        output_file: Optional[Path] = None
    ) -> Path:
        """Save batch processing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not output_file:
            output_file = self.results_dir / f"batch_results_{timestamp}.{output_format}"
            
        if output_format == 'json':
            results = []
            for job in self.jobs:
                results.append({
                    'id': job.id,
                    'type': job.type,
                    'status': job.status,
                    'duration': job.duration,
                    'input': job.input_data,
                    'result': job.result,
                    'error': job.error
                })
                
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
        elif output_format == 'csv':
            # Flatten results for CSV
            with open(output_file, 'w', newline='') as f:
                writer = None
                
                for job in self.jobs:
                    if job.status != 'completed':
                        continue
                        
                    if job.type == 'search' and job.result:
                        for paper in job.result.get('papers', []):
                            row = {
                                'job_id': job.id,
                                'query': job.result['query'],
                                'title': paper['paper']['title'],
                                'authors': '; '.join(paper['paper']['authors'][:3]),
                                'year': paper['paper']['year'],
                                'score': paper['score'],
                                'explanation': paper['explanation']
                            }
                            
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=row.keys())
                                writer.writeheader()
                                
                            writer.writerow(row)
                            
        return output_file
        
    def generate_report(self) -> str:
        """Generate a summary report of batch processing."""
        total = len(self.jobs)
        completed = sum(1 for j in self.jobs if j.status == 'completed')
        failed = sum(1 for j in self.jobs if j.status == 'failed')
        
        total_duration = sum(j.duration or 0 for j in self.jobs)
        
        report = f"""
Batch Processing Report
======================

Total Jobs: {total}
Completed: {completed}
Failed: {failed}
Success Rate: {completed/total*100:.1f}%

Total Processing Time: {total_duration:.1f} seconds
Average Time per Job: {total_duration/total:.1f} seconds

Job Types:
"""
        
        # Count by type
        type_counts = {}
        for job in self.jobs:
            type_counts[job.type] = type_counts.get(job.type, 0) + 1
            
        for job_type, count in type_counts.items():
            report += f"  - {job_type}: {count}\n"
            
        # Failed jobs
        if failed > 0:
            report += "\nFailed Jobs:\n"
            for job in self.jobs:
                if job.status == 'failed':
                    report += f"  - {job.id}: {job.error}\n"
                    
        return report


async def process_batch_file(
    file_path: Path,
    config: Optional[Config] = None,
    output_format: str = 'json'
) -> Path:
    """
    Convenience function to process a batch file.
    
    Args:
        file_path: Path to batch file (CSV or JSON)
        config: Configuration object
        output_format: Output format (json or csv)
        
    Returns:
        Path to output file
    """
    processor = BatchProcessor(config)
    
    # Load jobs
    console.print(f"[cyan]Loading batch file: {file_path}[/cyan]")
    jobs = processor.load_batch_file(file_path)
    console.print(f"[green]Loaded {len(jobs)} jobs[/green]\n")
    
    # Process with progress
    with Progress(console=console) as progress:
        task = progress.add_task("[yellow]Processing jobs...", total=len(jobs))
        
        def update_progress(completed, total):
            progress.update(task, completed=completed)
            
        await processor.process_jobs(progress_callback=update_progress)
        
    # Save results
    output_file = processor.save_results(output_format)
    console.print(f"\n[green]Results saved to: {output_file}[/green]")
    
    # Show report
    report = processor.generate_report()
    console.print("\n" + report)
    
    return output_file