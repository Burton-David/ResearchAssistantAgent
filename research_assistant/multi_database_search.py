"""
Unified multi-database search interface.

Provides a single interface to search across ArXiv, Semantic Scholar, PubMed,
and local vector stores simultaneously with result deduplication and ranking.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Union
from datetime import datetime
from enum import Enum

from .arxiv_collector import ArxivCollector, ArxivPaper
from .semantic_scholar_collector import SemanticScholarCollector, SemanticScholarPaper
from .pubmed_collector import PubMedCollector, PubMedPaper
from .vector_store import FAISSVectorStore
from .config import Config
from .rate_limiter import UnifiedRateLimiter

logger = logging.getLogger(__name__)


class PaperSource(Enum):
    """Source database for papers."""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    VECTOR_STORE = "vector_store"


@dataclass
class UnifiedPaper:
    """
    Unified paper representation across all databases.
    
    Normalizes different paper formats into a single interface.
    """
    id: str  # Unique identifier with source prefix
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: Optional[str] = None
    doi: Optional[str] = None
    source: PaperSource = PaperSource.ARXIV
    source_ids: Dict[str, str] = field(default_factory=dict)  # IDs in each database
    citation_count: Optional[int] = None
    keywords: List[str] = field(default_factory=list)
    publication_date: Optional[datetime] = None
    url: str = ""
    
    # Quality metrics
    venue_score: float = 0.0
    citation_velocity: float = 0.0  # Citations per year
    is_retracted: bool = False
    consensus_score: float = 0.0  # Agreement across databases
    
    def __hash__(self):
        """Make paper hashable for deduplication."""
        return hash(self.id)


class MultiDatabaseSearch:
    """
    Unified search interface for multiple paper databases.
    
    Features:
    - Concurrent search across multiple databases
    - Result deduplication based on title/DOI matching
    - Consensus scoring when paper appears in multiple databases
    - Retraction checking
    - Quality-based ranking
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        rate_limiter: Optional[UnifiedRateLimiter] = None,
        enable_arxiv: bool = True,
        enable_semantic_scholar: bool = True,
        enable_pubmed: bool = True,
        enable_vector_store: bool = True,
        pubmed_api_key: Optional[str] = None,
        pubmed_email: Optional[str] = None
    ):
        """
        Initialize multi-database search.
        
        Args:
            config: Configuration object
            rate_limiter: Shared rate limiter
            enable_*: Flags to enable/disable specific databases
            pubmed_api_key: NCBI API key for higher rate limits
            pubmed_email: Email for PubMed API (required)
        """
        self.config = config or Config()
        self.rate_limiter = rate_limiter or UnifiedRateLimiter()
        
        self.enable_arxiv = enable_arxiv
        self.enable_semantic_scholar = enable_semantic_scholar
        self.enable_pubmed = enable_pubmed
        self.enable_vector_store = enable_vector_store
        
        self.pubmed_api_key = pubmed_api_key
        self.pubmed_email = pubmed_email
        
        # Initialize collectors (will be created in context manager)
        self.arxiv: Optional[ArxivCollector] = None
        self.semantic_scholar: Optional[SemanticScholarCollector] = None
        self.pubmed: Optional[PubMedCollector] = None
        self.vector_store: Optional[FAISSVectorStore] = None
        
    async def __aenter__(self):
        """Context manager entry - initialize collectors."""
        # Create enabled collectors
        collectors = []
        
        if self.enable_arxiv:
            self.arxiv = ArxivCollector(unified_limiter=self.rate_limiter)
            collectors.append(self.arxiv.__aenter__())
            
        if self.enable_semantic_scholar:
            self.semantic_scholar = SemanticScholarCollector(unified_limiter=self.rate_limiter)
            collectors.append(self.semantic_scholar.__aenter__())
            
        if self.enable_pubmed:
            self.pubmed = PubMedCollector(
                unified_limiter=self.rate_limiter,
                api_key=self.pubmed_api_key,
                email=self.pubmed_email
            )
            collectors.append(self.pubmed.__aenter__())
            
        # Initialize all collectors
        if collectors:
            await asyncio.gather(*collectors)
            
        # Initialize vector store if enabled
        if self.enable_vector_store:
            self.vector_store = FAISSVectorStore(config=self.config)
            
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup collectors."""
        cleanup_tasks = []
        
        if self.arxiv:
            cleanup_tasks.append(self.arxiv.__aexit__(exc_type, exc_val, exc_tb))
        if self.semantic_scholar:
            cleanup_tasks.append(self.semantic_scholar.__aexit__(exc_type, exc_val, exc_tb))
        if self.pubmed:
            cleanup_tasks.append(self.pubmed.__aexit__(exc_type, exc_val, exc_tb))
            
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
    async def search(
        self,
        query: str,
        max_results_per_db: int = 20,
        deduplicate: bool = True,
        check_retractions: bool = True,
        require_consensus: int = 1,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[UnifiedPaper]:
        """
        Search across all enabled databases.
        
        Args:
            query: Search query
            max_results_per_db: Maximum results from each database
            deduplicate: Whether to remove duplicate papers
            check_retractions: Whether to check for retractions
            require_consensus: Minimum databases paper must appear in
            filters: Database-specific filters
            
        Returns:
            List of unified papers sorted by relevance
        """
        logger.info(f"Multi-database search for: {query}")
        
        # Prepare search tasks
        search_tasks = []
        
        if self.arxiv:
            search_tasks.append(("arxiv", self._search_arxiv(query, max_results_per_db, filters)))
            
        if self.semantic_scholar:
            search_tasks.append(("s2", self._search_semantic_scholar(query, max_results_per_db, filters)))
            
        if self.pubmed:
            search_tasks.append(("pubmed", self._search_pubmed(query, max_results_per_db, filters)))
            
        if self.vector_store:
            search_tasks.append(("vector", self._search_vector_store(query, max_results_per_db)))
            
        # Execute searches concurrently
        if not search_tasks:
            logger.warning("No databases enabled for search")
            return []
            
        # Gather results
        all_papers = []
        database_results = {}
        
        task_names = [name for name, _ in search_tasks]
        tasks = [task for _, task in search_tasks]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for db_name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Search error in {db_name}: {result}")
                database_results[db_name] = []
            else:
                database_results[db_name] = result
                all_papers.extend(result)
                
        logger.info(f"Found {len(all_papers)} total papers across {len(database_results)} databases")
        
        # Process results
        if deduplicate:
            all_papers = self._deduplicate_papers(all_papers)
            logger.info(f"After deduplication: {len(all_papers)} unique papers")
            
        # Calculate consensus scores
        all_papers = self._calculate_consensus_scores(all_papers, database_results)
        
        # Filter by consensus requirement
        if require_consensus > 1:
            all_papers = [p for p in all_papers if p.consensus_score >= require_consensus]
            logger.info(f"After consensus filter: {len(all_papers)} papers")
            
        # Check for retractions if enabled
        if check_retractions:
            all_papers = await self._check_retractions(all_papers)
            
        # Rank papers
        ranked_papers = self._rank_papers(all_papers)
        
        return ranked_papers
        
    async def _search_arxiv(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[UnifiedPaper]:
        """Search ArXiv and convert to unified format."""
        papers = []
        
        try:
            arxiv_filters = filters.get("arxiv", {}) if filters else {}
            results = await self.arxiv.search(
                query,
                max_results=max_results,
                **arxiv_filters
            )
            
            for arxiv_paper in results:
                unified = self._convert_arxiv_paper(arxiv_paper)
                papers.append(unified)
                
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            
        return papers
        
    async def _search_semantic_scholar(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[UnifiedPaper]:
        """Search Semantic Scholar and convert to unified format."""
        papers = []
        
        try:
            s2_filters = filters.get("semantic_scholar", {}) if filters else {}
            results = await self.semantic_scholar.search(
                query,
                limit=max_results,
                **s2_filters
            )
            
            for s2_paper in results:
                unified = self._convert_s2_paper(s2_paper)
                papers.append(unified)
                
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            
        return papers
        
    async def _search_pubmed(
        self,
        query: str,
        max_results: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[UnifiedPaper]:
        """Search PubMed and convert to unified format."""
        papers = []
        
        try:
            pubmed_filters = filters.get("pubmed", {}) if filters else {}
            results = await self.pubmed.search(
                query,
                max_results=max_results,
                **pubmed_filters
            )
            
            for pubmed_paper in results:
                unified = self._convert_pubmed_paper(pubmed_paper)
                papers.append(unified)
                
        except Exception as e:
            logger.error(f"PubMed search error: {e}")
            
        return papers
        
    async def _search_vector_store(
        self,
        query: str,
        max_results: int
    ) -> List[UnifiedPaper]:
        """Search local vector store and convert to unified format."""
        papers = []
        
        try:
            results = self.vector_store.search(query, k=max_results)
            
            for doc, score in results:
                unified = self._convert_vector_doc(doc, score)
                papers.append(unified)
                
        except Exception as e:
            logger.error(f"Vector store search error: {e}")
            
        return papers
        
    def _convert_arxiv_paper(self, paper: ArxivPaper) -> UnifiedPaper:
        """Convert ArXiv paper to unified format."""
        return UnifiedPaper(
            id=f"arxiv:{paper.arxiv_id}",
            title=paper.title,
            authors=paper.authors[:10],  # Limit authors
            abstract=paper.abstract,
            year=paper.published_date.year,
            venue=f"arXiv {paper.primary_category}",
            doi=paper.doi,
            source=PaperSource.ARXIV,
            source_ids={"arxiv": paper.arxiv_id},
            keywords=paper.categories,
            publication_date=paper.published_date,
            url=paper.arxiv_url
        )
        
    def _convert_s2_paper(self, paper: SemanticScholarPaper) -> UnifiedPaper:
        """Convert Semantic Scholar paper to unified format."""
        return UnifiedPaper(
            id=f"s2:{paper.paper_id}",
            title=paper.title,
            authors=[a["name"] for a in paper.authors[:10]],
            abstract=paper.abstract or "",
            year=paper.year or 0,
            venue=paper.venue,
            doi=paper.external_ids.get("DOI"),
            source=PaperSource.SEMANTIC_SCHOLAR,
            source_ids={"s2": paper.paper_id, **paper.external_ids},
            citation_count=paper.citation_count,
            keywords=paper.fields_of_study,
            publication_date=datetime.strptime(paper.publication_date, "%Y-%m-%d") if paper.publication_date else None,
            url=paper.s2_url
        )
        
    def _convert_pubmed_paper(self, paper: PubMedPaper) -> UnifiedPaper:
        """Convert PubMed paper to unified format."""
        source_ids = {"pubmed": paper.pmid}
        if paper.doi:
            source_ids["doi"] = paper.doi
        if paper.pmc_id:
            source_ids["pmc"] = paper.pmc_id
            
        return UnifiedPaper(
            id=f"pubmed:{paper.pmid}",
            title=paper.title,
            authors=paper.authors[:10],
            abstract=paper.abstract,
            year=paper.publication_date.year,
            venue=paper.journal,
            doi=paper.doi,
            source=PaperSource.PUBMED,
            source_ids=source_ids,
            keywords=paper.keywords + paper.mesh_terms[:5],
            publication_date=paper.publication_date,
            url=paper.pubmed_url
        )
        
    def _convert_vector_doc(self, doc: Any, score: float) -> UnifiedPaper:
        """Convert vector store document to unified format."""
        metadata = doc.metadata
        
        return UnifiedPaper(
            id=f"vector:{doc.id}",
            title=metadata.get("title", "Unknown"),
            authors=metadata.get("authors", [])[:10],
            abstract=doc.text[:500],  # Limit abstract
            year=metadata.get("year", 0),
            venue=metadata.get("venue"),
            doi=metadata.get("doi"),
            source=PaperSource.VECTOR_STORE,
            source_ids={"vector": doc.id},
            keywords=metadata.get("keywords", []),
            publication_date=datetime.fromisoformat(metadata["date"]) if "date" in metadata else None
        )
        
    def _deduplicate_papers(self, papers: List[UnifiedPaper]) -> List[UnifiedPaper]:
        """
        Deduplicate papers based on title similarity and DOI.
        
        Merges information from multiple sources into a single paper.
        """
        if not papers:
            return []
            
        # Group papers by DOI (if available)
        doi_groups = {}
        no_doi_papers = []
        
        for paper in papers:
            if paper.doi:
                if paper.doi not in doi_groups:
                    doi_groups[paper.doi] = []
                doi_groups[paper.doi].append(paper)
            else:
                no_doi_papers.append(paper)
                
        # Merge papers with same DOI
        merged_papers = []
        for doi, group in doi_groups.items():
            merged = self._merge_papers(group)
            merged_papers.append(merged)
            
        # Check title similarity for papers without DOI
        for paper in no_doi_papers:
            # Check if similar title exists in merged papers
            is_duplicate = False
            for merged in merged_papers:
                if self._title_similarity(paper.title, merged.title) > 0.85:
                    # Merge into existing paper
                    self._merge_paper_info(merged, paper)
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                merged_papers.append(paper)
                
        return merged_papers
        
    def _merge_papers(self, papers: List[UnifiedPaper]) -> UnifiedPaper:
        """Merge multiple papers into a single unified paper."""
        if len(papers) == 1:
            return papers[0]
            
        # Use the first paper as base
        merged = papers[0]
        
        # Merge information from other papers
        for paper in papers[1:]:
            self._merge_paper_info(merged, paper)
            
        return merged
        
    def _merge_paper_info(self, target: UnifiedPaper, source: UnifiedPaper):
        """Merge information from source paper into target."""
        # Merge source IDs
        target.source_ids.update(source.source_ids)
        
        # Update citation count (use maximum)
        if source.citation_count:
            if target.citation_count:
                target.citation_count = max(target.citation_count, source.citation_count)
            else:
                target.citation_count = source.citation_count
                
        # Merge keywords
        target.keywords = list(set(target.keywords + source.keywords))
        
        # Use better abstract if available
        if not target.abstract and source.abstract:
            target.abstract = source.abstract
        elif source.abstract and len(source.abstract) > len(target.abstract):
            target.abstract = source.abstract
            
        # Update venue if better quality
        if source.venue and not target.venue:
            target.venue = source.venue
            
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity score."""
        # Normalize titles
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()
        
        # Remove punctuation
        import string
        translator = str.maketrans('', '', string.punctuation)
        t1 = t1.translate(translator)
        t2 = t2.translate(translator)
        
        # Calculate Jaccard similarity
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    def _calculate_consensus_scores(
        self,
        papers: List[UnifiedPaper],
        database_results: Dict[str, List[UnifiedPaper]]
    ) -> List[UnifiedPaper]:
        """Calculate consensus scores based on appearance in multiple databases."""
        for paper in papers:
            # Count how many databases contain this paper
            appearances = 0
            for db_name, db_papers in database_results.items():
                for db_paper in db_papers:
                    if self._is_same_paper(paper, db_paper):
                        appearances += 1
                        break
                        
            paper.consensus_score = appearances
            
        return papers
        
    def _is_same_paper(self, paper1: UnifiedPaper, paper2: UnifiedPaper) -> bool:
        """Check if two papers are the same."""
        # Check DOI match
        if paper1.doi and paper2.doi and paper1.doi == paper2.doi:
            return True
            
        # Check source ID overlap
        for source, id1 in paper1.source_ids.items():
            if source in paper2.source_ids and paper2.source_ids[source] == id1:
                return True
                
        # Check title similarity
        if self._title_similarity(paper1.title, paper2.title) > 0.85:
            return True
            
        return False
        
    async def _check_retractions(self, papers: List[UnifiedPaper]) -> List[UnifiedPaper]:
        """Check papers for retractions (placeholder for future implementation)."""
        # TODO: Implement retraction checking via Retraction Watch API
        # For now, just return papers as-is
        return papers
        
    def _rank_papers(self, papers: List[UnifiedPaper]) -> List[UnifiedPaper]:
        """
        Rank papers by multiple quality factors.
        
        Considers:
        - Consensus score (appearance in multiple databases)
        - Citation count and velocity
        - Venue quality
        - Recency
        """
        for paper in papers:
            # Calculate venue score
            paper.venue_score = self._calculate_venue_score(paper.venue)
            
            # Calculate citation velocity
            if paper.citation_count and paper.year:
                years_old = datetime.now().year - paper.year
                if years_old > 0:
                    paper.citation_velocity = paper.citation_count / years_old
                    
        # Sort by combined score
        def score_paper(paper: UnifiedPaper) -> float:
            score = 0.0
            
            # Consensus score (0-1 normalized)
            score += min(paper.consensus_score / 3, 1.0) * 0.3
            
            # Citation velocity (log scale)
            if paper.citation_velocity > 0:
                import math
                score += min(math.log10(paper.citation_velocity + 1) / 2, 1.0) * 0.3
                
            # Venue score
            score += paper.venue_score * 0.2
            
            # Recency bonus
            if paper.year:
                years_old = datetime.now().year - paper.year
                recency_score = max(0, 1 - years_old / 10)  # Decay over 10 years
                score += recency_score * 0.2
                
            return score
            
        ranked = sorted(papers, key=score_paper, reverse=True)
        
        return ranked
        
    def _calculate_venue_score(self, venue: Optional[str]) -> float:
        """Calculate venue quality score."""
        if not venue:
            return 0.0
            
        # High-quality venues
        top_venues = {
            "Nature", "Science", "Cell", "PNAS", "Nature Medicine",
            "Nature Biotechnology", "Nature Genetics", "Lancet", "NEJM",
            "JAMA", "BMJ", "NeurIPS", "ICML", "ICLR", "CVPR", "ACL"
        }
        
        venue_lower = venue.lower()
        
        for top_venue in top_venues:
            if top_venue.lower() in venue_lower:
                return 1.0
                
        # ArXiv categories
        if "arxiv" in venue_lower:
            if any(cat in venue_lower for cat in ["cs.lg", "cs.ai", "cs.cl", "stat.ml"]):
                return 0.7
            else:
                return 0.5
                
        # Other journals
        if any(term in venue_lower for term in ["journal", "conference", "proceedings"]):
            return 0.6
            
        return 0.3