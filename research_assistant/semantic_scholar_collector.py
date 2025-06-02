"""
Semantic Scholar API collector for research papers.

This module interfaces with the Semantic Scholar Academic Graph API to collect
paper metadata, citations, and references. Implements proper rate limiting per
their API guidelines (100 requests per 5 minutes for anonymous users).
Reference: https://api.semanticscholar.org/
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Set

import aiohttp
from aiohttp import ClientTimeout

from .rate_limiter import UnifiedRateLimiter, APIType
from .exceptions import RateLimitError
from .resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


@dataclass
class SemanticScholarPaper:
    """Represents a paper from Semantic Scholar with rich metadata."""
    paper_id: str
    title: str
    abstract: Optional[str]
    authors: List[Dict[str, Any]]
    year: Optional[int]
    venue: Optional[str]
    publication_date: Optional[str]
    citation_count: int
    reference_count: int
    influential_citation_count: int
    is_open_access: bool
    fields_of_study: List[str]
    s2_url: str
    external_ids: Dict[str, str] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    tldr: Optional[Dict[str, str]] = None


class SemanticScholarCollector:
    """
    Collects research papers from Semantic Scholar API.
    
    The API provides rich metadata including citation graphs, author information,
    and paper embeddings. Free tier is limited to 100 requests per 5 minutes.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_FIELDS = [
        "paperId", "title", "abstract", "authors", "year", "venue",
        "publicationDate", "citationCount", "referenceCount", 
        "influentialCitationCount", "isOpenAccess", "fieldsOfStudy",
        "s2FieldsOfStudy", "url", "externalIds"
    ]
    
    def __init__(self, api_key: Optional[str] = None, unified_limiter: Optional[UnifiedRateLimiter] = None):
        """
        Initialize the Semantic Scholar collector.
        
        Args:
            api_key: Optional API key for higher rate limits
            unified_limiter: Optional unified rate limiter instance.
        """
        self.api_key = api_key
        # Create unified limiter with API key if provided
        api_keys = {"semantic_scholar": api_key} if api_key else {}
        self.unified_limiter = unified_limiter or UnifiedRateLimiter(api_keys=api_keys)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Context manager entry."""
        timeout = ClientTimeout(total=30)
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
            
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.session:
            await self.session.close()
            
    async def search(
        self,
        query: str,
        limit: int = 100,
        offset: int = 0,
        fields: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
        venues: Optional[List[str]] = None,
        fields_of_study: Optional[List[str]] = None,
        open_access_only: bool = False
    ) -> List[SemanticScholarPaper]:
        """
        Search for papers using Semantic Scholar's search endpoint.
        
        Args:
            query: Search query string
            limit: Maximum number of results (max 100 per request)
            offset: Pagination offset
            fields: Specific fields to retrieve (uses DEFAULT_FIELDS if None)
            year_range: Optional tuple of (start_year, end_year)
            venues: Optional list of venues to filter by
            fields_of_study: Optional list of fields to filter by
            open_access_only: If True, only return open access papers
            
        Returns:
            List of SemanticScholarPaper objects
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        fields = fields or self.DEFAULT_FIELDS
        params = {
            "query": query,
            "limit": min(limit, 100),
            "offset": offset,
            "fields": ",".join(fields)
        }
        
        # Build filters
        filters = []
        if year_range:
            filters.append(f"year:{year_range[0]}-{year_range[1]}")
        if venues:
            filters.append(f"venue:{','.join(venues)}")
        if fields_of_study:
            filters.append(f"fieldsOfStudy:{','.join(fields_of_study)}")
        if open_access_only:
            filters.append("openAccessPdf")
            
        if filters:
            params["filter"] = ",".join(filters)
            
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.SEMANTIC_SCHOLAR)
            logger.info(f"Searching Semantic Scholar: {query}")
            
            try:
                async with self.session.get(f"{self.BASE_URL}/paper/search", params=params) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.SEMANTIC_SCHOLAR,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("semantic_scholar", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                papers = []
                for paper_data in data.get("data", []):
                    paper = self._parse_paper(paper_data)
                    papers.append(paper)
                    
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.SEMANTIC_SCHOLAR)
                return papers
                
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.SEMANTIC_SCHOLAR)
                    raise RateLimitError("semantic_scholar", None)
                logger.error(f"Semantic Scholar API request failed: {e}")
                raise
                
    async def get_paper(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None,
        include_citations: bool = False,
        include_references: bool = False
    ) -> Optional[SemanticScholarPaper]:
        """
        Get detailed information about a specific paper.
        
        Args:
            paper_id: Semantic Scholar paper ID or external ID (DOI, ArXiv, etc.)
                     Examples: "649def34f8be52c8b66281af98ae884c09aef38b" or "DOI:10.1038/nature12373"
            fields: Specific fields to retrieve
            include_citations: Whether to fetch citation details
            include_references: Whether to fetch reference details
            
        Returns:
            SemanticScholarPaper object if found, None otherwise
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        fields = fields or self.DEFAULT_FIELDS
        
        # Add citation/reference fields if requested
        if include_citations:
            fields.extend(["citations", "citations.paperId", "citations.title"])
        if include_references:
            fields.extend(["references", "references.paperId", "references.title"])
            
        params = {"fields": ",".join(fields)}
        
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.SEMANTIC_SCHOLAR)
            logger.info(f"Fetching paper: {paper_id}")
            
            try:
                async with self.session.get(f"{self.BASE_URL}/paper/{paper_id}", params=params) as response:
                    if response.status == 404:
                        return None
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.SEMANTIC_SCHOLAR,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("semantic_scholar", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.SEMANTIC_SCHOLAR)
                return self._parse_paper(data)
                
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.SEMANTIC_SCHOLAR)
                    raise RateLimitError("semantic_scholar", None)
                logger.error(f"Failed to fetch paper {paper_id}: {e}")
                raise
                
    async def get_author_papers(
        self,
        author_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[SemanticScholarPaper]:
        """
        Get papers by a specific author.
        
        Args:
            author_id: Semantic Scholar author ID
            limit: Maximum number of papers to retrieve
            fields: Specific fields to retrieve
            
        Returns:
            List of papers by the author
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        fields = fields or self.DEFAULT_FIELDS
        params = {
            "fields": ",".join(fields),
            "limit": min(limit, 1000)  # Max 1000 for author endpoint
        }
        
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.SEMANTIC_SCHOLAR)
            logger.info(f"Fetching papers for author: {author_id}")
            
            try:
                async with self.session.get(f"{self.BASE_URL}/author/{author_id}/papers", params=params) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.SEMANTIC_SCHOLAR,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("semantic_scholar", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                papers = []
                for item in data.get("data", []):
                    paper = self._parse_paper(item)
                    papers.append(paper)
                    
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.SEMANTIC_SCHOLAR)
                return papers
                
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.SEMANTIC_SCHOLAR)
                    raise RateLimitError("semantic_scholar", None)
                logger.error(f"Failed to fetch author papers: {e}")
                raise
                
    async def get_paper_citations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[SemanticScholarPaper]:
        """
        Get papers that cite the given paper.
        
        Args:
            paper_id: Paper ID to get citations for
            limit: Maximum number of citations to retrieve
            fields: Specific fields to retrieve
            
        Returns:
            List of papers that cite the given paper
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        fields = fields or self.DEFAULT_FIELDS
        params = {
            "fields": ",".join(fields),
            "limit": min(limit, 1000)
        }
        
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.SEMANTIC_SCHOLAR)
            logger.info(f"Fetching citations for paper: {paper_id}")
            
            try:
                async with self.session.get(
                    f"{self.BASE_URL}/paper/{paper_id}/citations",
                    params=params
                ) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.SEMANTIC_SCHOLAR,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("semantic_scholar", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                papers = []
                for item in data.get("data", []):
                    if "citingPaper" in item:
                        paper = self._parse_paper(item["citingPaper"])
                        papers.append(paper)
                        
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.SEMANTIC_SCHOLAR)
                return papers
                
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.SEMANTIC_SCHOLAR)
                    raise RateLimitError("semantic_scholar", None)
                logger.error(f"Failed to fetch citations: {e}")
                raise
    
    async def get_recommendations(
        self,
        paper_id: str,
        limit: int = 100,
        fields: Optional[List[str]] = None
    ) -> List[SemanticScholarPaper]:
        """
        Get paper recommendations based on a seed paper.
        
        Uses Semantic Scholar's recommendation engine to find related papers
        based on citation patterns and content similarity.
        
        Args:
            paper_id: Seed paper ID for recommendations
            limit: Maximum number of recommendations
            fields: Specific fields to retrieve
            
        Returns:
            List of recommended papers
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        fields = fields or self.DEFAULT_FIELDS
        params = {
            "fields": ",".join(fields),
            "limit": min(limit, 100)
        }
        
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.SEMANTIC_SCHOLAR)
            logger.info(f"Getting recommendations for paper: {paper_id}")
            
            try:
                async with self.session.get(
                    f"{self.BASE_URL}/recommendations/v1/papers/forpaper/{paper_id}",
                    params=params
                ) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.SEMANTIC_SCHOLAR,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("semantic_scholar", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                papers = []
                for paper_data in data.get("recommendedPapers", []):
                    paper = self._parse_paper(paper_data)
                    papers.append(paper)
                    
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.SEMANTIC_SCHOLAR)
                return papers
                
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.SEMANTIC_SCHOLAR)
                    raise RateLimitError("semantic_scholar", None)
                logger.error(f"Failed to get recommendations: {e}")
                raise
                
    async def batch_get_papers(
        self,
        paper_ids: List[str],
        fields: Optional[List[str]] = None
    ) -> List[SemanticScholarPaper]:
        """
        Batch retrieve multiple papers in a single request.
        
        More efficient than individual requests when fetching multiple papers.
        Limited to 500 papers per request.
        
        Args:
            paper_ids: List of paper IDs to retrieve
            fields: Specific fields to retrieve
            
        Returns:
            List of papers (may be fewer than requested if some not found)
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        if len(paper_ids) > 500:
            raise ValueError("Batch request limited to 500 papers")
            
        fields = fields or self.DEFAULT_FIELDS
        params = {"fields": ",".join(fields)}
        
        # API expects JSON body for batch endpoint
        json_body = {"ids": paper_ids}
        
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.SEMANTIC_SCHOLAR)
            logger.info(f"Batch fetching {len(paper_ids)} papers")
            
            try:
                async with self.session.post(
                    f"{self.BASE_URL}/paper/batch",
                    params=params,
                    json=json_body
                ) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.SEMANTIC_SCHOLAR,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("semantic_scholar", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                papers = []
                for paper_data in data:
                    if paper_data:  # Skip null results
                        paper = self._parse_paper(paper_data)
                        papers.append(paper)
                        
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.SEMANTIC_SCHOLAR)
                return papers
                
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.SEMANTIC_SCHOLAR)
                    raise RateLimitError("semantic_scholar", None)
                logger.error(f"Batch request failed: {e}")
                raise
                
    def _parse_paper(self, data: Dict[str, Any]) -> SemanticScholarPaper:
        """Parse API response into SemanticScholarPaper object."""
        # Handle author data
        authors = []
        for author_data in data.get("authors", []):
            authors.append({
                "authorId": author_data.get("authorId"),
                "name": author_data.get("name"),
                "url": author_data.get("url")
            })
            
        # Extract fields of study
        fields_of_study = []
        for field in data.get("s2FieldsOfStudy", []):
            fields_of_study.append(field.get("category", ""))
            
        # Handle external IDs
        external_ids = data.get("externalIds", {}) or {}
        
        return SemanticScholarPaper(
            paper_id=data.get("paperId", ""),
            title=data.get("title", ""),
            abstract=data.get("abstract"),
            authors=authors,
            year=data.get("year"),
            venue=data.get("venue"),
            publication_date=data.get("publicationDate"),
            citation_count=data.get("citationCount", 0),
            reference_count=data.get("referenceCount", 0),
            influential_citation_count=data.get("influentialCitationCount", 0),
            is_open_access=data.get("isOpenAccess", False),
            fields_of_study=fields_of_study,
            s2_url=data.get("url", ""),
            external_ids=external_ids,
            citations=data.get("citations", []),
            references=data.get("references", []),
            embedding=data.get("embedding"),
            tldr=data.get("tldr")
        )