"""
ArXiv API collector for research papers.

This module handles searching and collecting papers from ArXiv using their REST API.
Implements rate limiting and proper error handling based on ArXiv API guidelines.
Reference: https://arxiv.org/help/api/index
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

try:
    import defusedxml.ElementTree as ET
except ImportError:
    # Fallback to standard ET with security configurations
    from xml.etree import ElementTree as ET
    # Configure ET to be more secure
    ET.XMLParser = ET.XMLParser(
        resolve_entities=False,
        forbid_dtd=True,
        forbid_entities=True,
        forbid_external=True
    )

import aiohttp
from aiohttp import ClientTimeout

from .rate_limiter import UnifiedRateLimiter, APIType
from .validators import Validators, ValidationError
from .resource_manager import get_resource_manager, ResourceExhaustedError
from .exceptions import APIError, APITimeoutError, RateLimitError, ensure_no_silent_failures

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents a paper from ArXiv with essential metadata."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    updated_date: datetime
    categories: List[str]
    pdf_url: str
    arxiv_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    primary_category: Optional[str] = None
    comment: Optional[str] = None


class ArxivCollector:
    """
    Collects research papers from ArXiv API with rate limiting.
    
    ArXiv recommends no more than 3 requests per second and bulk downloads
    should use their bulk data access methods. This implementation respects
    those limits.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    NAMESPACE = {'atom': 'http://www.w3.org/2005/Atom'}
    DEFAULT_MAX_RESULTS = 100
    
    def __init__(self, unified_limiter: Optional[UnifiedRateLimiter] = None):
        """
        Initialize the ArXiv collector.
        
        Args:
            unified_limiter: Optional unified rate limiter instance.
        """
        self.unified_limiter = unified_limiter or UnifiedRateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Context manager entry."""
        timeout = ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.session:
            await self.session.close()
            
    @ensure_no_silent_failures
    async def search(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> List[ArxivPaper]:
        """
        Search ArXiv for papers matching the query.
        
        Args:
            query: Search query using ArXiv query syntax
                   Examples: "cat:cs.AI", "au:Bengio", "ti:transformer"
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort criterion ("relevance", "lastUpdatedDate", "submittedDate")
            sort_order: Sort order ("ascending" or "descending")
            
        Returns:
            List of ArxivPaper objects
            
        Raises:
            ValueError: If parameters are invalid
            aiohttp.ClientError: If network request fails
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        # Validate inputs
        query = Validators.validate_query(query)
        max_results = Validators.validate_limit(max_results)
        
        if sort_by not in ["relevance", "lastUpdatedDate", "submittedDate"]:
            raise ValidationError("Invalid sort_by parameter", field="sort_by", value=sort_by)
            
        if sort_order not in ["ascending", "descending"]:
            raise ValidationError("Invalid sort_order parameter", field="sort_order", value=sort_order)
            
        resource_manager = get_resource_manager()
        
        params = {
            "search_query": query,
            "max_results": max_results,
            "start": start,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.ARXIV)
            logger.info(f"Searching ArXiv with query: {query}")
            
            try:
                async with self.session.get(self.BASE_URL, params=params) as response:
                    # Check response size
                    if response.headers.get('content-length'):
                        size = int(response.headers['content-length'])
                        resource_manager.check_response_size(size)
                        
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.ARXIV, 
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("arxiv", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    content = await response.text()
                    
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.ARXIV)
                return self._parse_arxiv_response(content)
                
            except asyncio.TimeoutError:
                raise APITimeoutError("arxiv", 30.0)
            except aiohttp.ClientError as e:
                if hasattr(e, 'status') and e.status == 429:
                    self.unified_limiter.on_rate_limit_error(APIType.ARXIV)
                raise APIError(
                    message=f"ArXiv API request failed: {e}",
                    service="arxiv",
                    status_code=getattr(e, 'status', None)
                )
                
    def _parse_arxiv_response(self, xml_content: str) -> List[ArxivPaper]:
        """
        Parse ArXiv API XML response into ArxivPaper objects.
        
        Args:
            xml_content: XML response from ArXiv API
            
        Returns:
            List of parsed ArxivPaper objects
        """
        # Validate input
        if not xml_content or not isinstance(xml_content, str):
            logger.warning("Invalid XML content provided")
            return []
            
        # Size limit to prevent XXE attacks
        if len(xml_content) > 10 * 1024 * 1024:  # 10MB limit
            logger.error("XML content exceeds size limit")
            return []
            
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML: {e}")
            return []
            
        papers = []
        
        for entry in root.findall('atom:entry', self.NAMESPACE):
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse ArXiv entry: {e}")
                continue
                
        return papers
        
    def _parse_entry(self, entry: Any) -> ArxivPaper:
        """Parse a single entry from ArXiv XML response."""
        # Extract ArXiv ID from the id URL
        id_url = entry.find('atom:id', self.NAMESPACE).text
        arxiv_id = id_url.split('/')[-1]
        
        # Parse dates
        published = entry.find('atom:published', self.NAMESPACE).text
        updated = entry.find('atom:updated', self.NAMESPACE).text
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', self.NAMESPACE):
            name = author.find('atom:name', self.NAMESPACE).text
            authors.append(name)
            
        # Extract categories
        categories = []
        primary_category = None
        
        primary_elem = entry.find('arxiv:primary_category', 
                                  {'arxiv': 'http://arxiv.org/schemas/atom'})
        if primary_elem is not None:
            primary_category = primary_elem.get('term')
            categories.append(primary_category)
            
        for category in entry.findall('atom:category', self.NAMESPACE):
            term = category.get('term')
            if term and term not in categories:
                categories.append(term)
                
        # Extract URLs
        pdf_url = None
        arxiv_url = None
        
        for link in entry.findall('atom:link', self.NAMESPACE):
            if link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
            elif link.get('type') == 'text/html':
                arxiv_url = link.get('href')
                
        # Extract optional fields
        doi_elem = entry.find('arxiv:doi', {'arxiv': 'http://arxiv.org/schemas/atom'})
        doi = doi_elem.text if doi_elem is not None else None
        
        journal_elem = entry.find('arxiv:journal_ref', 
                                  {'arxiv': 'http://arxiv.org/schemas/atom'})
        journal_ref = journal_elem.text if journal_elem is not None else None
        
        comment_elem = entry.find('arxiv:comment', 
                                  {'arxiv': 'http://arxiv.org/schemas/atom'})
        comment = comment_elem.text if comment_elem is not None else None
        
        return ArxivPaper(
            arxiv_id=arxiv_id,
            title=entry.find('atom:title', self.NAMESPACE).text.strip(),
            authors=authors,
            abstract=entry.find('atom:summary', self.NAMESPACE).text.strip(),
            published_date=datetime.fromisoformat(published.replace('Z', '+00:00')),
            updated_date=datetime.fromisoformat(updated.replace('Z', '+00:00')),
            categories=categories,
            pdf_url=pdf_url,
            arxiv_url=arxiv_url,
            doi=doi,
            journal_ref=journal_ref,
            primary_category=primary_category,
            comment=comment
        )
        
    async def get_paper_by_id(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """
        Fetch a specific paper by its ArXiv ID.
        
        Args:
            arxiv_id: ArXiv ID (e.g., "2103.15691" or "cs/0301001")
            
        Returns:
            ArxivPaper object if found, None otherwise
        """
        # Clean the ID (remove version if present)
        clean_id = arxiv_id.split('v')[0]
        
        papers = await self.search(f"id:{clean_id}", max_results=1)
        return papers[0] if papers else None
        
    async def search_by_author(
        self, 
        author: str, 
        max_results: int = DEFAULT_MAX_RESULTS
    ) -> List[ArxivPaper]:
        """
        Search for papers by author name.
        
        Args:
            author: Author name to search for
            max_results: Maximum number of results
            
        Returns:
            List of papers by the specified author
        """
        # ArXiv uses "au:" prefix for author searches
        return await self.search(f'au:"{author}"', max_results=max_results)
        
    async def search_by_category(
        self,
        category: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        days_back: Optional[int] = None
    ) -> List[ArxivPaper]:
        """
        Search for recent papers in a specific category.
        
        Args:
            category: ArXiv category (e.g., "cs.AI", "math.CO")
            max_results: Maximum number of results
            days_back: If specified, only returns papers from last N days
            
        Returns:
            List of papers in the specified category
        """
        query = f"cat:{category}"
        
        if days_back:
            # ArXiv doesn't support date queries directly, so we'll filter after
            papers = await self.search(
                query, 
                max_results=max_results * 2,  # Get extra to account for filtering
                sort_by="lastUpdatedDate"
            )
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_papers = [
                p for p in papers 
                if p.updated_date.replace(tzinfo=None) > cutoff_date
            ]
            return filtered_papers[:max_results]
            
        return await self.search(query, max_results=max_results)