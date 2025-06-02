"""
PubMed/NCBI E-utilities collector for biomedical research papers.

This module interfaces with the NCBI E-utilities API to collect papers from PubMed.
Implements proper rate limiting per NCBI guidelines (3 requests per second without API key,
10 requests per second with API key).
Reference: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from xml.etree import ElementTree as ET

import aiohttp
from aiohttp import ClientTimeout

from .rate_limiter import UnifiedRateLimiter, APIType
from .validators import Validators
from .resource_manager import get_resource_manager
from .exceptions import APIError, APITimeoutError, RateLimitError, ensure_no_silent_failures

logger = logging.getLogger(__name__)


@dataclass
class PubMedPaper:
    """Represents a paper from PubMed with metadata."""
    pmid: str
    title: str
    authors: List[str]
    abstract: str
    publication_date: datetime
    journal: str
    doi: Optional[str] = None
    pmc_id: Optional[str] = None
    mesh_terms: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)
    pubmed_url: str = ""
    
    def __post_init__(self):
        """Generate PubMed URL."""
        if not self.pubmed_url:
            self.pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class PubMedCollector:
    """
    Collects research papers from PubMed using NCBI E-utilities.
    
    The E-utilities API provides programmatic access to NCBI's databases.
    Rate limits: 3 requests/second without API key, 10 requests/second with key.
    """
    
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
    
    DEFAULT_DB = "pubmed"
    DEFAULT_RETMAX = 100
    
    def __init__(
        self, 
        unified_limiter: Optional[UnifiedRateLimiter] = None,
        api_key: Optional[str] = None,
        email: Optional[str] = None
    ):
        """
        Initialize PubMed collector.
        
        Args:
            unified_limiter: Unified rate limiter instance
            api_key: NCBI API key (optional but recommended)
            email: Email address (required by NCBI for identification)
        """
        self.unified_limiter = unified_limiter or UnifiedRateLimiter()
        self.api_key = api_key
        self.email = email or "research_assistant@example.com"
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
        max_results: int = DEFAULT_RETMAX,
        start: int = 0,
        sort: str = "relevance",
        date_range: Optional[tuple[str, str]] = None,
        publication_types: Optional[List[str]] = None
    ) -> List[PubMedPaper]:
        """
        Search PubMed for papers matching the query.
        
        Args:
            query: Search query using PubMed syntax
            max_results: Maximum number of results to return
            start: Starting index for pagination  
            sort: Sort order ("relevance", "date", "author", etc.)
            date_range: Optional tuple of (start_date, end_date) in YYYY/MM/DD format
            publication_types: Optional list of publication types to filter
            
        Returns:
            List of PubMedPaper objects
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        # Validate inputs
        query = Validators.validate_query(query)
        max_results = Validators.validate_limit(max_results)
        
        # First, search for PMIDs
        pmids = await self._search_pmids(
            query, max_results, start, sort, date_range, publication_types
        )
        
        if not pmids:
            return []
            
        # Then fetch full records
        papers = await self._fetch_papers(pmids)
        
        return papers
        
    async def _search_pmids(
        self,
        query: str,
        max_results: int,
        start: int,
        sort: str,
        date_range: Optional[tuple[str, str]],
        publication_types: Optional[List[str]]
    ) -> List[str]:
        """Search for PubMed IDs matching the query."""
        params = {
            "db": self.DEFAULT_DB,
            "term": query,
            "retmax": max_results,
            "retstart": start,
            "sort": sort,
            "retmode": "json",
            "email": self.email
        }
        
        # Add API key if available
        if self.api_key:
            params["api_key"] = self.api_key
            
        # Add date range if specified
        if date_range:
            params["datetype"] = "pdat"
            params["mindate"] = date_range[0]
            params["maxdate"] = date_range[1]
            
        # Add publication type filter
        if publication_types:
            pt_query = " OR ".join([f'"{pt}"[Publication Type]' for pt in publication_types])
            params["term"] = f"({query}) AND ({pt_query})"
            
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.PUBMED)
            logger.info(f"Searching PubMed with query: {query}")
            
            try:
                async with self.session.get(self.ESEARCH_URL, params=params) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.PUBMED,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("pubmed", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                # Extract PMIDs
                result = data.get("esearchresult", {})
                pmids = result.get("idlist", [])
                
                # Log search stats
                count = int(result.get("count", 0))
                logger.info(f"Found {count} total results, retrieved {len(pmids)} PMIDs")
                
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.PUBMED)
                
                return pmids
                
            except aiohttp.ClientError as e:
                logger.error(f"PubMed search failed: {e}")
                raise APIError(f"PubMed search failed: {e}")
                
    async def _fetch_papers(self, pmids: List[str]) -> List[PubMedPaper]:
        """Fetch full paper records for given PMIDs."""
        if not pmids:
            return []
            
        params = {
            "db": self.DEFAULT_DB,
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.PUBMED)
            logger.info(f"Fetching {len(pmids)} papers from PubMed")
            
            try:
                async with self.session.get(self.EFETCH_URL, params=params) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.PUBMED,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("pubmed", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    xml_content = await response.text()
                    
                # Parse XML response
                papers = self._parse_pubmed_xml(xml_content)
                
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.PUBMED)
                
                return papers
                
            except aiohttp.ClientError as e:
                logger.error(f"PubMed fetch failed: {e}")
                raise APIError(f"PubMed fetch failed: {e}")
                
    def _parse_pubmed_xml(self, xml_content: str) -> List[PubMedPaper]:
        """Parse PubMed XML response into paper objects."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")
            return []
            
        for article in root.findall(".//PubmedArticle"):
            try:
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse PubMed article: {e}")
                continue
                
        return papers
        
    def _parse_article(self, article: ET.Element) -> Optional[PubMedPaper]:
        """Parse a single PubMed article XML element."""
        # Extract PMID
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None:
            return None
            
        pmid = pmid_elem.text
        
        # Extract article info
        article_elem = article.find(".//Article")
        if article_elem is None:
            return None
            
        # Title
        title_elem = article_elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else "No title"
        
        # Abstract
        abstract_parts = []
        abstract_elem = article_elem.find(".//Abstract")
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall(".//AbstractText"):
                label = text_elem.get("Label", "")
                text = text_elem.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts)
        
        # Authors
        authors = []
        author_list = article_elem.find(".//AuthorList")
        if author_list is not None:
            for author in author_list.findall(".//Author"):
                last_name = author.findtext("LastName", "")
                fore_name = author.findtext("ForeName", "")
                if last_name:
                    name = f"{last_name}, {fore_name}" if fore_name else last_name
                    authors.append(name)
                    
        # Journal
        journal_elem = article_elem.find(".//Journal")
        journal = ""
        if journal_elem is not None:
            journal_title = journal_elem.findtext(".//Title", "")
            journal_abbr = journal_elem.findtext(".//ISOAbbreviation", "")
            journal = journal_title or journal_abbr
            
        # Publication date
        pub_date = self._extract_publication_date(article_elem)
        
        # DOI
        doi = None
        article_ids = article.find(".//ArticleIdList")
        if article_ids is not None:
            for id_elem in article_ids.findall(".//ArticleId"):
                if id_elem.get("IdType") == "doi":
                    doi = id_elem.text
                    break
                    
        # PMC ID
        pmc_id = None
        if article_ids is not None:
            for id_elem in article_ids.findall(".//ArticleId"):
                if id_elem.get("IdType") == "pmc":
                    pmc_id = id_elem.text
                    break
                    
        # MeSH terms
        mesh_terms = []
        mesh_list = article.find(".//MeshHeadingList")
        if mesh_list is not None:
            for mesh in mesh_list.findall(".//MeshHeading"):
                descriptor = mesh.findtext(".//DescriptorName", "")
                if descriptor:
                    mesh_terms.append(descriptor)
                    
        # Keywords
        keywords = []
        keyword_list = article.find(".//KeywordList")
        if keyword_list is not None:
            for kw in keyword_list.findall(".//Keyword"):
                if kw.text:
                    keywords.append(kw.text)
                    
        # Publication types
        pub_types = []
        pub_type_list = article_elem.find(".//PublicationTypeList")
        if pub_type_list is not None:
            for pt in pub_type_list.findall(".//PublicationType"):
                if pt.text:
                    pub_types.append(pt.text)
                    
        return PubMedPaper(
            pmid=pmid,
            title=title,
            authors=authors,
            abstract=abstract,
            publication_date=pub_date,
            journal=journal,
            doi=doi,
            pmc_id=pmc_id,
            mesh_terms=mesh_terms,
            keywords=keywords,
            publication_types=pub_types
        )
        
    def _extract_publication_date(self, article_elem: ET.Element) -> datetime:
        """Extract publication date from article element."""
        # Try ArticleDate first (electronic publication)
        article_date = article_elem.find(".//ArticleDate[@DateType='Electronic']")
        if article_date is not None:
            year = article_date.findtext("Year")
            month = article_date.findtext("Month")
            day = article_date.findtext("Day")
            if year:
                return self._create_date(year, month, day)
                
        # Fall back to PubDate
        pub_date = article_elem.find(".//PubDate")
        if pub_date is not None:
            year = pub_date.findtext("Year")
            month = pub_date.findtext("Month")
            day = pub_date.findtext("Day")
            if year:
                return self._create_date(year, month, day)
                
        # Default to current date if no date found
        return datetime.now()
        
    def _create_date(self, year: str, month: Optional[str], day: Optional[str]) -> datetime:
        """Create datetime from date components."""
        try:
            year_int = int(year)
            month_int = 1
            day_int = 1
            
            if month:
                # Handle month names
                month_names = {
                    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
                    "may": 5, "jun": 6, "jul": 7, "aug": 8,
                    "sep": 9, "oct": 10, "nov": 11, "dec": 12
                }
                month_lower = month[:3].lower()
                if month_lower in month_names:
                    month_int = month_names[month_lower]
                else:
                    try:
                        month_int = int(month)
                    except ValueError:
                        month_int = 1
                        
            if day:
                try:
                    day_int = int(day)
                except ValueError:
                    day_int = 1
                    
            return datetime(year_int, month_int, day_int)
            
        except (ValueError, TypeError):
            return datetime.now()
            
    async def get_paper_by_pmid(self, pmid: str) -> Optional[PubMedPaper]:
        """
        Fetch a specific paper by its PubMed ID.
        
        Args:
            pmid: PubMed ID
            
        Returns:
            PubMedPaper object if found, None otherwise
        """
        papers = await self._fetch_papers([pmid])
        return papers[0] if papers else None
        
    async def get_similar_papers(
        self,
        pmid: str,
        max_results: int = DEFAULT_RETMAX
    ) -> List[PubMedPaper]:
        """
        Get papers similar to a given paper using PubMed's related articles.
        
        Args:
            pmid: PubMed ID of the source paper
            max_results: Maximum number of similar papers to return
            
        Returns:
            List of similar papers
        """
        if not self.session:
            raise RuntimeError("Use async context manager or call __aenter__")
            
        params = {
            "dbfrom": self.DEFAULT_DB,
            "db": self.DEFAULT_DB,
            "id": pmid,
            "cmd": "neighbor",
            "retmode": "json",
            "email": self.email
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
            
        resource_manager = get_resource_manager()
        
        async with resource_manager.limit_api_request(timeout=30.0):
            await self.unified_limiter.acquire(APIType.PUBMED)
            logger.info(f"Finding papers similar to PMID: {pmid}")
            
            try:
                async with self.session.get(self.ELINK_URL, params=params) as response:
                    if response.status == 429:
                        retry_after = response.headers.get('retry-after')
                        self.unified_limiter.on_rate_limit_error(
                            APIType.PUBMED,
                            int(retry_after) if retry_after else None
                        )
                        raise RateLimitError("pubmed", int(retry_after) if retry_after else None)
                        
                    response.raise_for_status()
                    data = await response.json()
                    
                # Extract related PMIDs
                pmids = []
                link_sets = data.get("linksets", [])
                if link_sets:
                    link_set = link_sets[0]
                    links = link_set.get("linksetdbs", [])
                    for link_db in links:
                        if link_db.get("linkname") == "pubmed_pubmed":
                            pmids = link_db.get("links", [])[:max_results]
                            break
                            
                if not pmids:
                    return []
                    
                # Fetch full records
                papers = await self._fetch_papers(pmids)
                
                # Success - notify rate limiter
                self.unified_limiter.on_success(APIType.PUBMED)
                
                return papers
                
            except aiohttp.ClientError as e:
                logger.error(f"PubMed similar papers search failed: {e}")
                raise APIError(f"PubMed similar papers search failed: {e}")
                
    async def search_clinical_trials(
        self,
        query: str,
        max_results: int = DEFAULT_RETMAX
    ) -> List[PubMedPaper]:
        """
        Search specifically for clinical trials.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            
        Returns:
            List of clinical trial papers
        """
        # Add clinical trial filter to query
        ct_query = f"({query}) AND (Clinical Trial[Publication Type])"
        
        return await self.search(
            ct_query,
            max_results=max_results,
            publication_types=["Clinical Trial", "Randomized Controlled Trial"]
        )
        
    async def search_reviews(
        self,
        query: str,
        max_results: int = DEFAULT_RETMAX,
        systematic_only: bool = False
    ) -> List[PubMedPaper]:
        """
        Search specifically for review articles.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            systematic_only: If True, only return systematic reviews
            
        Returns:
            List of review papers
        """
        if systematic_only:
            pub_types = ["Systematic Review", "Meta-Analysis"]
        else:
            pub_types = ["Review", "Systematic Review", "Meta-Analysis"]
            
        review_query = f"({query}) AND (Review[Publication Type])"
        
        return await self.search(
            review_query,
            max_results=max_results,
            publication_types=pub_types
        )