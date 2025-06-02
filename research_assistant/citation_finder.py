"""
Citation finder that matches claims with relevant papers.

Integrates claim extraction with multi-database search to find
appropriate citations for scientific claims.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .claim_extractor import ClaimExtractor, ExtractedClaim, ClaimType
from .arxiv_collector import ArxivCollector
from .semantic_scholar_collector import SemanticScholarCollector
from .vector_store import FAISSVectorStore
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class CitationCandidate:
    """A potential citation for a claim."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: Optional[str]
    abstract: str
    relevance_score: float
    source_database: str  # arxiv, semantic_scholar, or vector_store
    citation_count: Optional[int] = None
    venue_quality_score: Optional[float] = None
    supporting_text: Optional[str] = None  # Relevant excerpt from paper
    confidence: float = 0.0


@dataclass
class CitationRecommendation:
    """A citation recommendation for a specific claim."""
    claim: ExtractedClaim
    candidates: List[CitationCandidate]
    explanation: str
    citation_needed_confidence: float


class CitationFinder:
    """
    Finds relevant citations for extracted claims.
    
    Searches multiple databases and ranks results by relevance
    and quality metrics.
    """
    
    # High-quality venues (conferences and journals)
    TOP_VENUES = {
        # ML/AI Conferences
        "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP", 
        "NAACL", "AAAI", "IJCAI", "KDD", "WWW", "SIGIR", "RecSys",
        
        # Journals
        "Nature", "Science", "Nature Machine Intelligence", "JMLR",
        "IEEE TPAMI", "ACM Computing Surveys", "Artificial Intelligence",
        
        # ArXiv categories considered high quality
        "cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"
    }
    
    def __init__(
        self,
        config: Optional[Config] = None,
        arxiv_collector: Optional[ArxivCollector] = None,
        semantic_scholar_collector: Optional[SemanticScholarCollector] = None,
        vector_store: Optional[FAISSVectorStore] = None
    ):
        """
        Initialize citation finder.
        
        Args:
            config: Configuration object
            arxiv_collector: ArXiv API client
            semantic_scholar_collector: Semantic Scholar API client
            vector_store: Local vector store of papers
        """
        self.config = config or Config()
        self.claim_extractor = ClaimExtractor()
        
        self.arxiv = arxiv_collector
        self.semantic_scholar = semantic_scholar_collector
        self.vector_store = vector_store
        
    async def find_citations_for_text(
        self,
        text: str,
        max_citations_per_claim: int = 5,
        min_relevance_score: float = 0.7
    ) -> List[CitationRecommendation]:
        """
        Find citations for all claims in a text.
        
        Args:
            text: Input text containing claims
            max_citations_per_claim: Maximum citations to return per claim
            min_relevance_score: Minimum relevance threshold
            
        Returns:
            List of citation recommendations
        """
        # Extract claims
        claims = self.claim_extractor.extract_claims(text)
        
        if not claims:
            logger.info("No claims found in text")
            return []
            
        # Analyze citation needs
        analysis = self.claim_extractor.analyze_citation_needs(claims)
        logger.info(f"Found {analysis['total_claims']} claims, "
                   f"{analysis['suggested_citations_needed']} need citations")
        
        # Find citations for each claim
        recommendations = []
        
        for claim in analysis["high_priority"]:
            recommendation = await self.find_citations_for_claim(
                claim,
                max_citations=max_citations_per_claim,
                min_relevance_score=min_relevance_score
            )
            recommendations.append(recommendation)
            
        return recommendations
        
    async def find_citations_for_claim(
        self,
        claim: ExtractedClaim,
        max_citations: int = 5,
        min_relevance_score: float = 0.7
    ) -> CitationRecommendation:
        """
        Find relevant citations for a specific claim.
        
        Args:
            claim: The claim to find citations for
            max_citations: Maximum number of citations to return
            min_relevance_score: Minimum relevance threshold
            
        Returns:
            Citation recommendation with ranked candidates
        """
        logger.info(f"Finding citations for {claim.claim_type.value} claim: {claim.text[:50]}...")
        
        # Generate search queries based on claim type
        search_queries = self._generate_search_queries(claim)
        
        # Search multiple databases concurrently
        candidates = await self._search_databases(search_queries, claim)
        
        # Rank candidates
        ranked_candidates = self._rank_candidates(candidates, claim)
        
        # Filter by relevance
        filtered_candidates = [
            c for c in ranked_candidates 
            if c.relevance_score >= min_relevance_score
        ][:max_citations]
        
        # Generate explanation
        explanation = self._generate_citation_explanation(claim, filtered_candidates)
        
        return CitationRecommendation(
            claim=claim,
            candidates=filtered_candidates,
            explanation=explanation,
            citation_needed_confidence=claim.confidence
        )
        
    def _generate_search_queries(self, claim: ExtractedClaim) -> List[str]:
        """Generate search queries for different databases."""
        queries = []
        
        # Use suggested search terms
        if claim.suggested_search_terms:
            # Combine search terms intelligently
            base_query = " ".join(claim.suggested_search_terms[:3])
            queries.append(base_query)
            
        # Add claim-type specific queries
        if claim.claim_type == ClaimType.STATISTICAL:
            # For statistical claims, search for the metric
            if "accuracy" in claim.text.lower():
                queries.append(f"{claim.keywords[0] if claim.keywords else ''} accuracy benchmark")
            elif "correlation" in claim.text.lower():
                queries.append(f"{claim.keywords[0] if claim.keywords else ''} correlation analysis")
                
        elif claim.claim_type == ClaimType.METHODOLOGICAL:
            # For methods, search for the technique name
            method_keywords = [k for k in claim.keywords if k[0].isupper()]
            if method_keywords:
                queries.append(f"{method_keywords[0]} method technique")
                
        elif claim.claim_type == ClaimType.COMPARATIVE:
            # For comparisons, search for benchmark papers
            queries.append(f"{' '.join(claim.keywords[:2])} comparison benchmark")
            
        elif claim.claim_type == ClaimType.THEORETICAL:
            # For theoretical claims, search for supporting theory
            queries.append(f"{' '.join(claim.keywords[:2])} theory mechanism")
            
        # Always add a fallback query with the claim text
        queries.append(claim.text[:100])
        
        return list(set(queries))  # Remove duplicates
        
    async def _search_databases(
        self,
        queries: List[str],
        claim: ExtractedClaim
    ) -> List[CitationCandidate]:
        """Search multiple databases for relevant papers."""
        candidates = []
        
        # Prepare search tasks
        tasks = []
        
        for query in queries:
            # Search ArXiv if available
            if self.arxiv:
                tasks.append(self._search_arxiv(query, claim))
                
            # Search Semantic Scholar if available
            if self.semantic_scholar:
                tasks.append(self._search_semantic_scholar(query, claim))
                
            # Search local vector store if available
            if self.vector_store:
                tasks.append(self._search_vector_store(query, claim))
                
        # Execute searches concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Search error: {result}")
                else:
                    candidates.extend(result)
                    
        return candidates
        
    async def _search_arxiv(
        self,
        query: str,
        claim: ExtractedClaim
    ) -> List[CitationCandidate]:
        """Search ArXiv for relevant papers."""
        candidates = []
        
        try:
            papers = await self.arxiv.search(query, max_results=10)
            
            for paper in papers:
                candidate = CitationCandidate(
                    paper_id=f"arxiv:{paper.arxiv_id}",
                    title=paper.title,
                    authors=paper.authors[:5],  # Limit authors
                    year=paper.published_date.year,
                    venue=f"arXiv {paper.primary_category}",
                    abstract=paper.abstract[:500],  # Limit abstract length
                    relevance_score=0.0,  # Will be calculated later
                    source_database="arxiv"
                )
                
                # Set venue quality based on category
                if paper.primary_category in self.TOP_VENUES:
                    candidate.venue_quality_score = 0.8
                else:
                    candidate.venue_quality_score = 0.5
                    
                candidates.append(candidate)
                
        except Exception as e:
            logger.error(f"ArXiv search error: {e}")
            
        return candidates
        
    async def _search_semantic_scholar(
        self,
        query: str,
        claim: ExtractedClaim
    ) -> List[CitationCandidate]:
        """Search Semantic Scholar for relevant papers."""
        candidates = []
        
        try:
            papers = await self.semantic_scholar.search(query, limit=10)
            
            for paper in papers:
                candidate = CitationCandidate(
                    paper_id=f"s2:{paper.paper_id}",
                    title=paper.title,
                    authors=[a["name"] for a in paper.authors[:5]],
                    year=paper.year or 0,
                    venue=paper.venue,
                    abstract=paper.abstract[:500] if paper.abstract else "",
                    relevance_score=0.0,
                    source_database="semantic_scholar",
                    citation_count=paper.citation_count
                )
                
                # Set venue quality
                if paper.venue and any(v in paper.venue for v in self.TOP_VENUES):
                    candidate.venue_quality_score = 0.9
                elif paper.venue:
                    candidate.venue_quality_score = 0.6
                else:
                    candidate.venue_quality_score = 0.4
                    
                candidates.append(candidate)
                
        except Exception as e:
            logger.error(f"Semantic Scholar search error: {e}")
            
        return candidates
        
    async def _search_vector_store(
        self,
        query: str,
        claim: ExtractedClaim
    ) -> List[CitationCandidate]:
        """Search local vector store for relevant papers."""
        candidates = []
        
        try:
            # Use claim context for better search
            search_text = f"{query} {claim.context}"
            results = self.vector_store.search(search_text, k=10)
            
            for doc, score in results:
                metadata = doc.metadata
                
                candidate = CitationCandidate(
                    paper_id=f"local:{doc.id}",
                    title=metadata.get("title", "Unknown"),
                    authors=metadata.get("authors", [])[:5],
                    year=metadata.get("year", 0),
                    venue=metadata.get("venue", ""),
                    abstract=doc.text[:500],
                    relevance_score=float(score),
                    source_database="vector_store",
                    supporting_text=doc.text[:200]  # Include relevant excerpt
                )
                
                # Estimate venue quality
                if metadata.get("venue") and any(v in metadata["venue"] for v in self.TOP_VENUES):
                    candidate.venue_quality_score = 0.9
                else:
                    candidate.venue_quality_score = 0.5
                    
                candidates.append(candidate)
                
        except Exception as e:
            logger.error(f"Vector store search error: {e}")
            
        return candidates
        
    def _rank_candidates(
        self,
        candidates: List[CitationCandidate],
        claim: ExtractedClaim
    ) -> List[CitationCandidate]:
        """
        Rank citation candidates by multiple factors.
        
        Considers:
        - Text relevance to claim
        - Venue quality
        - Citation count
        - Recency
        - Claim type specific factors
        """
        if not candidates:
            return []
            
        # Remove duplicates based on title similarity
        unique_candidates = self._deduplicate_candidates(candidates)
        
        # Calculate relevance scores
        for candidate in unique_candidates:
            candidate.relevance_score = self._calculate_relevance_score(
                candidate, claim
            )
            
        # Sort by relevance score
        ranked = sorted(
            unique_candidates,
            key=lambda c: c.relevance_score,
            reverse=True
        )
        
        return ranked
        
    def _calculate_relevance_score(
        self,
        candidate: CitationCandidate,
        claim: ExtractedClaim
    ) -> float:
        """
        Calculate relevance score for a candidate.
        
        Combines multiple factors into a single score.
        """
        score = 0.0
        
        # Text similarity (if from vector store, already have score)
        if candidate.source_database == "vector_store":
            score += candidate.relevance_score * 0.4
        else:
            # Simple keyword matching for other sources
            keyword_matches = sum(
                1 for keyword in claim.keywords
                if keyword.lower() in candidate.title.lower() 
                or keyword.lower() in candidate.abstract.lower()
            )
            score += min(keyword_matches / max(len(claim.keywords), 1), 1.0) * 0.4
            
        # Venue quality
        if candidate.venue_quality_score:
            score += candidate.venue_quality_score * 0.2
            
        # Citation count (normalized)
        if candidate.citation_count is not None:
            # Log scale for citations
            import math
            citation_score = min(math.log10(candidate.citation_count + 1) / 3, 1.0)
            score += citation_score * 0.2
            
        # Recency bonus (papers from last 5 years)
        current_year = datetime.now().year
        if candidate.year and candidate.year > current_year - 5:
            recency_score = (candidate.year - (current_year - 5)) / 5
            score += recency_score * 0.1
            
        # Claim type specific bonuses
        if claim.claim_type == ClaimType.COMPARATIVE:
            # Bonus for benchmark/comparison papers
            if any(term in candidate.title.lower() 
                   for term in ["benchmark", "comparison", "evaluation", "survey"]):
                score += 0.1
                
        elif claim.claim_type == ClaimType.METHODOLOGICAL:
            # Bonus for papers introducing methods
            if any(term in candidate.title.lower() 
                   for term in ["novel", "new", "proposed", "introducing"]):
                score += 0.1
                
        return min(score, 1.0)  # Cap at 1.0
        
    def _deduplicate_candidates(
        self,
        candidates: List[CitationCandidate]
    ) -> List[CitationCandidate]:
        """Remove duplicate papers based on title similarity."""
        if not candidates:
            return []
            
        unique = []
        seen_titles = set()
        
        for candidate in candidates:
            # Simple deduplication by normalized title
            normalized_title = candidate.title.lower().strip()
            
            # Check if very similar title exists
            is_duplicate = False
            for seen in seen_titles:
                if self._title_similarity(normalized_title, seen) > 0.9:
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique.append(candidate)
                seen_titles.add(normalized_title)
                
        return unique
        
    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate simple title similarity."""
        # Very basic - could use more sophisticated methods
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    def _generate_citation_explanation(
        self,
        claim: ExtractedClaim,
        candidates: List[CitationCandidate]
    ) -> str:
        """Generate explanation for why citations were chosen."""
        if not candidates:
            return (
                f"No suitable citations found for this {claim.claim_type.value} claim. "
                f"Consider revising the claim or providing more context."
            )
            
        explanation = f"Found {len(candidates)} relevant citations for this {claim.claim_type.value} claim. "
        
        # Add type-specific explanation
        if claim.claim_type == ClaimType.STATISTICAL:
            explanation += "These papers provide statistical evidence or benchmarks. "
        elif claim.claim_type == ClaimType.METHODOLOGICAL:
            explanation += "These papers describe the methods or techniques mentioned. "
        elif claim.claim_type == ClaimType.COMPARATIVE:
            explanation += "These papers provide comparative analysis or benchmarks. "
        elif claim.claim_type == ClaimType.THEORETICAL:
            explanation += "These papers discuss the theoretical foundations. "
            
        # Mention top result
        top = candidates[0]
        explanation += f"The most relevant is '{top.title}' "
        if top.venue_quality_score and top.venue_quality_score > 0.8:
            explanation += "from a high-quality venue."
        else:
            explanation += f"with a relevance score of {top.relevance_score:.2f}."
            
        return explanation