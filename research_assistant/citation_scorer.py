"""
Citation quality scoring system.

Evaluates citation quality based on multiple factors including venue impact,
citation metrics, author credibility, and field-specific considerations.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import math

logger = logging.getLogger(__name__)


class FieldOfStudy(Enum):
    """Academic fields with different citation patterns."""
    COMPUTER_SCIENCE = "cs"
    BIOLOGY = "bio"
    MEDICINE = "med"
    PHYSICS = "phys"
    CHEMISTRY = "chem"
    MATHEMATICS = "math"
    ENGINEERING = "eng"
    SOCIAL_SCIENCES = "soc"
    GENERAL = "general"


@dataclass
class VenueMetrics:
    """Metrics for academic venues (journals/conferences)."""
    name: str
    impact_factor: Optional[float] = None
    h_index: Optional[int] = None
    tier: Optional[str] = None  # A*, A, B, C for conferences
    field: FieldOfStudy = FieldOfStudy.GENERAL
    is_predatory: bool = False
    acceptance_rate: Optional[float] = None


@dataclass
class AuthorMetrics:
    """Metrics for paper authors."""
    name: str
    h_index: Optional[int] = None
    total_citations: Optional[int] = None
    affiliation: Optional[str] = None
    is_corresponding: bool = False


@dataclass
class CitationScore:
    """Detailed citation quality score."""
    total_score: float  # 0-100
    venue_score: float  # 0-25
    impact_score: float  # 0-25
    author_score: float  # 0-20
    recency_score: float  # 0-15
    consensus_score: float  # 0-15
    
    # Detailed breakdown
    venue_metrics: Optional[VenueMetrics] = None
    citation_count: int = 0
    citation_velocity: float = 0.0
    years_since_publication: int = 0
    self_citation_ratio: float = 0.0
    database_appearances: int = 1
    
    # Penalties
    penalties: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    # Explanation
    explanation: str = ""


class CitationScorer:
    """
    Scores citation quality using multiple metrics.
    
    Provides field-aware scoring that accounts for different citation
    patterns in different academic disciplines.
    """
    
    # Top-tier venues by field
    TOP_VENUES = {
        FieldOfStudy.COMPUTER_SCIENCE: {
            # Conferences
            "NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP",
            "AAAI", "IJCAI", "KDD", "WWW", "SIGMOD", "VLDB", "ICSE", "PLDI",
            # Journals
            "JMLR", "IEEE TPAMI", "ACM Computing Surveys", "Nature Machine Intelligence"
        },
        FieldOfStudy.BIOLOGY: {
            "Nature", "Science", "Cell", "Nature Genetics", "Nature Biotechnology",
            "PNAS", "Current Biology", "EMBO Journal", "Molecular Cell"
        },
        FieldOfStudy.MEDICINE: {
            "NEJM", "Lancet", "JAMA", "BMJ", "Nature Medicine", "Cell",
            "Annals of Internal Medicine", "Circulation", "Journal of Clinical Oncology"
        },
        FieldOfStudy.PHYSICS: {
            "Physical Review Letters", "Nature Physics", "Science", "Nature",
            "Reviews of Modern Physics", "Physical Review X"
        }
    }
    
    # Expected citation rates by field (citations per year)
    FIELD_CITATION_RATES = {
        FieldOfStudy.COMPUTER_SCIENCE: 5.0,  # CS papers cite quickly
        FieldOfStudy.BIOLOGY: 8.0,
        FieldOfStudy.MEDICINE: 10.0,
        FieldOfStudy.PHYSICS: 6.0,
        FieldOfStudy.CHEMISTRY: 7.0,
        FieldOfStudy.MATHEMATICS: 2.0,  # Math cites slowly
        FieldOfStudy.ENGINEERING: 4.0,
        FieldOfStudy.SOCIAL_SCIENCES: 3.0,
        FieldOfStudy.GENERAL: 5.0
    }
    
    def __init__(
        self,
        field: FieldOfStudy = FieldOfStudy.GENERAL,
        custom_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize citation scorer.
        
        Args:
            field: Primary field of study
            custom_weights: Custom scoring weights
        """
        self.field = field
        
        # Default scoring weights
        self.weights = {
            "venue": 0.25,
            "impact": 0.25,
            "author": 0.20,
            "recency": 0.15,
            "consensus": 0.15
        }
        
        if custom_weights:
            self.weights.update(custom_weights)
            
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
    def score_citation(
        self,
        paper_info: Dict[str, Any],
        citing_authors: Optional[List[str]] = None,
        current_year: Optional[int] = None
    ) -> CitationScore:
        """
        Score a citation's quality.
        
        Args:
            paper_info: Dictionary with paper metadata including:
                - title, authors, year, venue, doi
                - citation_count, abstract
                - database_appearances (consensus)
                - venue_metrics (optional)
                - author_metrics (optional)
            citing_authors: Authors of the paper doing the citing (for self-citation)
            current_year: Current year for recency calculation
            
        Returns:
            Detailed citation score
        """
        if current_year is None:
            current_year = datetime.now().year
            
        score = CitationScore(
            total_score=0,
            venue_score=0,
            impact_score=0,
            author_score=0,
            recency_score=0,
            consensus_score=0
        )
        
        # Extract basic info
        year = paper_info.get("year", current_year)
        score.years_since_publication = current_year - year
        score.citation_count = paper_info.get("citation_count", 0)
        score.database_appearances = paper_info.get("database_appearances", 1)
        
        # Calculate sub-scores
        score.venue_score = self._score_venue(paper_info, score)
        score.impact_score = self._score_impact(paper_info, score)
        score.author_score = self._score_authors(paper_info, score)
        score.recency_score = self._score_recency(paper_info, score)
        score.consensus_score = self._score_consensus(paper_info, score)
        
        # Check for self-citations
        if citing_authors:
            self._check_self_citations(paper_info, citing_authors, score)
            
        # Apply penalties
        self._apply_penalties(paper_info, score)
        
        # Calculate total score
        score.total_score = (
            score.venue_score * self.weights["venue"] +
            score.impact_score * self.weights["impact"] +
            score.author_score * self.weights["author"] +
            score.recency_score * self.weights["recency"] +
            score.consensus_score * self.weights["consensus"]
        ) * 100
        
        # Apply penalty multipliers
        for penalty_value in score.penalties.values():
            score.total_score *= (1 - penalty_value)
            
        # Generate explanation
        score.explanation = self._generate_explanation(score)
        
        return score
        
    def _score_venue(self, paper_info: Dict[str, Any], score: CitationScore) -> float:
        """Score venue quality (0-1)."""
        venue = paper_info.get("venue", "")
        if not venue:
            return 0.3  # Unknown venue
            
        # Check if venue metrics provided
        venue_metrics = paper_info.get("venue_metrics")
        if venue_metrics:
            score.venue_metrics = venue_metrics
            
            # Predatory venue check
            if venue_metrics.is_predatory:
                score.warnings.append("Published in potentially predatory venue")
                score.penalties["predatory_venue"] = 0.8
                return 0.0
                
            # Use impact factor if available
            if venue_metrics.impact_factor:
                # Normalize impact factor (field-dependent)
                if self.field == FieldOfStudy.MEDICINE:
                    normalized_if = min(venue_metrics.impact_factor / 50, 1.0)
                elif self.field == FieldOfStudy.COMPUTER_SCIENCE:
                    normalized_if = min(venue_metrics.impact_factor / 10, 1.0)
                else:
                    normalized_if = min(venue_metrics.impact_factor / 20, 1.0)
                return normalized_if
                
        # Check against known top venues
        venue_lower = venue.lower()
        
        # Check field-specific top venues
        field_venues = self.TOP_VENUES.get(self.field, set())
        for top_venue in field_venues:
            if top_venue.lower() in venue_lower:
                return 1.0
                
        # Check general top venues
        all_top_venues = set()
        for venues in self.TOP_VENUES.values():
            all_top_venues.update(venues)
            
        for top_venue in all_top_venues:
            if top_venue.lower() in venue_lower:
                return 0.8  # Good but not field-specific
                
        # ArXiv handling
        if "arxiv" in venue_lower:
            # Some ArXiv categories are better than others
            if any(cat in venue_lower for cat in ["cs.lg", "cs.ai", "cs.cl", "stat.ml"]):
                return 0.6
            return 0.4
            
        # Generic conference/journal
        if any(term in venue_lower for term in ["conference", "proceedings", "journal", "transactions"]):
            return 0.5
            
        return 0.3
        
    def _score_impact(self, paper_info: Dict[str, Any], score: CitationScore) -> float:
        """Score citation impact (0-1)."""
        citations = score.citation_count
        years = max(score.years_since_publication, 1)
        
        # Calculate citation velocity
        score.citation_velocity = citations / years
        
        # Get expected rate for field
        expected_rate = self.FIELD_CITATION_RATES.get(self.field, 5.0)
        
        # Normalize based on field expectations
        if score.citation_velocity >= expected_rate * 3:
            return 1.0  # Highly cited
        elif score.citation_velocity >= expected_rate:
            return 0.7 + (score.citation_velocity - expected_rate) / (expected_rate * 2) * 0.3
        else:
            return score.citation_velocity / expected_rate * 0.7
            
    def _score_authors(self, paper_info: Dict[str, Any], score: CitationScore) -> float:
        """Score author credibility (0-1)."""
        authors = paper_info.get("authors", [])
        if not authors:
            return 0.5
            
        author_metrics = paper_info.get("author_metrics", [])
        
        if author_metrics:
            # Use provided metrics
            total_h_index = sum(m.h_index for m in author_metrics if m.h_index)
            avg_h_index = total_h_index / len(author_metrics) if author_metrics else 0
            
            # Normalize h-index (field-dependent)
            if self.field == FieldOfStudy.MEDICINE:
                normalized_h = min(avg_h_index / 100, 1.0)
            elif self.field == FieldOfStudy.COMPUTER_SCIENCE:
                normalized_h = min(avg_h_index / 50, 1.0)
            else:
                normalized_h = min(avg_h_index / 70, 1.0)
                
            return normalized_h
            
        # Fallback: score based on author count and affiliations
        if len(authors) == 0:
            return 0.3
        elif len(authors) > 20:
            # Very large author list (common in physics/medicine)
            if self.field in [FieldOfStudy.PHYSICS, FieldOfStudy.MEDICINE]:
                return 0.7
            else:
                score.warnings.append("Unusually large author list")
                return 0.5
        else:
            # Normal author count
            return 0.6
            
    def _score_recency(self, paper_info: Dict[str, Any], score: CitationScore) -> float:
        """Score recency/timeliness (0-1)."""
        years_old = score.years_since_publication
        
        # Field-specific recency preferences
        if self.field == FieldOfStudy.COMPUTER_SCIENCE:
            # CS values recency highly
            if years_old <= 2:
                return 1.0
            elif years_old <= 5:
                return 0.8
            elif years_old <= 10:
                return 0.5
            else:
                return 0.3
                
        elif self.field == FieldOfStudy.MATHEMATICS:
            # Math values timelessness
            if years_old <= 5:
                return 0.9
            elif years_old <= 20:
                return 0.8
            else:
                return 0.7
                
        else:
            # General decay
            if years_old <= 3:
                return 1.0
            elif years_old <= 7:
                return 0.7
            elif years_old <= 15:
                return 0.5
            else:
                return 0.3
                
    def _score_consensus(self, paper_info: Dict[str, Any], score: CitationScore) -> float:
        """Score database consensus (0-1)."""
        appearances = score.database_appearances
        
        if appearances >= 3:
            return 1.0
        elif appearances == 2:
            return 0.7
        else:
            return 0.4
            
    def _check_self_citations(
        self,
        paper_info: Dict[str, Any],
        citing_authors: List[str],
        score: CitationScore
    ):
        """Check for self-citations."""
        paper_authors = paper_info.get("authors", [])
        
        # Normalize author names for comparison
        paper_authors_normalized = set()
        for author in paper_authors:
            # Simple normalization - could be improved
            normalized = author.lower().strip()
            paper_authors_normalized.add(normalized)
            # Also add last name only
            if "," in normalized:
                last_name = normalized.split(",")[0].strip()
                paper_authors_normalized.add(last_name)
                
        citing_authors_normalized = set()
        for author in citing_authors:
            normalized = author.lower().strip()
            citing_authors_normalized.add(normalized)
            if "," in normalized:
                last_name = normalized.split(",")[0].strip()
                citing_authors_normalized.add(last_name)
                
        # Calculate overlap
        overlap = paper_authors_normalized.intersection(citing_authors_normalized)
        
        if overlap:
            self_ratio = len(overlap) / len(paper_authors_normalized)
            score.self_citation_ratio = self_ratio
            
            if self_ratio > 0.5:
                score.warnings.append("High self-citation detected")
                score.penalties["self_citation"] = 0.2
            elif self_ratio > 0.3:
                score.warnings.append("Moderate self-citation detected")
                score.penalties["self_citation"] = 0.1
                
    def _apply_penalties(self, paper_info: Dict[str, Any], score: CitationScore):
        """Apply additional penalties."""
        # Retraction check
        if paper_info.get("is_retracted", False):
            score.warnings.append("Paper has been retracted!")
            score.penalties["retracted"] = 0.9
            
        # Missing abstract
        if not paper_info.get("abstract"):
            score.warnings.append("No abstract available")
            score.penalties["no_abstract"] = 0.1
            
        # Suspicious patterns
        if score.citation_velocity > self.FIELD_CITATION_RATES.get(self.field, 5.0) * 10:
            score.warnings.append("Suspiciously high citation rate")
            score.penalties["suspicious_citations"] = 0.3
            
    def _generate_explanation(self, score: CitationScore) -> str:
        """Generate human-readable explanation of the score."""
        explanations = []
        
        # Venue explanation
        if score.venue_score >= 0.8:
            explanations.append("Published in a top-tier venue")
        elif score.venue_score >= 0.5:
            explanations.append("Published in a reputable venue")
        else:
            explanations.append("Venue quality is uncertain")
            
        # Impact explanation
        if score.citation_velocity > 0:
            expected_rate = self.FIELD_CITATION_RATES.get(self.field, 5.0)
            if score.citation_velocity >= expected_rate * 2:
                explanations.append(f"Highly cited ({score.citation_count} citations, {score.citation_velocity:.1f}/year)")
            elif score.citation_velocity >= expected_rate:
                explanations.append(f"Well-cited ({score.citation_count} citations)")
            else:
                explanations.append(f"Moderately cited ({score.citation_count} citations)")
                
        # Consensus explanation
        if score.database_appearances >= 3:
            explanations.append("Found in multiple databases (high confidence)")
        elif score.database_appearances == 2:
            explanations.append("Found in 2 databases")
            
        # Warnings
        if score.warnings:
            explanations.append("Warnings: " + "; ".join(score.warnings))
            
        return " ".join(explanations)
        
    def compare_citations(
        self,
        citations: List[Dict[str, Any]],
        citing_authors: Optional[List[str]] = None
    ) -> List[Tuple[Dict[str, Any], CitationScore]]:
        """
        Compare and rank multiple citations.
        
        Args:
            citations: List of citation candidates
            citing_authors: Authors doing the citing
            
        Returns:
            List of (citation, score) tuples sorted by quality
        """
        scored_citations = []
        
        for citation in citations:
            score = self.score_citation(citation, citing_authors)
            scored_citations.append((citation, score))
            
        # Sort by total score
        scored_citations.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return scored_citations
        
    def get_field_from_venue(self, venue: str) -> FieldOfStudy:
        """Infer field of study from venue name."""
        venue_lower = venue.lower()
        
        # CS indicators
        if any(term in venue_lower for term in ["computer", "computing", "software", "acm", "ieee"]):
            return FieldOfStudy.COMPUTER_SCIENCE
            
        # Biology indicators  
        if any(term in venue_lower for term in ["biology", "biological", "cell", "genetics", "molecular"]):
            return FieldOfStudy.BIOLOGY
            
        # Medicine indicators
        if any(term in venue_lower for term in ["medicine", "medical", "clinical", "health", "lancet", "jama"]):
            return FieldOfStudy.MEDICINE
            
        # Physics indicators
        if any(term in venue_lower for term in ["physics", "physical review", "quantum"]):
            return FieldOfStudy.PHYSICS
            
        # Chemistry indicators
        if any(term in venue_lower for term in ["chemistry", "chemical"]):
            return FieldOfStudy.CHEMISTRY
            
        # Math indicators
        if any(term in venue_lower for term in ["mathematics", "mathematical", "annals of math"]):
            return FieldOfStudy.MATHEMATICS
            
        return FieldOfStudy.GENERAL