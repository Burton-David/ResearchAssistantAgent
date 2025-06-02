"""
Citation explanation system.

Provides detailed explanations for why specific citations were recommended,
including relevance analysis and quality justification.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text

from .claim_extractor import ExtractedClaim as Claim, ClaimType
from .citation_scorer import CitationScore, FieldOfStudy

logger = logging.getLogger(__name__)
console = Console()


class RelevanceReason(Enum):
    """Types of relevance between claim and citation."""
    EXACT_MATCH = "exact_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    METHODOLOGY_MATCH = "methodology_match"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    CONTRADICTING_EVIDENCE = "contradicting_evidence"
    BACKGROUND_CONTEXT = "background_context"
    RELATED_WORK = "related_work"


@dataclass
class CitationExplanation:
    """Detailed explanation for a citation recommendation."""
    claim: Claim
    paper_title: str
    paper_info: Dict[str, Any]
    relevance_score: float
    quality_score: CitationScore
    
    # Explanation components
    relevance_reasons: List[Tuple[RelevanceReason, str]]
    key_matches: List[str]
    confidence: float
    recommendation: str  # 'strong', 'moderate', 'weak'
    warnings: List[str]
    
    def __str__(self) -> str:
        """Generate human-readable explanation."""
        parts = [
            f"Citation: {self.paper_title}",
            f"For claim: {self.claim.text[:100]}...",
            f"Relevance score: {self.relevance_score:.2f}",
            f"Quality score: {self.quality_score.total_score:.1f}/100",
            f"Recommendation: {self.recommendation}",
        ]
        
        if self.relevance_reasons:
            parts.append("\nRelevance reasons:")
            for reason, explanation in self.relevance_reasons:
                parts.append(f"  - {reason.value}: {explanation}")
                
        if self.warnings:
            parts.append("\nWarnings:")
            for warning in self.warnings:
                parts.append(f"  ⚠ {warning}")
                
        return "\n".join(parts)


class CitationExplainer:
    """
    Explains why citations were recommended for specific claims.
    
    Provides detailed justification including:
    - Relevance analysis
    - Quality assessment
    - Confidence scoring
    - Potential issues or warnings
    """
    
    def __init__(self, field: FieldOfStudy = FieldOfStudy.GENERAL):
        self.field = field
        
    def explain_citation(
        self,
        claim: Claim,
        paper_info: Dict[str, Any],
        relevance_score: float,
        quality_score: CitationScore,
        matched_terms: Optional[List[str]] = None
    ) -> CitationExplanation:
        """
        Generate detailed explanation for a citation recommendation.
        
        Args:
            claim: The claim requiring citation
            paper_info: Paper metadata
            relevance_score: Relevance score from citation finder
            quality_score: Quality score from citation scorer
            matched_terms: Terms that matched between claim and paper
            
        Returns:
            Detailed explanation object
        """
        relevance_reasons = self._analyze_relevance(
            claim, paper_info, relevance_score, matched_terms
        )
        
        warnings = self._check_warnings(
            claim, paper_info, quality_score
        )
        
        recommendation = self._determine_recommendation(
            relevance_score, quality_score.total_score, warnings
        )
        
        confidence = self._calculate_confidence(
            relevance_score, quality_score.total_score, len(warnings)
        )
        
        return CitationExplanation(
            claim=claim,
            paper_title=paper_info.get('title', 'Unknown'),
            paper_info=paper_info,
            relevance_score=relevance_score,
            quality_score=quality_score,
            relevance_reasons=relevance_reasons,
            key_matches=matched_terms or [],
            confidence=confidence,
            recommendation=recommendation,
            warnings=warnings
        )
        
    def _analyze_relevance(
        self,
        claim: Claim,
        paper_info: Dict[str, Any],
        relevance_score: float,
        matched_terms: Optional[List[str]]
    ) -> List[Tuple[RelevanceReason, str]]:
        """Analyze why the paper is relevant to the claim."""
        reasons = []
        
        # Check for exact matches
        if matched_terms and len(matched_terms) > 3:
            reasons.append((
                RelevanceReason.EXACT_MATCH,
                f"Found {len(matched_terms)} matching key terms"
            ))
            
        # High semantic similarity
        if relevance_score > 0.8:
            reasons.append((
                RelevanceReason.SEMANTIC_SIMILARITY,
                "High semantic similarity between claim and paper abstract"
            ))
            
        # Methodology match for methodological claims
        if claim.claim_type == ClaimType.METHODOLOGICAL:
            if any(term in paper_info.get('title', '').lower() 
                   for term in ['method', 'approach', 'technique', 'algorithm']):
                reasons.append((
                    RelevanceReason.METHODOLOGY_MATCH,
                    "Paper describes relevant methodology"
                ))
                
        # Statistical evidence for statistical claims
        if claim.claim_type == ClaimType.STATISTICAL:
            if any(term in paper_info.get('abstract', '').lower()
                   for term in ['results', 'findings', 'data', 'analysis']):
                reasons.append((
                    RelevanceReason.SUPPORTING_EVIDENCE,
                    "Paper contains statistical evidence"
                ))
                
        # Background context for theoretical claims
        if claim.claim_type == ClaimType.THEORETICAL:
            if relevance_score > 0.6:
                reasons.append((
                    RelevanceReason.BACKGROUND_CONTEXT,
                    "Provides theoretical background"
                ))
                
        # If no specific reasons found but still relevant
        if not reasons and relevance_score > 0.5:
            reasons.append((
                RelevanceReason.RELATED_WORK,
                "Related work in the same domain"
            ))
            
        return reasons
        
    def _check_warnings(
        self,
        claim: Claim,
        paper_info: Dict[str, Any],
        quality_score: CitationScore
    ) -> List[str]:
        """Check for potential issues with the citation."""
        warnings = []
        
        # Check publication year
        paper_year = paper_info.get('year', 0)
        if paper_year and claim.claim_type == ClaimType.STATISTICAL:
            years_old = quality_score.years_since_publication
            if years_old > 5:
                warnings.append(
                    f"Statistical claim citing {years_old}-year-old data"
                )
                
        # Check venue quality
        if quality_score.venue_score < 0.5:
            warnings.append("Published in lower-tier venue")
            
        # Check citation count for established claims
        if claim.confidence > 0.8 and quality_score.citation_count < 10:
            warnings.append(
                "Low citation count for supporting well-established claim"
            )
            
        # Field mismatch
        paper_field = self._infer_field(paper_info)
        if paper_field != self.field and paper_field != FieldOfStudy.GENERAL:
            warnings.append(
                f"Paper from different field ({paper_field.value})"
            )
            
        # Add quality score warnings
        warnings.extend(quality_score.warnings)
        
        return warnings
        
    def _determine_recommendation(
        self,
        relevance_score: float,
        quality_score: float,
        warnings: List[str]
    ) -> str:
        """Determine recommendation strength."""
        combined_score = (relevance_score * 0.6 + quality_score / 100 * 0.4)
        
        if combined_score > 0.8 and len(warnings) == 0:
            return "strong"
        elif combined_score > 0.6 or (combined_score > 0.5 and len(warnings) <= 1):
            return "moderate"
        else:
            return "weak"
            
    def _calculate_confidence(
        self,
        relevance_score: float,
        quality_score: float,
        warning_count: int
    ) -> float:
        """Calculate confidence in the recommendation."""
        base_confidence = (relevance_score * 0.6 + quality_score / 100 * 0.4)
        
        # Reduce confidence for warnings
        confidence = base_confidence * (0.9 ** warning_count)
        
        return min(max(confidence, 0.0), 1.0)
        
    def _infer_field(self, paper_info: Dict[str, Any]) -> FieldOfStudy:
        """Infer field of study from paper metadata."""
        title = paper_info.get('title', '').lower()
        venue = paper_info.get('venue', '').lower()
        abstract = paper_info.get('abstract', '').lower()
        
        # Simple keyword-based inference
        if any(term in title + venue for term in ['comput', 'algorithm', 'software']):
            return FieldOfStudy.COMPUTER_SCIENCE
        elif any(term in title + venue for term in ['medic', 'clinic', 'patient']):
            return FieldOfStudy.MEDICINE
        elif any(term in title + venue for term in ['bio', 'cell', 'gene']):
            return FieldOfStudy.BIOLOGY
        elif any(term in title + venue for term in ['physic', 'quantum']):
            return FieldOfStudy.PHYSICS
        elif any(term in title + venue for term in ['chem', 'molecul']):
            return FieldOfStudy.CHEMISTRY
        elif any(term in title + venue for term in ['math', 'theorem', 'proof']):
            return FieldOfStudy.MATHEMATICS
            
        return FieldOfStudy.GENERAL
        
    def display_explanation(self, explanation: CitationExplanation):
        """Display explanation with rich formatting."""
        # Create main panel
        title_color = {
            'strong': 'green',
            'moderate': 'yellow',
            'weak': 'red'
        }.get(explanation.recommendation, 'white')
        
        panel = Panel(
            f"[bold]{explanation.paper_title}[/bold]",
            title=f"[{title_color}]{explanation.recommendation.upper()} RECOMMENDATION[/{title_color}]",
            border_style=title_color
        )
        console.print(panel)
        
        # Create details table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Relevance Score", f"{explanation.relevance_score:.2f}")
        table.add_row("Quality Score", f"{explanation.quality_score.total_score:.1f}/100")
        table.add_row("Confidence", f"{explanation.confidence:.2f}")
        table.add_row("Claim Type", explanation.claim.claim_type.value)
        
        console.print(table)
        
        # Show relevance reasons
        if explanation.relevance_reasons:
            console.print("\n[bold]Why this citation?[/bold]")
            reason_tree = Tree("Relevance Analysis")
            
            for reason, desc in explanation.relevance_reasons:
                icon = {
                    RelevanceReason.EXACT_MATCH: "🎯",
                    RelevanceReason.SEMANTIC_SIMILARITY: "🔗",
                    RelevanceReason.METHODOLOGY_MATCH: "🔧",
                    RelevanceReason.SUPPORTING_EVIDENCE: "📊",
                    RelevanceReason.BACKGROUND_CONTEXT: "📚",
                    RelevanceReason.RELATED_WORK: "🔍"
                }.get(reason, "•")
                
                reason_tree.add(f"{icon} {desc}")
                
            console.print(reason_tree)
            
        # Show matched terms
        if explanation.key_matches:
            console.print(f"\n[bold]Matched terms:[/bold] {', '.join(explanation.key_matches)}")
            
        # Show warnings
        if explanation.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in explanation.warnings:
                console.print(f"  ⚠ {warning}")
                
        # Show quality breakdown
        console.print(f"\n[bold]Quality Assessment:[/bold]")
        console.print(f"  Venue: {'⭐' * int(explanation.quality_score.venue_score * 5)}")
        console.print(f"  Impact: {'⭐' * int(explanation.quality_score.impact_score * 5)}")
        console.print(f"  Authors: {'⭐' * int(explanation.quality_score.author_score * 5)}")
        
        # Final recommendation
        rec_text = {
            'strong': "[green]Highly recommended - excellent match with high quality[/green]",
            'moderate': "[yellow]Recommended - good match, consider context[/yellow]",
            'weak': "[red]Use with caution - limited relevance or quality concerns[/red]"
        }.get(explanation.recommendation, "")
        
        console.print(f"\n[bold]Recommendation:[/bold] {rec_text}")
        
    def compare_citations(
        self,
        claim: Claim,
        citation_options: List[Tuple[Dict[str, Any], float, CitationScore]]
    ) -> List[CitationExplanation]:
        """
        Compare multiple citation options for a claim.
        
        Args:
            claim: The claim requiring citation
            citation_options: List of (paper_info, relevance_score, quality_score) tuples
            
        Returns:
            List of explanations sorted by recommendation strength
        """
        explanations = []
        
        for paper_info, relevance_score, quality_score in citation_options:
            explanation = self.explain_citation(
                claim, paper_info, relevance_score, quality_score
            )
            explanations.append(explanation)
            
        # Sort by confidence
        explanations.sort(key=lambda e: e.confidence, reverse=True)
        
        return explanations