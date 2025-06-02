"""Tests for citation quality scoring."""

import pytest
from datetime import datetime
from research_assistant.citation_scorer import (
    CitationScorer, CitationScore, FieldOfStudy, VenueMetrics, AuthorMetrics
)


class TestCitationScorer:
    """Test cases for citation scoring."""
    
    def test_venue_scoring(self):
        """Test venue quality scoring."""
        scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
        
        # Top-tier CS venue
        paper1 = {
            "title": "Test Paper",
            "venue": "NeurIPS 2023",
            "year": 2023,
            "citation_count": 10
        }
        score1 = scorer.score_citation(paper1)
        assert score1.venue_score == 1.0
        
        # ArXiv paper
        paper2 = {
            "title": "Test Paper",
            "venue": "arXiv cs.LG",
            "year": 2023,
            "citation_count": 5
        }
        score2 = scorer.score_citation(paper2)
        assert 0.5 <= score2.venue_score <= 0.7
        
        # Unknown venue
        paper3 = {
            "title": "Test Paper",
            "venue": "Unknown Conference",
            "year": 2023,
            "citation_count": 2
        }
        score3 = scorer.score_citation(paper3)
        assert score3.venue_score < 0.6
    
    def test_impact_scoring(self):
        """Test citation impact scoring."""
        scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
        
        # Highly cited paper
        paper1 = {
            "title": "Influential Paper",
            "year": 2020,
            "citation_count": 100,
            "venue": "ICML"
        }
        score1 = scorer.score_citation(paper1, current_year=2024)
        assert score1.citation_velocity == 25.0  # 100 citations / 4 years
        assert score1.impact_score >= 0.9
        
        # Moderately cited paper
        paper2 = {
            "title": "Regular Paper",
            "year": 2021,
            "citation_count": 15,
            "venue": "Conference"
        }
        score2 = scorer.score_citation(paper2, current_year=2024)
        assert score2.citation_velocity == 5.0  # 15 citations / 3 years
        assert 0.6 <= score2.impact_score <= 0.8
        
        # Low citation paper
        paper3 = {
            "title": "New Paper",
            "year": 2023,
            "citation_count": 1,
            "venue": "Workshop"
        }
        score3 = scorer.score_citation(paper3, current_year=2024)
        assert score3.impact_score < 0.5
    
    def test_author_scoring(self):
        """Test author credibility scoring."""
        scorer = CitationScorer()
        
        # Paper with author metrics
        paper1 = {
            "title": "Test Paper",
            "authors": ["Author A", "Author B"],
            "year": 2023,
            "author_metrics": [
                AuthorMetrics(name="Author A", h_index=50),
                AuthorMetrics(name="Author B", h_index=30)
            ]
        }
        score1 = scorer.score_citation(paper1)
        assert score1.author_score > 0.5
        
        # Paper without metrics
        paper2 = {
            "title": "Test Paper",
            "authors": ["Unknown Author"],
            "year": 2023
        }
        score2 = scorer.score_citation(paper2)
        assert 0.4 <= score2.author_score <= 0.7
    
    def test_recency_scoring(self):
        """Test recency scoring with field-specific preferences."""
        current_year = 2024
        
        # CS paper (values recency)
        cs_scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
        
        new_paper = {"title": "New CS Paper", "year": 2023}
        old_paper = {"title": "Old CS Paper", "year": 2010}
        
        new_score = cs_scorer.score_citation(new_paper, current_year=current_year)
        old_score = cs_scorer.score_citation(old_paper, current_year=current_year)
        
        assert new_score.recency_score > old_score.recency_score
        assert new_score.recency_score >= 0.9
        assert old_score.recency_score <= 0.5
        
        # Math paper (values timelessness)
        math_scorer = CitationScorer(field=FieldOfStudy.MATHEMATICS)
        
        old_math_paper = {"title": "Classic Math Theorem", "year": 1990}
        old_math_score = math_scorer.score_citation(old_math_paper, current_year=current_year)
        
        assert old_math_score.recency_score >= 0.7  # Still valued despite age
    
    def test_consensus_scoring(self):
        """Test database consensus scoring."""
        scorer = CitationScorer()
        
        # Paper in multiple databases
        paper1 = {
            "title": "Well-verified Paper",
            "year": 2022,
            "database_appearances": 3
        }
        score1 = scorer.score_citation(paper1)
        assert score1.consensus_score == 1.0
        
        # Paper in single database
        paper2 = {
            "title": "Single Source Paper",
            "year": 2022,
            "database_appearances": 1
        }
        score2 = scorer.score_citation(paper2)
        assert score2.consensus_score < 0.5
    
    def test_self_citation_detection(self):
        """Test self-citation detection and penalties."""
        scorer = CitationScorer()
        
        paper = {
            "title": "Test Paper",
            "authors": ["Smith, John", "Doe, Jane", "Johnson, Bob"],
            "year": 2022,
            "citation_count": 50
        }
        
        # High self-citation
        citing_authors1 = ["Smith, John", "Doe, Jane", "Other, Author"]
        score1 = scorer.score_citation(paper, citing_authors=citing_authors1)
        assert score1.self_citation_ratio > 0.5
        assert "self_citation" in score1.penalties
        assert "self-citation" in score1.explanation.lower()
        
        # No self-citation
        citing_authors2 = ["Different, Author", "Another, Person"]
        score2 = scorer.score_citation(paper, citing_authors=citing_authors2)
        assert score2.self_citation_ratio == 0.0
        assert "self_citation" not in score2.penalties
    
    def test_penalty_application(self):
        """Test various penalties."""
        scorer = CitationScorer()
        
        # Retracted paper
        retracted_paper = {
            "title": "Retracted Paper",
            "year": 2020,
            "is_retracted": True,
            "citation_count": 100,
            "venue": "Nature"
        }
        score = scorer.score_citation(retracted_paper)
        assert "retracted" in score.penalties
        assert score.total_score < 20  # Heavily penalized
        assert "retracted" in score.explanation.lower()
        
        # Predatory venue
        predatory_paper = {
            "title": "Questionable Paper",
            "year": 2023,
            "venue": "Predatory Journal",
            "venue_metrics": VenueMetrics(
                name="Predatory Journal",
                is_predatory=True
            )
        }
        score = scorer.score_citation(predatory_paper)
        assert score.venue_score == 0.0
        assert "predatory" in score.explanation.lower()
    
    def test_total_score_calculation(self):
        """Test total score calculation with weights."""
        scorer = CitationScorer(
            field=FieldOfStudy.COMPUTER_SCIENCE,
            custom_weights={
                "venue": 0.3,
                "impact": 0.3,
                "author": 0.2,
                "recency": 0.1,
                "consensus": 0.1
            }
        )
        
        paper = {
            "title": "Good Paper",
            "venue": "NeurIPS",
            "year": 2023,
            "citation_count": 50,
            "authors": ["Famous, Author"],
            "database_appearances": 2
        }
        
        score = scorer.score_citation(paper, current_year=2024)
        
        # Should have high scores in most categories
        assert score.venue_score >= 0.9
        assert score.impact_score >= 0.8
        assert score.total_score >= 70
        
        # Check explanation
        assert len(score.explanation) > 0
        assert "top-tier" in score.explanation.lower()
    
    def test_citation_comparison(self):
        """Test comparing multiple citations."""
        scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
        
        citations = [
            {
                "title": "Highly Cited Paper",
                "venue": "ICML",
                "year": 2022,
                "citation_count": 100,
                "database_appearances": 3
            },
            {
                "title": "Recent Paper",
                "venue": "NeurIPS",
                "year": 2024,
                "citation_count": 5,
                "database_appearances": 2
            },
            {
                "title": "Old Classic",
                "venue": "Journal of ML",
                "year": 2010,
                "citation_count": 500,
                "database_appearances": 1
            }
        ]
        
        ranked = scorer.compare_citations(citations)
        
        # Should return sorted list
        assert len(ranked) == 3
        assert ranked[0][1].total_score >= ranked[1][1].total_score
        assert ranked[1][1].total_score >= ranked[2][1].total_score
        
        # Top paper should be the highly cited one with good consensus
        assert "Highly Cited" in ranked[0][0]["title"]
    
    def test_field_specific_scoring(self):
        """Test that different fields score differently."""
        paper = {
            "title": "Test Paper",
            "venue": "Generic Conference",
            "year": 2020,
            "citation_count": 20,
            "authors": ["Author"] * 30  # Large author list
        }
        
        # CS scorer
        cs_scorer = CitationScorer(field=FieldOfStudy.COMPUTER_SCIENCE)
        cs_score = cs_scorer.score_citation(paper, current_year=2024)
        
        # Physics scorer (large collaborations common)
        physics_scorer = CitationScorer(field=FieldOfStudy.PHYSICS)
        physics_score = physics_scorer.score_citation(paper, current_year=2024)
        
        # Physics should score large author lists higher
        assert physics_score.author_score > cs_score.author_score
        
        # Check warnings
        assert any("large author list" in w for w in cs_score.warnings)
        assert not any("large author list" in w for w in physics_score.warnings)
    
    def test_edge_cases(self):
        """Test edge cases and missing data."""
        scorer = CitationScorer()
        
        # Empty paper info
        empty_paper = {}
        score = scorer.score_citation(empty_paper)
        assert score.total_score > 0  # Should still produce a score
        
        # Missing critical fields
        partial_paper = {
            "title": "Partial Paper"
            # Missing year, venue, authors, etc.
        }
        score = scorer.score_citation(partial_paper)
        assert score.total_score > 0
        assert len(score.warnings) > 0