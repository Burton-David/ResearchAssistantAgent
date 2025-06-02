"""Tests for claim extraction functionality."""

import pytest
from research_assistant.claim_extractor import (
    ClaimExtractor, ExtractedClaim, ClaimType
)


class TestClaimExtractor:
    """Test cases for claim extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create a claim extractor instance."""
        return ClaimExtractor()
    
    def test_extract_statistical_claims(self, extractor):
        """Test extraction of statistical claims."""
        text = """
        Our method achieved a 95.2% accuracy on the test set, which represents
        a 12% improvement over the baseline. The correlation coefficient was
        r=0.89 (p<0.001), indicating a strong positive relationship.
        """
        
        claims = extractor.extract_claims(text)
        statistical_claims = [c for c in claims if c.claim_type == ClaimType.STATISTICAL]
        
        assert len(statistical_claims) >= 3
        
        # Check for percentage claim
        percentage_claims = [c for c in statistical_claims if "95.2%" in c.text or "12%" in c.text]
        assert len(percentage_claims) >= 1
        
        # Check for p-value claim
        p_value_claims = [c for c in statistical_claims if "p<0.001" in c.text]
        assert len(p_value_claims) >= 1
        
        # Check for correlation claim
        correlation_claims = [c for c in statistical_claims if "r=0.89" in c.text or "correlation" in c.text.lower()]
        assert len(correlation_claims) >= 1
    
    def test_extract_methodological_claims(self, extractor):
        """Test extraction of methodological claims."""
        text = """
        We employed the BERT model for text classification. This study used
        a novel data augmentation technique based on back-translation.
        The implementation leverages the Transformer architecture.
        """
        
        claims = extractor.extract_claims(text)
        method_claims = [c for c in claims if c.claim_type == ClaimType.METHODOLOGICAL]
        
        assert len(method_claims) >= 2
        
        # Check for specific methods mentioned
        bert_claims = [c for c in method_claims if "BERT" in c.text or "employed" in c.text]
        assert len(bert_claims) >= 1
        
        # Check keywords extraction
        for claim in method_claims:
            assert len(claim.keywords) > 0
    
    def test_extract_comparative_claims(self, extractor):
        """Test extraction of comparative claims."""
        text = """
        Our approach outperforms the state-of-the-art baseline by a significant
        margin. The proposed method is more efficient than traditional approaches,
        achieving better results with less computational resources.
        """
        
        claims = extractor.extract_claims(text)
        comparative_claims = [c for c in claims if c.claim_type == ClaimType.COMPARATIVE]
        
        assert len(comparative_claims) >= 2
        
        # Check for outperforms claim
        outperform_claims = [c for c in comparative_claims if "outperforms" in c.text]
        assert len(outperform_claims) >= 1
        
        # Check for "better than" type claims
        better_claims = [c for c in comparative_claims if "more efficient" in c.text or "better" in c.text]
        assert len(better_claims) >= 1
    
    def test_extract_theoretical_claims(self, extractor):
        """Test extraction of theoretical claims."""
        text = """
        These results suggest that the mechanism behind the improvement is
        related to attention distribution. Our findings indicate that the
        model learns hierarchical representations, which supports the
        theory of emergent complexity in neural networks.
        """
        
        claims = extractor.extract_claims(text)
        theoretical_claims = [c for c in claims if c.claim_type == ClaimType.THEORETICAL]
        
        assert len(theoretical_claims) >= 2
        
        # Check for "suggest" claims
        suggest_claims = [c for c in theoretical_claims if "suggest" in c.text.lower()]
        assert len(suggest_claims) >= 1
        
        # Check for "indicate" claims
        indicate_claims = [c for c in theoretical_claims if "indicate" in c.text.lower()]
        assert len(indicate_claims) >= 1
    
    def test_extract_causal_claims(self, extractor):
        """Test extraction of causal claims."""
        text = """
        The improved performance is due to the multi-head attention mechanism.
        This causes the model to focus on relevant features. Consequently,
        the error rate decreases significantly. The effect of regularization
        leads to better generalization.
        """
        
        claims = extractor.extract_claims(text)
        causal_claims = [c for c in claims if c.claim_type == ClaimType.CAUSAL]
        
        assert len(causal_claims) >= 3
        
        # Check for different causal indicators
        due_to_claims = [c for c in causal_claims if "due to" in c.text]
        assert len(due_to_claims) >= 1
        
        leads_to_claims = [c for c in causal_claims if "leads to" in c.text]
        assert len(leads_to_claims) >= 1
    
    def test_claim_context_extraction(self, extractor):
        """Test that claims include proper context."""
        text = "Previous work showed that the accuracy was 85%. Our method achieves 95% accuracy. This is a significant improvement."
        
        claims = extractor.extract_claims(text, context_window=50)
        
        assert len(claims) > 0
        
        for claim in claims:
            # Check context is provided
            assert claim.context
            assert "[CLAIM]" in claim.context
            assert "[/CLAIM]" in claim.context
            
            # Check context includes surrounding text
            assert len(claim.context) > len(claim.text)
    
    def test_keyword_extraction(self, extractor):
        """Test keyword extraction from claims."""
        text = "The ResNet-50 model achieved 94.5% accuracy on ImageNet dataset."
        
        claims = extractor.extract_claims(text)
        
        assert len(claims) > 0
        
        claim = claims[0]
        assert len(claim.keywords) > 0
        
        # Should extract model name and dataset
        keywords_lower = [k.lower() for k in claim.keywords]
        assert any("resnet" in k for k in keywords_lower) or any("imagenet" in k for k in keywords_lower)
    
    def test_search_term_generation(self, extractor):
        """Test generation of search terms for claims."""
        text = "BERT outperforms GPT-2 on text classification tasks by 5%."
        
        claims = extractor.extract_claims(text)
        comparative_claims = [c for c in claims if c.claim_type == ClaimType.COMPARATIVE]
        
        assert len(comparative_claims) > 0
        
        claim = comparative_claims[0]
        assert len(claim.suggested_search_terms) > 0
        
        # Should include model names
        search_terms_lower = [t.lower() for t in claim.suggested_search_terms]
        assert any("bert" in t for t in search_terms_lower) or any("gpt" in t for t in search_terms_lower)
    
    def test_deduplication(self, extractor):
        """Test that overlapping claims are deduplicated."""
        text = "The accuracy improved by 10%, a 10% improvement over baseline."
        
        claims = extractor.extract_claims(text)
        
        # Should not have duplicate claims for the same percentage
        percentages = [c.text for c in claims if "10%" in c.text]
        
        # Check no exact duplicates
        assert len(percentages) == len(set(percentages))
        
        # Check no overlapping claims
        for i, claim1 in enumerate(claims):
            for claim2 in claims[i+1:]:
                # Claims should not overlap
                assert not (claim1.start_char < claim2.end_char and claim2.start_char < claim1.end_char)
    
    def test_confidence_filtering(self, extractor):
        """Test that low confidence claims can be filtered."""
        text = "The results show improvement in performance metrics."
        
        # Get all claims
        all_claims = extractor.extract_claims(text, min_confidence=0.0)
        
        # Get high confidence claims only
        high_conf_claims = extractor.extract_claims(text, min_confidence=0.8)
        
        # Should have filtered some claims
        assert len(high_conf_claims) <= len(all_claims)
    
    def test_analyze_citation_needs(self, extractor):
        """Test citation needs analysis."""
        text = """
        Our method achieved 95% accuracy, outperforming the baseline by 10%.
        This suggests that the attention mechanism is crucial. We used BERT
        for encoding, which caused significant improvements.
        """
        
        claims = extractor.extract_claims(text)
        analysis = extractor.analyze_citation_needs(claims)
        
        assert analysis["total_claims"] == len(claims)
        assert len(analysis["by_type"]) > 0
        assert "high_priority" in analysis
        assert analysis["suggested_citations_needed"] >= 0
        
        # Check type breakdown
        assert sum(analysis["by_type"].values()) == analysis["total_claims"]
    
    def test_empty_text(self, extractor):
        """Test handling of empty text."""
        claims = extractor.extract_claims("")
        
        assert claims == []
        
        analysis = extractor.analyze_citation_needs(claims)
        assert analysis["total_claims"] == 0
    
    def test_no_claims_text(self, extractor):
        """Test text with no identifiable claims."""
        text = "This is a simple sentence with no specific claims."
        
        claims = extractor.extract_claims(text)
        
        # May or may not find claims depending on patterns
        # Just ensure it doesn't crash
        assert isinstance(claims, list)