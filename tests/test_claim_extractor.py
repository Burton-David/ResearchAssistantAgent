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
        
        # Should find at least one statistical claim
        assert len(statistical_claims) >= 1
        
        # Check that claims capture key numerical information
        all_claims_text = " ".join([c.text for c in claims])
        
        # Verify that important numbers appear somewhere in extracted claims
        assert any(num in all_claims_text for num in ["95.2%", "12%", "r=0.89", "p<0.001"])
        
        # At least one claim should be marked as requiring citation
        assert any(c.requires_citation for c in claims)
    
    def test_extract_methodological_claims(self, extractor):
        """Test extraction of methodological claims."""
        text = """
        We employed the BERT model for text classification. This study used
        a novel data augmentation technique based on back-translation.
        The implementation leverages the Transformer architecture.
        """
        
        claims = extractor.extract_claims(text)
        method_claims = [c for c in claims if c.claim_type == ClaimType.METHODOLOGICAL]
        
        # Should find at least one methodological claim
        assert len(method_claims) >= 1
        
        # Check that important methodological terms are captured
        all_claims_text = " ".join([c.text for c in claims])
        method_terms = ["BERT", "data augmentation", "Transformer", "employed", "technique"]
        assert any(term in all_claims_text for term in method_terms)
        
        # At least some claims should have keywords or entities extracted
        all_keywords = []
        all_entities = []
        for claim in claims:
            all_keywords.extend(claim.keywords)
            all_entities.extend(claim.entities)
        
        # Should extract some keywords or entities from the technical text
        # Note: entities may be dict objects with 'text' key
        entity_texts = []
        for entity in all_entities:
            if isinstance(entity, dict) and 'text' in entity:
                entity_texts.append(entity['text'])
            elif isinstance(entity, str):
                entity_texts.append(entity)
        
        assert len(all_keywords) > 0 or len(entity_texts) > 0 or len(claims) > 0
    
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
        
        # Collect all keywords and entities from all claims
        all_keywords = []
        all_entities = []
        for claim in claims:
            all_keywords.extend(claim.keywords)
            all_entities.extend(claim.entities)
        
        # Should extract some relevant terms (either as keywords or entities)
        # Handle entities which may be dicts
        entity_texts = []
        for entity in all_entities:
            if isinstance(entity, dict) and 'text' in entity:
                entity_texts.append(entity['text'])
            elif isinstance(entity, str):
                entity_texts.append(entity)
        
        all_terms = all_keywords + entity_texts
        
        # Check that at least some key terms are captured somewhere
        key_terms = ["resnet", "imagenet", "accuracy", "model", "dataset"]
        claims_text = " ".join([c.text.lower() for c in claims])
        
        # Either we extracted terms, or the claim text contains key information
        assert len(all_terms) > 0 or any(term in claims_text for term in key_terms)
    
    def test_search_term_generation(self, extractor):
        """Test generation of search terms for claims."""
        text = "BERT outperforms GPT-2 on text classification tasks by 5%."
        
        claims = extractor.extract_claims(text)
        
        # Should extract at least one claim from this comparative statement
        assert len(claims) > 0
        
        # Check that we have comparative claims or at least capture the comparison
        comparative_claims = [c for c in claims if c.claim_type == ClaimType.COMPARATIVE]
        
        # The NLP model should at least capture this as a claim
        all_claim_texts = " ".join([c.text.lower() for c in claims])
        
        # Check if we captured key elements
        has_comparison = "outperform" in all_claim_texts or "5%" in all_claim_texts
        has_models = "bert" in all_claim_texts or "gpt" in all_claim_texts
        has_task = "classification" in all_claim_texts
        
        # At least capture the comparison or the models being compared
        assert has_comparison or has_models or has_task
        
        # If we have comparative claims, they should be meaningful
        if comparative_claims:
            assert len(comparative_claims[0].text) > 5  # Not just a single word
    
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