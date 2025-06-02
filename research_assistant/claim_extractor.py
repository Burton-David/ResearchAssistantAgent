"""
Intelligent claim extraction from research papers.

Identifies and categorizes different types of claims that require citations:
- Statistical claims (percentages, increases, correlations)
- Methodological claims (techniques, approaches, algorithms)
- Comparative claims (outperforms, better than, superior to)
- Theoretical claims (suggests, indicates, implies mechanisms)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import spacy
from spacy.matcher import Matcher

logger = logging.getLogger(__name__)


class ClaimType(Enum):
    """Types of claims that typically require citations."""
    STATISTICAL = "statistical"
    METHODOLOGICAL = "methodological"
    COMPARATIVE = "comparative"
    THEORETICAL = "theoretical"
    FACTUAL = "factual"
    CAUSAL = "causal"
    EVALUATIVE = "evaluative"


@dataclass
class ExtractedClaim:
    """Represents an extracted claim from text."""
    text: str
    claim_type: ClaimType
    confidence: float
    context: str  # Surrounding text for context
    start_char: int
    end_char: int
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict[str, str]] = field(default_factory=list)
    requires_citation: bool = True
    suggested_search_terms: List[str] = field(default_factory=list)


class ClaimExtractor:
    """
    Extracts claims from research papers using NLP and pattern matching.
    
    Uses spaCy for linguistic analysis and custom patterns for claim detection.
    """
    
    # Statistical claim patterns
    STATISTICAL_PATTERNS = [
        # Percentage patterns
        r'\b\d+(?:\.\d+)?%\s+(?:increase|decrease|improvement|reduction|growth|decline)',
        r'(?:increased|decreased|improved|reduced|grew|declined)\s+by\s+\d+(?:\.\d+)?%',
        
        # Correlation patterns
        r'(?:correlation|correlates?)\s+(?:of|between|with)\s+[rR]?\s*=?\s*[-+]?\d*\.?\d+',
        r'(?:significant|strong|weak|moderate)\s+correlation',
        
        # Statistical significance
        r'[pP]\s*[<>=]\s*0?\.\d+',
        r'statistically\s+significant',
        r'confidence\s+interval',
        
        # Effect sizes
        r'effect\s+size\s+(?:of|=)\s*\d*\.?\d+',
        r'Cohen\'s?\s+d\s*=\s*\d*\.?\d+',
        
        # Sample sizes
        r'[nN]\s*=\s*\d+',
        r'sample\s+(?:size|of)\s+\d+',
    ]
    
    # Methodological claim patterns
    METHODOLOGICAL_PATTERNS = [
        r'(?:we|this\s+study)\s+(?:used?|employ(?:ed)?|implement(?:ed)?|applied?|develop(?:ed)?)',
        r'(?:using|employing|implementing|applying)\s+(?:the)?\s*[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+(?:method|technique|approach|algorithm)',
        r'(?:novel|new|proposed|modified)\s+(?:method|technique|approach|algorithm)',
        r'based\s+on\s+(?:the)?\s*[A-Z][A-Za-z]+',
    ]
    
    # Comparative claim patterns
    COMPARATIVE_PATTERNS = [
        r'(?:outperform(?:s|ed)?|better\s+than|superior\s+to|exceed(?:s|ed)?)',
        r'(?:compared\s+to|in\s+comparison\s+with|versus|vs\.?)\s+[A-Za-z]+',
        r'(?:more|less)\s+(?:accurate|efficient|effective|robust)\s+than',
        r'(?:highest|lowest|best|worst)\s+(?:performance|accuracy|results?)',
        r'(?:state-of-the-art|SOTA|baseline)\s+(?:performance|results?)',
    ]
    
    # Theoretical claim patterns
    THEORETICAL_PATTERNS = [
        r'(?:suggest(?:s|ed)?|indicate(?:s|d)?|imply?|implies|demonstrate(?:s|d)?)\s+that',
        r'(?:evidence|results?)\s+(?:for|of|supporting)',
        r'(?:consistent|inconsistent)\s+with\s+(?:the)?\s*[A-Za-z]+\s+(?:theory|hypothesis|model)',
        r'(?:support(?:s|ed)?|contradict(?:s|ed)?|confirm(?:s|ed)?)\s+(?:the)?\s*(?:hypothesis|theory)',
        r'(?:mechanism|process|phenomenon)\s+(?:of|for|behind)',
    ]
    
    # Causal claim patterns
    CAUSAL_PATTERNS = [
        r'(?:caus(?:es?|ed|ing)|leads?\s+to|results?\s+in|due\s+to|because\s+of)',
        r'(?:effect|impact|influence)\s+(?:of|on)',
        r'(?:consequently|therefore|thus|hence)\s+[A-Za-z]+',
        r'(?:induces?|triggers?|promotes?|inhibits?)',
    ]
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize claim extractor with spaCy model.
        
        Args:
            model_name: Name of spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"spaCy model {model_name} not found. Installing...")
            import subprocess
            subprocess.check_call(["python", "-m", "spacy", "download", model_name])
            self.nlp = spacy.load(model_name)
            
        self.matcher = Matcher(self.nlp.vocab)
        self._init_patterns()
        
    def _init_patterns(self):
        """Initialize spaCy matcher patterns for claim detection."""
        # Add patterns for different claim types
        
        # Statistical claims with numbers
        self.matcher.add("STATISTICAL_CLAIM", [
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["percent", "%", "increase", "decrease"]}}],
            [{"LOWER": "p"}, {"TEXT": {"REGEX": "[<>=]"}}, {"LIKE_NUM": True}],
            [{"LOWER": {"IN": ["correlation", "effect"]}}, {"LOWER": "size"}, {"LIKE_NUM": True}],
        ])
        
        # Comparative claims
        self.matcher.add("COMPARATIVE_CLAIM", [
            [{"LOWER": {"IN": ["outperforms", "outperformed", "exceeds", "exceeded"]}},
             {"POS": {"IN": ["NOUN", "PROPN"]}}],
            [{"LOWER": {"IN": ["better", "worse", "superior", "inferior"]}}, 
             {"LOWER": "than"}],
            [{"LOWER": {"IN": ["more", "less"]}},
             {"POS": "ADJ"},
             {"LOWER": "than"}],
        ])
        
        # Methodological claims
        self.matcher.add("METHODOLOGICAL_CLAIM", [
            [{"LOWER": {"IN": ["we", "this"]}},
             {"LOWER": {"IN": ["used", "use", "employed", "employ", "implemented", "implement"]}},
             {"POS": {"IN": ["NOUN", "PROPN"]}}],
            [{"LOWER": {"IN": ["using", "employing", "implementing"]}},
             {"POS": "DET", "OP": "?"},
             {"POS": "PROPN"}],
        ])
        
    def extract_claims(
        self,
        text: str,
        context_window: int = 100,
        min_confidence: float = 0.5
    ) -> List[ExtractedClaim]:
        """
        Extract claims from text.
        
        Args:
            text: Input text to analyze
            context_window: Characters of context to include
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of extracted claims
        """
        claims = []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract claims using regex patterns
        claims.extend(self._extract_statistical_claims(text, doc, context_window))
        claims.extend(self._extract_methodological_claims(text, doc, context_window))
        claims.extend(self._extract_comparative_claims(text, doc, context_window))
        claims.extend(self._extract_theoretical_claims(text, doc, context_window))
        claims.extend(self._extract_causal_claims(text, doc, context_window))
        
        # Extract claims using spaCy matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            claim_type = self._get_claim_type_from_match(self.nlp.vocab.strings[match_id])
            
            claim = ExtractedClaim(
                text=span.text,
                claim_type=claim_type,
                confidence=0.8,
                context=self._get_context(text, span.start_char, span.end_char, context_window),
                start_char=span.start_char,
                end_char=span.end_char,
                keywords=self._extract_keywords(span),
                entities=self._extract_entities(span)
            )
            
            if claim.confidence >= min_confidence:
                claims.append(claim)
                
        # Remove duplicates and overlapping claims
        claims = self._deduplicate_claims(claims)
        
        # Generate search terms for each claim
        for claim in claims:
            claim.suggested_search_terms = self._generate_search_terms(claim, doc)
            
        return claims
        
    def _extract_statistical_claims(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        context_window: int
    ) -> List[ExtractedClaim]:
        """Extract statistical claims using regex patterns."""
        claims = []
        
        for pattern in self.STATISTICAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Get surrounding sentence
                sent_start, sent_end = self._get_sentence_bounds(doc, match.start())
                sentence_text = text[sent_start:sent_end]
                
                claim = ExtractedClaim(
                    text=match.group(),
                    claim_type=ClaimType.STATISTICAL,
                    confidence=0.9,
                    context=self._get_context(text, match.start(), match.end(), context_window),
                    start_char=match.start(),
                    end_char=match.end(),
                    keywords=self._extract_statistical_keywords(match.group())
                )
                claims.append(claim)
                
        return claims
        
    def _extract_methodological_claims(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        context_window: int
    ) -> List[ExtractedClaim]:
        """Extract methodological claims."""
        claims = []
        
        for pattern in self.METHODOLOGICAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = ExtractedClaim(
                    text=match.group(),
                    claim_type=ClaimType.METHODOLOGICAL,
                    confidence=0.85,
                    context=self._get_context(text, match.start(), match.end(), context_window),
                    start_char=match.start(),
                    end_char=match.end()
                )
                claims.append(claim)
                
        return claims
        
    def _extract_comparative_claims(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        context_window: int
    ) -> List[ExtractedClaim]:
        """Extract comparative claims."""
        claims = []
        
        for pattern in self.COMPARATIVE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = ExtractedClaim(
                    text=match.group(),
                    claim_type=ClaimType.COMPARATIVE,
                    confidence=0.85,
                    context=self._get_context(text, match.start(), match.end(), context_window),
                    start_char=match.start(),
                    end_char=match.end()
                )
                claims.append(claim)
                
        return claims
        
    def _extract_theoretical_claims(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        context_window: int
    ) -> List[ExtractedClaim]:
        """Extract theoretical claims."""
        claims = []
        
        for pattern in self.THEORETICAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = ExtractedClaim(
                    text=match.group(),
                    claim_type=ClaimType.THEORETICAL,
                    confidence=0.8,
                    context=self._get_context(text, match.start(), match.end(), context_window),
                    start_char=match.start(),
                    end_char=match.end()
                )
                claims.append(claim)
                
        return claims
        
    def _extract_causal_claims(
        self,
        text: str,
        doc: spacy.tokens.Doc,
        context_window: int
    ) -> List[ExtractedClaim]:
        """Extract causal claims."""
        claims = []
        
        for pattern in self.CAUSAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claim = ExtractedClaim(
                    text=match.group(),
                    claim_type=ClaimType.CAUSAL,
                    confidence=0.8,
                    context=self._get_context(text, match.start(), match.end(), context_window),
                    start_char=match.start(),
                    end_char=match.end()
                )
                claims.append(claim)
                
        return claims
        
    def _get_context(self, text: str, start: int, end: int, window: int) -> str:
        """Get surrounding context for a claim."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # Mark the claim within context
        claim_start = start - context_start
        claim_end = end - context_start
        
        return (
            context[:claim_start] + 
            "[CLAIM]" + 
            context[claim_start:claim_end] + 
            "[/CLAIM]" + 
            context[claim_end:]
        )
        
    def _get_sentence_bounds(self, doc: spacy.tokens.Doc, char_pos: int) -> Tuple[int, int]:
        """Get sentence boundaries for a character position."""
        for sent in doc.sents:
            if sent.start_char <= char_pos < sent.end_char:
                return sent.start_char, sent.end_char
        return char_pos, char_pos + 1
        
    def _extract_keywords(self, span: spacy.tokens.Span) -> List[str]:
        """Extract important keywords from a span."""
        keywords = []
        
        for token in span:
            # Include nouns, proper nouns, and numbers
            if token.pos_ in ["NOUN", "PROPN", "NUM"] and not token.is_stop:
                keywords.append(token.text)
                
        return list(set(keywords))
        
    def _extract_entities(self, span: spacy.tokens.Span) -> List[Dict[str, str]]:
        """Extract named entities from a span."""
        entities = []
        
        for ent in span.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        return entities
        
    def _extract_statistical_keywords(self, text: str) -> List[str]:
        """Extract statistical keywords from text."""
        keywords = []
        
        # Extract numbers and percentages
        numbers = re.findall(r'\d+(?:\.\d+)?%?', text)
        keywords.extend(numbers)
        
        # Extract statistical terms
        stat_terms = re.findall(
            r'(?:correlation|significance|confidence|effect\s+size|sample\s+size|p-value)',
            text,
            re.IGNORECASE
        )
        keywords.extend(stat_terms)
        
        return list(set(keywords))
        
    def _get_claim_type_from_match(self, match_label: str) -> ClaimType:
        """Map spaCy match label to claim type."""
        mapping = {
            "STATISTICAL_CLAIM": ClaimType.STATISTICAL,
            "COMPARATIVE_CLAIM": ClaimType.COMPARATIVE,
            "METHODOLOGICAL_CLAIM": ClaimType.METHODOLOGICAL,
        }
        return mapping.get(match_label, ClaimType.FACTUAL)
        
    def _deduplicate_claims(self, claims: List[ExtractedClaim]) -> List[ExtractedClaim]:
        """Remove duplicate and overlapping claims."""
        if not claims:
            return claims
            
        # Sort by start position
        sorted_claims = sorted(claims, key=lambda c: (c.start_char, -c.confidence))
        
        deduped = []
        last_end = -1
        
        for claim in sorted_claims:
            # Skip if overlapping with previous claim
            if claim.start_char < last_end:
                continue
                
            deduped.append(claim)
            last_end = claim.end_char
            
        return deduped
        
    def _generate_search_terms(
        self,
        claim: ExtractedClaim,
        doc: spacy.tokens.Doc
    ) -> List[str]:
        """Generate search terms for finding citations for a claim."""
        search_terms = []
        
        # Add keywords from the claim
        search_terms.extend(claim.keywords)
        
        # Add entity names
        for entity in claim.entities:
            search_terms.append(entity["text"])
            
        # Add claim-type specific terms
        if claim.claim_type == ClaimType.STATISTICAL:
            # For statistical claims, include the metric
            metric_patterns = [
                r'(?:accuracy|precision|recall|f1[-\s]?score|performance|error\s+rate)',
                r'(?:correlation|regression|significance)',
            ]
            for pattern in metric_patterns:
                matches = re.findall(pattern, claim.context, re.IGNORECASE)
                search_terms.extend(matches)
                
        elif claim.claim_type == ClaimType.METHODOLOGICAL:
            # Extract method names (usually capitalized)
            method_names = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', claim.context)
            search_terms.extend(method_names)
            
        elif claim.claim_type == ClaimType.COMPARATIVE:
            # Extract what's being compared
            comparison_targets = re.findall(
                r'(?:compared\s+to|versus|vs\.?)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)',
                claim.context,
                re.IGNORECASE
            )
            search_terms.extend(comparison_targets)
            
        # Remove duplicates and clean
        search_terms = list(set(term.strip() for term in search_terms if term.strip()))
        
        # Limit to most relevant terms
        return search_terms[:5]
        
    def analyze_citation_needs(
        self,
        claims: List[ExtractedClaim]
    ) -> Dict[str, Any]:
        """
        Analyze citation needs for extracted claims.
        
        Returns statistics and recommendations for citation placement.
        """
        analysis = {
            "total_claims": len(claims),
            "by_type": {},
            "high_priority": [],
            "suggested_citations_needed": 0
        }
        
        # Count by type
        for claim_type in ClaimType:
            type_claims = [c for c in claims if c.claim_type == claim_type]
            analysis["by_type"][claim_type.value] = len(type_claims)
            
        # Identify high priority claims (high confidence, requires citation)
        high_priority = [
            c for c in claims 
            if c.confidence >= 0.8 and c.requires_citation
        ]
        analysis["high_priority"] = high_priority
        analysis["suggested_citations_needed"] = len(high_priority)
        
        return analysis