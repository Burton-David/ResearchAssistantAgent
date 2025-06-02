"""
Research Assistant Agent - AI-powered paper collection and analysis.
"""

__version__ = "0.1.0"

from .arxiv_collector import ArxivCollector, ArxivPaper
from .semantic_scholar_collector import SemanticScholarCollector, SemanticScholarPaper
from .paper_analyzer import PaperAnalyzer, PaperAnalysis, AnalysisType
from .vector_store import FAISSVectorStore, Document, EmbeddingGenerator
from .rate_limiter import RateLimiter, MultiServiceRateLimiter, AdaptiveRateLimiter, UnifiedRateLimiter, APIType
from .text_chunker import PaperChunker, TextChunk
from .config import Config, EmbeddingConfig, ChunkingConfig, VectorStoreConfig, EmbeddingModel

__all__ = [
    "ArxivCollector",
    "ArxivPaper",
    "SemanticScholarCollector", 
    "SemanticScholarPaper",
    "PaperAnalyzer",
    "PaperAnalysis",
    "AnalysisType",
    "FAISSVectorStore",
    "Document",
    "EmbeddingGenerator",
    "RateLimiter",
    "MultiServiceRateLimiter",
    "AdaptiveRateLimiter",
    "UnifiedRateLimiter",
    "APIType",
    "PaperChunker",
    "TextChunk",
    "Config",
    "EmbeddingConfig",
    "ChunkingConfig", 
    "VectorStoreConfig",
    "EmbeddingModel",
]