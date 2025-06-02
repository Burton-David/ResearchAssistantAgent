"""
Configuration settings for Research Assistant.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models."""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"


@dataclass
class EmbeddingConfig:
    """Configuration for text embeddings."""
    model_type: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS
    model_name: str = "all-MiniLM-L6-v2"  # Default sentence transformer model
    dimension: int = 384  # Dimension for all-MiniLM-L6-v2
    openai_model: str = "text-embedding-3-small"  # OpenAI embedding model
    openai_dimension: int = 1536  # OpenAI embedding dimension
    batch_size: int = 32  # Batch size for encoding
    show_progress: bool = True
    
    @property
    def current_dimension(self) -> int:
        """Get dimension for the current model type."""
        if self.model_type == EmbeddingModel.OPENAI:
            return self.openai_dimension
        return self.dimension


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 1000  # Characters per chunk
    chunk_overlap: int = 200  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    
    # Paper-specific chunking
    separate_sections: bool = True  # Chunk by paper sections
    sections_to_chunk: list[str] = None  # Which sections to include
    
    def __post_init__(self):
        if self.sections_to_chunk is None:
            self.sections_to_chunk = [
                "abstract",
                "introduction", 
                "methodology",
                "methods",
                "results",
                "discussion",
                "conclusion"
            ]


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_path: Path = None
    index_type: str = "flat"  # FAISS index type
    metric: str = "cosine"  # Distance metric
    
    # IVF parameters
    nlist: int = 100
    nprobe: int = 10
    
    # HNSW parameters
    ef_construction: int = 200
    ef_search: int = 50
    
    # Persistence
    auto_save: bool = True
    save_interval: int = 100  # Save after N documents
    
    def __post_init__(self):
        if self.store_path is None:
            self.store_path = Path.home() / ".research_assistant" / "vector_store"


@dataclass
class Config:
    """Main configuration container."""
    embedding: EmbeddingConfig = None
    chunking: ChunkingConfig = None
    vector_store: VectorStoreConfig = None
    
    # API Keys - never store directly
    _openai_api_key: Optional[str] = None
    _semantic_scholar_api_key: Optional[str] = None
    
    def __post_init__(self):
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.vector_store is None:
            self.vector_store = VectorStoreConfig()
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from secure storage or environment."""
        if self._openai_api_key:
            return self._openai_api_key
            
        # Try environment variable first
        if key := os.getenv("OPENAI_API_KEY"):
            return key
            
        # Try secure storage
        try:
            from .security import get_secure_config
            return get_secure_config().get_api_key("openai")
        except Exception as e:
            logger.debug(f"Could not load from secure storage: {e}")
            return None
            
    @openai_api_key.setter
    def openai_api_key(self, value: Optional[str]):
        """Set OpenAI API key (in memory only)."""
        self._openai_api_key = value
        
    @property
    def semantic_scholar_api_key(self) -> Optional[str]:
        """Get Semantic Scholar API key from secure storage or environment."""
        if self._semantic_scholar_api_key:
            return self._semantic_scholar_api_key
            
        # Try environment variable first
        if key := os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
            return key
            
        # Try secure storage
        try:
            from .security import get_secure_config
            return get_secure_config().get_api_key("semantic_scholar")
        except Exception as e:
            logger.debug(f"Could not load from secure storage: {e}")
            return None
            
    @semantic_scholar_api_key.setter
    def semantic_scholar_api_key(self, value: Optional[str]):
        """Set Semantic Scholar API key (in memory only)."""
        self._semantic_scholar_api_key = value
        
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        config = cls()
        
        # Override with environment variables if set
        if os.getenv("EMBEDDING_MODEL"):
            model_type = os.getenv("EMBEDDING_MODEL").lower()
            if model_type in ["openai", "sentence-transformers"]:
                config.embedding.model_type = EmbeddingModel(model_type)
                
        if os.getenv("VECTOR_STORE_PATH"):
            config.vector_store.store_path = Path(os.getenv("VECTOR_STORE_PATH"))
            
        return config


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config