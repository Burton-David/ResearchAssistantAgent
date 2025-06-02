"""Tests for configuration management."""

import os
from pathlib import Path
import pytest
from research_assistant.config import Config, EmbeddingModel


class TestConfig:
    """Test cases for Config class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        
        # Test embedding defaults
        assert config.embedding.model_type == EmbeddingModel.SENTENCE_TRANSFORMERS
        assert config.embedding.model_name == "all-MiniLM-L6-v2"
        assert config.embedding.dimension == 384
        assert config.embedding.show_progress is True
        
        # Test vector store defaults
        assert config.vector_store.store_path == Path.home() / ".research_assistant" / "vector_store"
        assert config.vector_store.index_type == "flat"
        assert config.vector_store.metric == "cosine"
        
        # Test chunking defaults
        assert config.chunking.chunk_size == 1000
        assert config.chunking.chunk_overlap == 200
    
    def test_config_from_env(self, monkeypatch):
        """Test loading configuration from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("EMBEDDING_MODEL", "openai")
        monkeypatch.setenv("VECTOR_STORE_PATH", "/tmp/test_store")
        
        config = Config.from_env()
        
        assert config.openai_api_key == "test-key-123"
        assert config.embedding.model_type == EmbeddingModel.OPENAI
        assert config.vector_store.store_path == Path("/tmp/test_store")
    
    def test_embedding_dimension_mapping(self):
        """Test dimension configuration for different models."""
        config = Config()
        
        # Test Sentence Transformers default
        assert config.embedding.model_type == EmbeddingModel.SENTENCE_TRANSFORMERS
        assert config.embedding.dimension == 384
        assert config.embedding.current_dimension == 384
        
        # Test OpenAI dimension
        config.embedding.model_type = EmbeddingModel.OPENAI
        assert config.embedding.current_dimension == 1536  # Uses openai_dimension
    
    def test_invalid_enum_from_env(self, monkeypatch):
        """Test handling of invalid enum values from environment."""
        monkeypatch.setenv("EMBEDDING_MODEL", "invalid_type")
        
        config = Config.from_env()
        # Should fall back to default
        assert config.embedding.model_type == EmbeddingModel.SENTENCE_TRANSFORMERS