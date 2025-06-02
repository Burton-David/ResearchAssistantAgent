"""Tests for vector store functionality."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from research_assistant.vector_store import FAISSVectorStore, EmbeddingGenerator
from research_assistant.config import Config, EmbeddingModel


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""
    
    def test_sentence_transformer_embeddings(self):
        """Test generating embeddings with sentence transformers."""
        config = Config()
        config.embedding.model_type = EmbeddingModel.SENTENCE_TRANSFORMERS
        generator = EmbeddingGenerator(config)
        
        texts = ["Hello world", "Machine learning is awesome"]
        embeddings = generator.generate_embeddings(texts)
        
        assert len(embeddings) == 2
        assert embeddings[0].shape == (384,)
        assert embeddings[1].shape == (384,)
        assert not np.array_equal(embeddings[0], embeddings[1])
    
    def test_single_text_embedding(self):
        """Test generating embedding for single text."""
        config = Config()
        generator = EmbeddingGenerator(config)
        
        embeddings = generator.generate_embeddings(["Test text"])
        embedding = embeddings[0]
        
        assert embedding.shape == (384,)
        assert isinstance(embedding, np.ndarray)


class TestFAISSVectorStore:
    """Test cases for FAISSVectorStore."""
    
    def test_add_and_search(self):
        """Test adding papers and searching."""
        config = Config()
        store = FAISSVectorStore(config=config)
        
        # Add papers
        store.add_paper(
            paper_id="paper1",
            paper_text="Deep learning for natural language processing",
            paper_metadata={"title": "NLP with DL"},
            generate_embedding=True
        )
        
        store.add_paper(
            paper_id="paper2", 
            paper_text="Computer vision using convolutional neural networks",
            paper_metadata={"title": "CV with CNN"},
            generate_embedding=True
        )
        
        # Search
        results = store.search("language processing", k=1)
        
        assert len(results) == 1
        assert results[0][0].metadata["title"] == "NLP with DL"
        assert results[0][1] > 0.3  # Similarity score (lower threshold for sentence transformers)
    
    def test_persistence(self):
        """Test saving and loading vector store."""
        config = Config()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_store"
            
            # Create and save store
            store1 = FAISSVectorStore(config=config)
            store1.add_paper(
                paper_id="test1",
                paper_text="Quantum computing fundamentals",
                paper_metadata={"field": "quantum"},
                generate_embedding=True
            )
            store1.save(save_path)
            
            # Load store
            store2 = FAISSVectorStore.load(save_path, config=config)
            
            # Verify data persisted
            results = store2.search("quantum", k=1)
            assert len(results) >= 1  # May have been chunked
            if results:
                assert "quantum" in results[0][0].metadata.get("field", "")
    
    def test_chunking(self):
        """Test paper chunking functionality."""
        config = Config()
        store = FAISSVectorStore(config=config)
        
        long_text = """
        # Abstract
        This is the abstract.
        
        ## Introduction
        This is a very long introduction section that contains many words and ideas.
        """ + " ".join(["word"] * 500)
        
        store.add_paper(
            paper_id="chunked_paper",
            paper_text=long_text,
            paper_metadata={"title": "Long Paper"},
            chunk_paper=True,
            generate_embedding=True
        )
        
        # Should have created multiple chunks
        assert len(store.documents) > 1
        assert all("chunk_" in doc.metadata.get("chunk_id", "") 
                  for doc in store.documents.values()
                  if doc.metadata.get("paper_id") == "chunked_paper")
    
    def test_empty_store_search(self):
        """Test searching in empty store."""
        config = Config()
        store = FAISSVectorStore(config=config)
        
        results = store.search("test query", k=5)
        assert len(results) == 0
    
    def test_get_statistics(self):
        """Test getting store statistics."""
        config = Config()
        store = FAISSVectorStore(config=config)
        
        # Add some papers
        for i in range(3):
            store.add_paper(
                paper_id=f"paper_{i}",
                paper_text=f"Paper {i} content about machine learning",
                paper_metadata={"index": i},
                generate_embedding=True
            )
        
        stats = store.get_statistics()
        
        assert stats["total_documents"] == 3
        # No total_papers in stats, check total_chunks instead
        assert stats["total_chunks"] == 3
        assert stats["dimension"] == 384
        assert stats["index_type"] == "flat"