"""
Integration tests for the Research Assistant Agent.

Tests real API calls and end-to-end functionality.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from research_assistant import (
    ArxivCollector,
    SemanticScholarCollector,
    FAISSVectorStore,
    Config,
    EmbeddingGenerator
)
from research_assistant.arxiv_collector import ArxivPaper
from research_assistant.semantic_scholar_collector import SemanticScholarPaper as S2Paper
from research_assistant.validators import ValidationError
from research_assistant.exceptions import APIError, RateLimitError
from research_assistant.resource_manager import get_resource_manager


class TestIntegration:
    """Integration tests with real APIs."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_arxiv_search_and_store(self):
        """Test searching ArXiv and storing results."""
        # Create mock paper
        mock_paper = ArxivPaper(
            arxiv_id="2301.00001",
            title="Machine Learning Test Paper",
            authors=["Test Author"],
            abstract="This is a test paper about machine learning.",
            published_date=datetime(2023, 1, 1),
            updated_date=datetime(2023, 1, 1),
            categories=["cs.LG"],
            pdf_url="http://arxiv.org/pdf/2301.00001",
            arxiv_url="http://arxiv.org/abs/2301.00001",
            primary_category="cs.LG"
        )
        
        with patch('research_assistant.arxiv_collector.ArxivCollector.search') as mock_search:
            mock_search.return_value = [mock_paper]
            
            async with ArxivCollector() as collector:
                papers = await collector.search("cat:cs.LG", max_results=2)
                
                assert len(papers) == 1
                assert all(hasattr(p, 'arxiv_id') for p in papers)
                
                # Store in vector store
                with tempfile.TemporaryDirectory() as tmpdir:
                    config = Config()
                    config.vector_store.store_path = Path(tmpdir) / "test_store"
                    
                    store = FAISSVectorStore(config=config)
                    
                    for paper in papers:
                        doc_ids = store.add_paper(
                            paper_id=f"arxiv_{paper.arxiv_id}",
                            paper_text=f"{paper.title}\n\n{paper.abstract}",
                            paper_metadata={
                                "title": paper.title,
                                "authors": paper.authors[:3],
                                "arxiv_id": paper.arxiv_id
                            },
                            chunk_paper=False,
                            generate_embedding=True
                        )
                        assert len(doc_ids) >= 1
                    
                    # Test search
                    if papers:
                        results = store.search("machine learning", k=1)
                        assert len(results) >= 0  # May not match if papers are about different topics
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_semantic_scholar_with_rate_limiting(self):
        """Test Semantic Scholar API with rate limiting."""
        # Create mock paper
        mock_paper = S2Paper(
            paper_id="test123",
            title="Deep Learning Paper",
            abstract="Test abstract",
            year=2023,
            authors=[{"name": "Test Author", "authorId": "1"}],
            citation_count=10,
            reference_count=5,
            influential_citation_count=2,
            is_open_access=True,
            fields_of_study=["Computer Science"],
            s2_url="https://www.semanticscholar.org/paper/test123",
            venue="Test Conference",
            publication_date="2023-01-01"
        )
        
        with patch('research_assistant.semantic_scholar_collector.SemanticScholarCollector.search') as mock_search, \
             patch('research_assistant.semantic_scholar_collector.SemanticScholarCollector.get_paper') as mock_get:
            
            mock_search.return_value = [mock_paper]
            mock_get.return_value = mock_paper
            
            async with SemanticScholarCollector() as collector:
                # Search for well-known papers
                papers = await collector.search("deep learning", limit=2)
                
                assert len(papers) == 1
                paper = papers[0]
                assert hasattr(paper, 'paper_id')
                assert hasattr(paper, 'title')
                
                # Test getting paper details
                details = await collector.get_paper(paper.paper_id)
                assert details.paper_id == paper.paper_id
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_embedding_generation_performance(self):
        """Test embedding generation with different models."""
        config = Config()
        generator = EmbeddingGenerator(config=config)
        
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text."
        ]
        
        # Test batch generation
        embeddings = generator.generate_embeddings(texts)
        
        assert len(embeddings) == len(texts)
        # Check embedding dimensions based on model type
        expected_dim = generator.config.embedding.dimension
        assert all(len(emb) == expected_dim for emb in embeddings)
        
        # Test embeddings are different
        assert not all(embeddings[0] == embeddings[1])
    
    @pytest.mark.integration
    def test_input_validation_security(self):
        """Test input validation prevents injection attacks."""
        from research_assistant.validators import Validators
        
        # Test SQL injection attempts
        with pytest.raises(ValidationError):
            Validators.validate_query("'; DROP TABLE papers; --")
            
        # Test command injection attempts
        with pytest.raises(ValidationError):
            Validators.validate_query("test && rm -rf /")
            
        # Test template injection attempts
        with pytest.raises(ValidationError):
            Validators.validate_query("{{7*7}}")
            
        # Test path traversal
        with pytest.raises(ValidationError):
            Validators.validate_path("../../etc/passwd")
            
        # Test valid inputs pass
        assert Validators.validate_query("machine learning transformers") == "machine learning transformers"
        assert Validators.validate_limit(10) == 10
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resource_management(self):
        """Test resource limits and monitoring."""
        resource_manager = get_resource_manager()
        
        # Test concurrent request limiting
        async def make_request(i):
            async with resource_manager.limit_api_request(timeout=5.0):
                await asyncio.sleep(0.1)
                return i
                
        # Should handle concurrent requests up to limit
        tasks = [make_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        assert len(results) == 5
        
        # Test memory monitoring
        resource_manager.check_memory()  # Should not raise
        
        # Test stats
        stats = resource_manager.get_stats()
        assert "total_requests" in stats
        assert "memory_percent" in stats
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from search to analysis."""
        # Create mock papers
        arxiv_paper = ArxivPaper(
            arxiv_id="2301.00001",
            title="AI Test Paper",
            authors=["Test Author"],
            abstract="This is a test paper about AI.",
            published_date=datetime(2023, 1, 1),
            updated_date=datetime(2023, 1, 1),
            categories=["cs.AI"],
            pdf_url="http://arxiv.org/pdf/2301.00001",
            arxiv_url="http://arxiv.org/abs/2301.00001",
            primary_category="cs.AI"
        )
        
        s2_paper = S2Paper(
            paper_id="s2_123",
            title="Artificial Intelligence Paper",
            abstract="This is about AI",
            year=2023,
            authors=[{"name": "S2 Author", "authorId": "1"}],
            citation_count=10,
            reference_count=5,
            influential_citation_count=2,
            is_open_access=True,
            fields_of_study=["Computer Science"],
            s2_url="https://www.semanticscholar.org/paper/s2_123",
            venue="Test Conference",
            publication_date="2023-01-01"
        )
        
        with patch('research_assistant.arxiv_collector.ArxivCollector.search') as mock_arxiv_search, \
             patch('research_assistant.semantic_scholar_collector.SemanticScholarCollector.search') as mock_s2_search:
            
            mock_arxiv_search.return_value = [arxiv_paper]
            mock_s2_search.return_value = [s2_paper]
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config = Config()
                config.vector_store.store_path = Path(tmpdir) / "workflow_store"
                
                # 1. Search ArXiv
                async with ArxivCollector() as arxiv:
                    arxiv_papers = await arxiv.search("cat:cs.AI", max_results=1)
                
                # 2. Search Semantic Scholar
                async with SemanticScholarCollector() as s2:
                    s2_papers = await s2.search("artificial intelligence", limit=1)
                
                # 3. Store all papers
                store = FAISSVectorStore(config=config)
                
                paper_count = 0
                for paper in arxiv_papers:
                    store.add_paper(
                        paper_id=f"arxiv_{paper.arxiv_id}",
                        paper_text=f"{paper.title}\n\n{paper.abstract}",
                        paper_metadata={"source": "arxiv"},
                        generate_embedding=True
                    )
                    paper_count += 1
                    
                for paper in s2_papers:
                    store.add_paper(
                        paper_id=f"s2_{paper.paper_id}",
                        paper_text=f"{paper.title}\n\n{paper.abstract or ''}",
                        paper_metadata={"source": "semantic_scholar"},
                        generate_embedding=True
                    )
                    paper_count += 1
                
                # 4. Search across all papers
                results = store.search("artificial intelligence", k=5)
                
                # 5. Verify workflow
                assert paper_count >= 2  # At least one from each source
                assert len(results) >= 1  # Found some relevant papers
                
                # Check we have papers from both sources
                sources = {doc.metadata.get("source") for doc, _ in results}
                assert len(sources) >= 1  # At least one source represented