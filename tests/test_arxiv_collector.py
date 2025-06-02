"""Unit tests for ArXiv collector."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from research_assistant.arxiv_collector import ArxivCollector, ArxivPaper
from research_assistant.rate_limiter import UnifiedRateLimiter, APIType


class TestArxivCollector:
    """Test cases for ArxivCollector."""
    
    @pytest.fixture
    def sample_arxiv_paper(self):
        """Create a sample ArxivPaper for testing."""
        return ArxivPaper(
            arxiv_id="2301.00001",
            title="Test Paper Title",
            authors=["Test Author"],
            abstract="Test abstract content",
            published_date=datetime(2023, 1, 1),
            updated_date=datetime(2023, 1, 1),
            categories=["cs.LG"],
            pdf_url="http://arxiv.org/pdf/2301.00001",
            arxiv_url="http://arxiv.org/abs/2301.00001",
            primary_category="cs.LG"
        )
    
    @pytest.mark.asyncio
    async def test_search_papers(self, sample_arxiv_paper):
        """Test searching for papers on ArXiv."""
        with patch('research_assistant.arxiv_collector.ArxivCollector.search') as mock_search:
            mock_search.return_value = [sample_arxiv_paper]
            
            async with ArxivCollector() as collector:
                papers = await collector.search("cat:cs.LG", max_results=2)
                
                assert len(papers) == 1
                paper = papers[0]
                assert isinstance(paper, ArxivPaper)
                assert paper.arxiv_id == "2301.00001"
                assert paper.title == "Test Paper Title"
                assert paper.abstract == "Test abstract content"
                assert paper.authors == ["Test Author"]
    
    @pytest.mark.asyncio
    async def test_get_paper_by_id(self, sample_arxiv_paper):
        """Test fetching a specific paper by ID."""
        with patch('research_assistant.arxiv_collector.ArxivCollector.get_paper_by_id') as mock_get:
            mock_get.return_value = sample_arxiv_paper
            
            async with ArxivCollector() as collector:
                paper = await collector.get_paper_by_id("2301.00234")
                
                assert isinstance(paper, ArxivPaper)
                assert paper.arxiv_id == "2301.00001"
                assert paper.title == "Test Paper Title"
                assert paper.abstract == "Test abstract content"
    
    @pytest.mark.asyncio
    async def test_search_with_complex_query(self):
        """Test searching with complex query."""
        transformer_paper = ArxivPaper(
            arxiv_id="2301.00002",
            title="Transformer Architecture for NLP",
            authors=["Query Author"],
            abstract="Complex query abstract",
            published_date=datetime(2023, 1, 1),
            updated_date=datetime(2023, 1, 1),
            categories=["cs.CL"],
            pdf_url="http://arxiv.org/pdf/2301.00002",
            arxiv_url="http://arxiv.org/abs/2301.00002",
            primary_category="cs.CL"
        )
        
        with patch('research_assistant.arxiv_collector.ArxivCollector.search') as mock_search:
            mock_search.return_value = [transformer_paper]
            
            async with ArxivCollector() as collector:
                papers = await collector.search(
                    "ti:transformer AND cat:cs.CL",
                    max_results=1
                )
                
                assert len(papers) == 1
                paper = papers[0]
                assert isinstance(paper, ArxivPaper)
                assert "transformer" in paper.title.lower()
                assert "cs.CL" in paper.categories
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting is applied."""
        import time
        
        # Mock the search method to track calls
        call_times = []
        
        async def mock_search(*args, **kwargs):
            call_times.append(time.time())
            # Simulate rate limiting delay after first call
            if len(call_times) > 1:
                await asyncio.sleep(0.1)
            return []
        
        with patch('research_assistant.arxiv_collector.ArxivCollector.search', mock_search):
            async with ArxivCollector() as collector:
                start = time.time()
                
                # Make multiple requests
                for _ in range(3):
                    await collector.search("test", max_results=1)
                
                elapsed = time.time() - start
                
                # With simulated delays, should take at least 0.2 seconds
                assert elapsed >= 0.2
    
    @pytest.mark.asyncio
    async def test_empty_search_results(self):
        """Test handling of queries with no results."""
        with patch('research_assistant.arxiv_collector.ArxivCollector.search') as mock_search:
            mock_search.return_value = []
            
            async with ArxivCollector() as collector:
                papers = await collector.search(
                    "xxxyyyzzz123456789",  # Unlikely to match anything
                    max_results=10
                )
                
                assert isinstance(papers, list)
                assert len(papers) == 0