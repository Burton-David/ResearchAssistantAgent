"""Integration tests for Semantic Scholar collector."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import aiohttp
from research_assistant.semantic_scholar_collector import (
    SemanticScholarCollector, 
    SemanticScholarPaper
)
from research_assistant.exceptions import RateLimitError


class TestSemanticScholarCollector:
    """Test cases for SemanticScholarCollector."""
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_search_papers(self, mock_get):
        """Test searching papers on Semantic Scholar."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "data": [
                {
                    "paperId": "123456",
                    "title": "Deep Learning",
                    "abstract": "A paper about deep learning",
                    "authors": [{"name": "John Doe", "authorId": "1"}],
                    "year": 2023,
                    "venue": "ICML",
                    "citationCount": 100,
                    "referenceCount": 50,
                    "influentialCitationCount": 10,
                    "isOpenAccess": True,
                    "s2FieldsOfStudy": [{"category": "Computer Science"}],
                    "url": "https://example.com/paper"
                }
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with SemanticScholarCollector() as collector:
            papers = await collector.search("deep learning", limit=2)
            
            assert len(papers) == 1
            paper = papers[0]
            assert isinstance(paper, SemanticScholarPaper)
            assert paper.paper_id == "123456"
            assert paper.title == "Deep Learning"
            assert paper.abstract == "A paper about deep learning"
            assert isinstance(paper.authors, list)
            assert paper.year == 2023
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_get_paper_by_id(self, mock_get):
        """Test fetching specific paper by ID."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "paperId": "204e3073870fae3d05bcbc2f6a8e263d9b72e776",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce a new language representation model called BERT",
            "authors": [{"name": "Jacob Devlin", "authorId": "123"}],
            "year": 2019,
            "citationCount": 10000,
            "referenceCount": 60,
            "influentialCitationCount": 5000,
            "isOpenAccess": True,
            "s2FieldsOfStudy": [{"category": "Computer Science"}],
            "url": "https://example.com/bert"
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with SemanticScholarCollector() as collector:
            paper = await collector.get_paper("204e3073870fae3d05bcbc2f6a8e263d9b72e776")
            
            assert isinstance(paper, SemanticScholarPaper)
            assert paper.paper_id == "204e3073870fae3d05bcbc2f6a8e263d9b72e776"
            assert "bert" in paper.title.lower()
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_search_with_fields_filter(self, mock_get):
        """Test searching with specific fields."""
        # Mock response with limited fields
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value={
            "data": [
                {
                    "paperId": "789",
                    "title": "Machine Learning",
                    "year": 2022,
                    "citationCount": 50
                }
            ]
        })
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with SemanticScholarCollector() as collector:
            papers = await collector.search(
                "machine learning",
                fields=["title", "year", "citationCount"],
                limit=1
            )
            
            assert len(papers) == 1
            paper = papers[0]
            assert hasattr(paper, "title")
            assert hasattr(paper, "year")
            assert hasattr(paper, "citation_count")
    
    @pytest.mark.asyncio 
    @patch('aiohttp.ClientSession.get')
    async def test_get_paper_citations(self, mock_get):
        """Test fetching paper citations."""
        # Mock search response
        search_response = AsyncMock()
        search_response.status = 200
        search_response.raise_for_status = AsyncMock()
        search_response.json = AsyncMock(return_value={
            "data": [
                {
                    "paperId": "123",
                    "title": "Deep Learning Paper",
                    "year": 2020
                }
            ]
        })
        
        # Mock citations response
        citations_response = AsyncMock()
        citations_response.status = 200
        citations_response.raise_for_status = AsyncMock()
        citations_response.json = AsyncMock(return_value={
            "data": [
                {
                    "citingPaper": {
                        "paperId": "456",
                        "title": "Citing Paper 1",
                        "year": 2021
                    }
                },
                {
                    "citingPaper": {
                        "paperId": "789",
                        "title": "Citing Paper 2",
                        "year": 2022
                    }
                }
            ]
        })
        
        # Set up mock to return different responses
        mock_get.return_value.__aenter__.side_effect = [search_response, citations_response]
        
        async with SemanticScholarCollector() as collector:
            papers = await collector.search("deep learning", limit=1)
            assert len(papers) == 1
            
            citations = await collector.get_paper_citations(
                papers[0].paper_id,
                limit=5
            )
            
            assert isinstance(citations, list)
            assert len(citations) == 2
            assert all(isinstance(c, SemanticScholarPaper) for c in citations)
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_rate_limiting(self, mock_get):
        """Test rate limiting is applied."""
        from research_assistant.rate_limiter import UnifiedRateLimiter, APIType
        
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = AsyncMock()
        mock_response.json = AsyncMock(return_value={"data": []})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Create a custom rate limiter with lower limits for testing
        custom_limiter = UnifiedRateLimiter()
        
        # Test that rate limiter is used by checking method calls
        with patch.object(custom_limiter, 'acquire', wraps=custom_limiter.acquire) as mock_acquire:
            async with SemanticScholarCollector(unified_limiter=custom_limiter) as collector:
                # Make multiple requests
                for i in range(3):
                    await collector.search(f"test{i}", limit=1)
                
                # Verify rate limiter was called for each request
                assert mock_acquire.call_count == 3
                # Verify it was called with the correct API type
                for call in mock_acquire.call_args_list:
                    assert call[0][0] == APIType.SEMANTIC_SCHOLAR
    
    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_rate_limit_error_handling(self, mock_get):
        """Test handling of 429 rate limit errors."""
        # Mock 429 response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {'retry-after': '60'}
        mock_get.return_value.__aenter__.return_value = mock_response
        
        async with SemanticScholarCollector() as collector:
            with pytest.raises(RateLimitError) as exc_info:
                await collector.search("test", limit=1)
            
            assert exc_info.value.service == "semantic_scholar"
            assert exc_info.value.retry_after == 60