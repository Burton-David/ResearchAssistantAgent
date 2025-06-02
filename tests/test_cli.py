"""Tests for CLI functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import pytest
from click.testing import CliRunner
from research_assistant.cli import cli, main
from research_assistant.arxiv_collector import ArxivPaper


class TestCLI:
    """Test cases for CLI commands."""
    
    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()
    
    @pytest.fixture
    def mock_arxiv_paper(self):
        """Create a mock ArXiv paper."""
        from datetime import datetime
        return ArxivPaper(
            arxiv_id="2301.00234",
            title="Test Paper",
            abstract="This is a test abstract for the paper.",
            authors=["Author One", "Author Two"],
            published_date=datetime(2023, 1, 1),
            updated_date=datetime(2023, 1, 1),
            categories=["cs.LG"],
            primary_category="cs.LG",
            pdf_url="https://arxiv.org/pdf/2301.00234.pdf",
            arxiv_url="https://arxiv.org/abs/2301.00234"
        )
    
    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Research Assistant" in result.output
        assert "Commands:" in result.output
    
    @patch("research_assistant.cli.ArxivCollector")
    def test_search_command(self, mock_collector_class, runner, mock_arxiv_paper):
        """Test search command."""
        # Setup mock
        mock_collector = AsyncMock()
        mock_collector.__aenter__.return_value = mock_collector
        mock_collector.search.return_value = [mock_arxiv_paper]
        mock_collector_class.return_value = mock_collector
        
        result = runner.invoke(cli, ["search", "machine learning", "--limit", "1"])
        
        assert result.exit_code == 0
        assert "Test Paper" in result.output
        assert "Author One" in result.output
    
    @patch("research_assistant.cli.ArxivCollector")
    @patch("research_assistant.cli.FAISSVectorStore")
    def test_search_with_store(self, mock_store_class, mock_collector_class, 
                              runner, mock_arxiv_paper):
        """Test search command with --store flag."""
        # Setup mocks
        mock_collector = AsyncMock()
        mock_collector.__aenter__.return_value = mock_collector
        mock_collector.search.return_value = [mock_arxiv_paper]
        mock_collector_class.return_value = mock_collector
        
        mock_store = Mock()
        mock_store.add_paper = Mock()
        mock_store.save = Mock()
        mock_store_class.return_value = mock_store
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(cli, [
                "search", "test query",
                "--store",
                "--store-path", tmpdir
            ])
        
        assert result.exit_code == 0
        assert mock_store.add_paper.called
        assert mock_store.save.called
    
    @patch("research_assistant.cli.FAISSVectorStore")
    def test_similarity_search(self, mock_store_class, runner):
        """Test similarity search command."""
        # Setup mock
        mock_store = Mock()
        mock_doc = Mock()
        mock_doc.metadata = {"title": "Test Paper"}
        mock_doc.text = "Test content"
        mock_store.search.return_value = [(mock_doc, 0.95)]
        # Mock both constructor and load method
        mock_store_class.return_value = mock_store
        mock_store_class.load.return_value = mock_store
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy store file
            store_path = Path(tmpdir) / "vector_store"
            store_path.mkdir()
            (store_path / "index.faiss").touch()
            
            result = runner.invoke(cli, [
                "similarity-search", "test query",
                "--store-path", str(store_path),
                "--limit", "1"
            ])
        
        assert result.exit_code == 0
        assert "Test Paper" in result.output
        assert "0.95" in result.output
    
    @patch("research_assistant.cli.FAISSVectorStore")
    def test_stats_command(self, mock_store_class, runner):
        """Test stats command."""
        # Setup mock
        mock_store = Mock()
        mock_store.get_statistics.return_value = {
            "total_documents": 10,
            "total_papers": 5,
            "embedding_dimension": 384,
            "index_type": "Flat"
        }
        # Mock both constructor and load method
        mock_store_class.return_value = mock_store
        mock_store_class.load.return_value = mock_store
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy store file
            store_path = Path(tmpdir) / "vector_store"
            store_path.mkdir()
            (store_path / "index.faiss").touch()
            
            result = runner.invoke(cli, ["stats", "--store-path", str(store_path)])
        
        assert result.exit_code == 0
        assert "Total Documents" in result.output
        assert "10" in result.output