#!/usr/bin/env python3
"""
Secure Research Assistant Demo - Shows production-ready features.
"""

import asyncio
import os
from pathlib import Path
from research_assistant import (
    ArxivCollector,
    Config,
    FAISSVectorStore,
)
from research_assistant.validators import Validators, ValidationError
from research_assistant.resource_manager import get_resource_manager
from research_assistant.exceptions import ResearchAssistantError, ErrorHandler
from research_assistant.security import get_secure_config


async def secure_search_demo():
    """Demonstrate secure search with validation and error handling."""
    print("=== Secure Research Assistant Demo ===\n")
    
    # 1. Input validation
    print("1. Testing input validation...")
    queries = [
        "machine learning",  # Valid
        "<script>alert('xss')</script>",  # XSS attempt
        "a" * 1001,  # Too long
        "quantum computing AND optimization",  # Valid complex query
    ]
    
    for query in queries:
        try:
            validated = Validators.validate_query(query[:50] + "..." if len(query) > 50 else query)
            print(f"  ✓ Valid query: '{validated}'")
        except ValidationError as e:
            print(f"  ✗ Rejected: {ErrorHandler.create_user_message(e)}")
    
    # 2. Resource management
    print("\n2. Resource monitoring...")
    resource_manager = get_resource_manager()
    stats = resource_manager.get_stats()
    print(f"  Memory usage: {stats['memory_percent']:.1f}%")
    print(f"  CPU usage: {stats['cpu_percent']:.1f}%")
    print(f"  Max concurrent requests: {stats['max_concurrent_requests']}")
    
    # 3. Secure API key handling
    print("\n3. Secure configuration...")
    secure_config = get_secure_config()
    
    # Check if API keys are configured
    if secure_config.get_api_key("openai"):
        print("  ✓ OpenAI API key configured securely")
    else:
        print("  ℹ OpenAI API key not configured (embeddings will use sentence-transformers)")
    
    # 4. Real search with error handling
    print("\n4. Performing secure search...")
    
    config = Config()
    config.vector_store.store_path = Path.home() / ".research_assistant" / "secure_demo"
    
    try:
        async with ArxivCollector() as arxiv:
            # Search with validated query
            query = Validators.validate_query("machine learning security")
            
            # Use resource manager for the search
            async with resource_manager.limit_api_request():
                papers = await arxiv.search(query, max_results=2)
                
            print(f"  Found {len(papers)} papers")
            
            # Store with embeddings
            if papers:
                store = FAISSVectorStore(config=config)
                
                for paper in papers:
                    print(f"\n  Processing: {paper.title[:60]}...")
                    
                    # Generate embeddings with resource limits
                    async with resource_manager.limit_embedding_operation():
                        doc_ids = store.add_paper(
                            paper_id=f"arxiv_{paper.arxiv_id}",
                            paper_text=f"{paper.title}\n\n{paper.abstract}",
                            paper_metadata={
                                "title": paper.title,
                                "authors": paper.authors[:3],
                                "arxiv_id": paper.arxiv_id,
                                "url": paper.arxiv_url
                            },
                            chunk_paper=False,
                            generate_embedding=True
                        )
                    
                    print(f"    ✓ Stored with ID: {doc_ids[0]}")
                
                # Save securely
                store.save(config.vector_store.store_path)
                print(f"\n  ✓ Vector store saved to: {config.vector_store.store_path}")
                
                # Demonstrate search
                print("\n5. Testing similarity search...")
                results = store.search("adversarial attacks", k=2)
                
                for i, (doc, score) in enumerate(results, 1):
                    print(f"\n  Result {i} (score: {score:.3f}):")
                    print(f"    Title: {doc.metadata['title'][:60]}...")
                    print(f"    URL: {doc.metadata['url']}")
                
    except ResearchAssistantError as e:
        print(f"\n  Error: {ErrorHandler.create_user_message(e)}")
        if ErrorHandler.get_recovery_strategy(e):
            print(f"  Recovery: {ErrorHandler.get_recovery_strategy(e)}")
    except Exception as e:
        print(f"\n  Unexpected error: {e}")
        ErrorHandler.handle_error(e)
    
    print("\n=== Demo Complete ===")


def main():
    """Run the secure demo."""
    # Configure resource limits for production
    from research_assistant.resource_manager import configure_resource_limits
    
    configure_resource_limits(
        max_concurrent_requests=5,
        max_concurrent_embeddings=2,
        max_memory_percent=75.0,
        default_timeout=30.0
    )
    
    # Run demo
    asyncio.run(secure_search_demo())


if __name__ == "__main__":
    main()