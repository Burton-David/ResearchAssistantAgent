#!/usr/bin/env python3
"""
Research Assistant Demo - Shows all key features in one place.
"""

import asyncio
from pathlib import Path
from research_assistant import (
    ArxivCollector,
    SemanticScholarCollector,
    FAISSVectorStore,
    Config,
    EmbeddingModel
)


async def main():
    """Demonstrate key features of the Research Assistant."""
    
    print("Research Assistant Demo")
    print("=" * 50)
    
    # Configure the system
    config = Config()
    config.embedding.model_type = EmbeddingModel.SENTENCE_TRANSFORMERS
    config.embedding.show_progress = True
    
    # Create vector store
    store = FAISSVectorStore(config=config)
    
    # 1. Search ArXiv
    print("\n1. Searching ArXiv for papers...")
    async with ArxivCollector() as arxiv:
        papers = await arxiv.search("cat:cs.LG", max_results=3)
        
    for paper in papers:
        print(f"   - {paper.title[:60]}...")
        
    # 2. Add papers to vector store
    print("\n2. Generating embeddings and storing papers...")
    for paper in papers:
        store.add_paper(
            paper_id=f"arxiv_{paper.arxiv_id}",
            paper_text=f"{paper.title}\n\n{paper.abstract}",
            paper_metadata={
                "title": paper.title,
                "authors": paper.authors[:3],
                "arxiv_id": paper.arxiv_id
            },
            chunk_paper=False,
            generate_embedding=True,
            show_progress=False
        )
    
    # 3. Search for similar papers
    print("\n3. Searching for similar papers...")
    query = "deep learning optimization"
    results = store.search(query, k=3)
    
    print(f"   Query: '{query}'")
    print("   Results:")
    for i, (doc, score) in enumerate(results, 1):
        title = doc.metadata.get("title", "Unknown")[:50]
        print(f"   {i}. {title}... (score: {score:.3f})")
    
    # 4. Save the store
    save_path = Path.home() / ".research_assistant" / "demo_store"
    store.save(save_path)
    print(f"\n4. Vector store saved to {save_path}")
    
    print("\n✅ Demo complete!")
    print("\nYou can also use the CLI:")
    print("  research-assistant search 'machine learning' --store")
    print("  research-assistant similarity-search 'your query'")
    print("  research-assistant stats")


if __name__ == "__main__":
    asyncio.run(main())