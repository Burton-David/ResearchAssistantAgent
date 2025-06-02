"""
FAISS vector store with embedding generation for research papers.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import faiss
from tqdm import tqdm

from .config import get_config, EmbeddingModel
from .text_chunker import PaperChunker, TextChunk

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            text=data["text"],
            metadata=data["metadata"],
            embedding=data.get("embedding")
        )
    
    
class EmbeddingGenerator:
    """Handles embedding generation with multiple backends."""
    
    def __init__(self, config=None):
        """Initialize embedding generator."""
        self.config = config or get_config()
        self.model = None
        self.tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is not None:
            return
            
        if self.config.embedding.model_type == EmbeddingModel.SENTENCE_TRANSFORMERS:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading sentence transformer: {self.config.embedding.model_name}")
                self.model = SentenceTransformer(self.config.embedding.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        elif self.config.embedding.model_type == EmbeddingModel.OPENAI:
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
            try:
                import openai
                self.model = openai.AsyncOpenAI(api_key=self.config.openai_api_key)
            except ImportError:
                raise ImportError("openai not installed. Install with: pip install openai")
                
    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        
        if self.config.embedding.model_type == EmbeddingModel.SENTENCE_TRANSFORMERS:
            return self._generate_sentence_transformer_embeddings(texts, show_progress)
        else:
            # For async OpenAI, we need to run in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._generate_openai_embeddings(texts, show_progress)
                )
            finally:
                loop.close()
                
    def _generate_sentence_transformer_embeddings(
        self,
        texts: List[str],
        show_progress: bool
    ) -> np.ndarray:
        """Generate embeddings using sentence transformers."""
        batch_size = self.config.embedding.batch_size
        
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Generating embeddings")
            
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
            
            if show_progress:
                pbar.update(len(batch))
                
        if show_progress:
            pbar.close()
            
        return np.vstack(embeddings)
        
    async def _generate_openai_embeddings(
        self,
        texts: List[str],
        show_progress: bool
    ) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        embeddings = []
        
        if show_progress:
            pbar = tqdm(total=len(texts), desc="Generating OpenAI embeddings")
            
        # Process in batches to avoid rate limits
        batch_size = min(self.config.embedding.batch_size, 100)  # OpenAI limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.model.embeddings.create(
                    model=self.config.embedding.openai_model,
                    input=batch
                )
                
                batch_embeddings = [e.embedding for e in response.data]
                embeddings.extend(batch_embeddings)
                
                if show_progress:
                    pbar.update(len(batch))
                    
            except Exception as e:
                logger.error(f"Error generating OpenAI embeddings: {e}")
                raise
                
        if show_progress:
            pbar.close()
            
        return np.array(embeddings)


class FAISSVectorStore:
    """
    FAISS-based vector store with integrated embedding generation.
    """
    
    def __init__(
        self,
        dimension: Optional[int] = None,
        index_type: str = "flat",
        metric: str = "cosine",
        nlist: int = 100,
        nprobe: int = 10,
        ef_construction: int = 200,
        ef_search: int = 50,
        config=None
    ):
        """Initialize FAISS vector store with embedding support."""
        self.config = config or get_config()
        
        # Use dimension from config if not specified
        if dimension is None:
            dimension = self.config.embedding.current_dimension
            
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        
        # Create index
        self.index = self._create_index(
            index_type, dimension, metric,
            nlist, ef_construction
        )
        
        # Search parameters
        self.nprobe = nprobe
        self.ef_search = ef_search
        
        # Document storage
        self.documents: Dict[int, Document] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.next_idx = 0
        
        # Components
        self.embedding_generator = EmbeddingGenerator(config)
        self.chunker = PaperChunker(self.config.chunking)
        
        # Auto-save tracking
        self._unsaved_count = 0
        
    def _create_index(
        self,
        index_type: str,
        dimension: int,
        metric: str,
        nlist: int,
        ef_construction: int
    ) -> faiss.Index:
        """Create FAISS index based on specified type."""
        # Ensure dimension is a standard Python int
        dimension = int(dimension)
        
        if index_type == "flat":
            if metric == "cosine":
                index = faiss.IndexFlatIP(dimension)
            elif metric == "l2":
                index = faiss.IndexFlatL2(dimension)
            else:
                index = faiss.IndexFlatIP(dimension)
            return faiss.IndexIDMap(index)
            
        elif index_type == "ivf":
            nlist = int(nlist)
            if metric == "cosine":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            else:
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            return faiss.IndexIDMap(index)
            
        elif index_type == "hnsw":
            ef_construction = int(ef_construction)
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = ef_construction
            return faiss.IndexIDMap(index)
            
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
    def add_paper(
        self,
        paper_id: str,
        paper_text: str,
        paper_metadata: Dict[str, Any],
        chunk_paper: bool = True,
        generate_embedding: bool = True,
        show_progress: bool = True
    ) -> List[str]:
        """
        Add a research paper to the vector store.
        
        Args:
            paper_id: Unique identifier for the paper
            paper_text: Full text or abstract of the paper
            paper_metadata: Metadata about the paper
            chunk_paper: Whether to chunk the paper into sections
            generate_embedding: Whether to generate embeddings
            show_progress: Show progress bar for embedding generation
            
        Returns:
            List of document IDs that were added
        """
        documents = []
        
        if chunk_paper and len(paper_text) > self.config.chunking.chunk_size:
            # Chunk the paper
            chunks = self.chunker.chunk_paper(paper_text, paper_id, paper_metadata)
            
            for chunk in chunks:
                doc = Document(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={**paper_metadata, **chunk.metadata}
                )
                documents.append(doc)
        else:
            # Single document for the whole paper
            doc = Document(
                id=paper_id,
                text=paper_text,
                metadata=paper_metadata
            )
            documents.append(doc)
            
        # Generate embeddings if requested
        if generate_embedding:
            texts = [doc.text for doc in documents]
            embeddings = self.embedding_generator.generate_embeddings(
                texts, 
                show_progress=show_progress and len(texts) > 1
            )
            
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding.tolist()
                
            # Add to index
            self.add_documents(documents, embeddings)
        else:
            # Add without embeddings
            self.add_documents(documents)
            
        return [doc.id for doc in documents]
        
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[np.ndarray] = None
    ) -> None:
        """Add documents to the vector store."""
        if not documents:
            return
            
        # Prepare embeddings
        if embeddings is None and documents[0].embedding is not None:
            embeddings = np.array([doc.embedding for doc in documents])
            
        if embeddings is None:
            logger.warning("No embeddings provided, documents won't be searchable")
            return
            
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
            
        # Generate indices
        indices = []
        for doc in documents:
            idx = self.next_idx
            self.documents[idx] = doc
            self.id_to_idx[doc.id] = idx
            indices.append(idx)
            self.next_idx += 1
            
        indices = np.array(indices, dtype=np.int64)
        
        # Train index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            logger.info(f"Training IVF index with {len(embeddings)} vectors")
            base_index = faiss.downcast_index(self.index.index)
            base_index.train(embeddings.astype(np.float32))
            
        # Add to index
        self.index.add_with_ids(embeddings.astype(np.float32), indices)
        logger.info(f"Added {len(documents)} documents to vector store")
        
        # Auto-save if configured
        self._unsaved_count += len(documents)
        if (self.config.vector_store.auto_save and 
            self._unsaved_count >= self.config.vector_store.save_interval):
            self.save(self.config.vector_store.store_path)
            self._unsaved_count = 0
            
    def search(
        self,
        query: Union[str, List[float], np.ndarray],
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text or embedding vector
            k: Number of results
            filter_metadata: Optional metadata filters
            
        Returns:
            List of (document, score) tuples
        """
        # Generate embedding if query is text
        if isinstance(query, str):
            query_embedding = self.embedding_generator.generate_embeddings(
                [query], 
                show_progress=False
            )[0]
        else:
            query_embedding = np.array(query)
            
        # Ensure 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Normalize for cosine similarity
        if self.metric == "cosine":
            faiss.normalize_L2(query_embedding)
            
        # Set search parameters
        if self.index_type == "ivf":
            base_index = faiss.downcast_index(self.index.index)
            base_index.nprobe = self.nprobe
        elif self.index_type == "hnsw":
            base_index = faiss.downcast_index(self.index.index)
            base_index.hnsw.efSearch = self.ef_search
            
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k * 2)  # Get extra for filtering
        
        # Filter and collect results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
                
            doc = self.documents.get(idx)
            if doc is None:
                continue
                
            # Apply metadata filters
            if filter_metadata:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_metadata.items()
                )
                if not match:
                    continue
                    
            results.append((doc, float(score)))
            
            if len(results) >= k:
                break
                
        return results
        
    def save(self, path: Union[str, Path]) -> None:
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save metadata
        metadata = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "next_idx": self.next_idx,
            "config": {
                "embedding_model": self.config.embedding.model_type.value,
                "embedding_model_name": self.config.embedding.model_name,
                "chunk_size": self.config.chunking.chunk_size
            }
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save documents as JSON
        documents_data = {
            "documents": {
                str(idx): doc.to_dict() for idx, doc in self.documents.items()
            },
            "id_to_idx": self.id_to_idx,
            "version": "1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        with open(path / "documents.json", "w") as f:
            json.dump(documents_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved vector store to {path}")
        self._unsaved_count = 0
        
    @classmethod
    def load(cls, path: Union[str, Path], config=None) -> "FAISSVectorStore":
        """Load vector store from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            
        # Create instance
        store = cls(
            dimension=metadata["dimension"],
            index_type=metadata["index_type"],
            metric=metadata["metric"],
            config=config
        )
        
        # Load index
        store.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents from JSON
        documents_path = path / "documents.json"
        
        # Check for legacy pickle file
        legacy_path = path / "documents.pkl"
        if not documents_path.exists() and legacy_path.exists():
            raise ValueError(
                "Found legacy pickle file. Please migrate your data to JSON format.\n"
                "Run: python -m research_assistant.migrate_vector_store <path>"
            )
        
        with open(documents_path, "r") as f:
            data = json.load(f)
            
        # Validate version
        version = data.get("version", "0.0")
        if version != "1.0":
            raise ValueError(f"Unsupported documents format version: {version}")
            
        # Reconstruct documents
        store.documents = {
            int(idx): Document.from_dict(doc_data)
            for idx, doc_data in data["documents"].items()
        }
        store.id_to_idx = data["id_to_idx"]
            
        store.next_idx = metadata["next_idx"]
        
        logger.info(f"Loaded vector store from {path}")
        return store
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.documents),
            "index_type": self.index_type,
            "dimension": self.dimension,
            "metric": self.metric,
            "embedding_model": self.config.embedding.model_type.value,
            "index_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True,
            "index_size": self.index.ntotal
        }
        
    def rebuild_index(self):
        """Rebuild the index from stored documents."""
        if not self.documents:
            logger.warning("No documents to rebuild index from")
            return
            
        # Extract embeddings
        embeddings = []
        indices = []
        
        for idx, doc in self.documents.items():
            if doc.embedding:
                embeddings.append(doc.embedding)
                indices.append(idx)
                
        if not embeddings:
            logger.warning("No embeddings found in documents")
            return
            
        embeddings = np.array(embeddings, dtype=np.float32)
        indices = np.array(indices, dtype=np.int64)
        
        # Reset index
        self.index.reset()
        
        # Re-add embeddings
        if self.metric == "cosine":
            faiss.normalize_L2(embeddings)
            
        if self.index_type == "ivf":
            base_index = faiss.downcast_index(self.index.index)
            base_index.train(embeddings.astype(np.float32))
            
        self.index.add_with_ids(embeddings.astype(np.float32), indices)
        logger.info(f"Rebuilt index with {len(embeddings)} vectors")