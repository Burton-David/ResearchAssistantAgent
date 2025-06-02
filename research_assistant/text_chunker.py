"""
Text chunking utilities for research papers.
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from .config import ChunkingConfig


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    text: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    chunk_id: str
    
    @property
    def char_count(self) -> int:
        """Get character count of the chunk."""
        return len(self.text)


class PaperChunker:
    """
    Chunks research papers intelligently by sections and size.
    
    Handles common paper structures and maintains context across chunks.
    """
    
    # Common section headers in research papers
    SECTION_PATTERNS = [
        r"^#+\s*(abstract|introduction|related work|background|"
        r"methodology|methods|approach|experiments|results|evaluation|"
        r"discussion|conclusion|future work|references|acknowledgments)",
        r"^\d+\.?\s*(introduction|related work|background|"
        r"methodology|methods|approach|experiments|results|evaluation|"
        r"discussion|conclusion|future work)",
        r"^(abstract|introduction|related work|background|"
        r"methodology|methods|approach|experiments|results|evaluation|"
        r"discussion|conclusion|future work|references|acknowledgments)\s*:?"
    ]
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker with configuration.
        
        Args:
            config: Chunking configuration
        """
        self.config = config or ChunkingConfig()
        self.section_regex = re.compile("|".join(self.SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)
        
    def chunk_paper(
        self,
        text: str,
        paper_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[TextChunk]:
        """
        Chunk a research paper into manageable pieces.
        
        Args:
            text: Full paper text
            paper_id: Unique identifier for the paper
            metadata: Additional metadata for all chunks
            
        Returns:
            List of text chunks
        """
        metadata = metadata or {}
        
        if self.config.separate_sections:
            # Try to chunk by sections first
            sections = self._extract_sections(text)
            if sections:
                return self._chunk_sections(sections, paper_id, metadata)
                
        # Fallback to simple chunking
        return self._simple_chunk(text, paper_id, metadata)
        
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract sections from paper text."""
        sections = []
        
        # Find all section headers
        matches = list(self.section_regex.finditer(text))
        
        if not matches:
            return sections
            
        # Extract text between sections
        for i, match in enumerate(matches):
            section_name = match.group().strip().lower()
            section_name = re.sub(r'[^a-z\s]', '', section_name).strip()
            
            # Skip if not in desired sections
            if (self.config.sections_to_chunk and 
                section_name not in self.config.sections_to_chunk):
                continue
                
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start:end].strip()
            
            if section_text and len(section_text) >= self.config.min_chunk_size:
                sections.append({
                    'name': section_name,
                    'text': section_text,
                    'start': start,
                    'end': end
                })
                
        return sections
        
    def _chunk_sections(
        self,
        sections: List[Dict[str, Any]],
        paper_id: str,
        base_metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk individual sections."""
        chunks = []
        
        for section in sections:
            section_chunks = self._chunk_text(
                section['text'],
                start_offset=section['start']
            )
            
            for i, (chunk_text, start, end) in enumerate(section_chunks):
                chunk_metadata = {
                    **base_metadata,
                    'section': section['name'],
                    'section_chunk_index': i,
                    'section_chunk_total': len(section_chunks)
                }
                
                chunk = TextChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    start_char=start,
                    end_char=end,
                    chunk_id=f"{paper_id}_{section['name']}_{i}"
                )
                chunks.append(chunk)
                
        return chunks
        
    def _simple_chunk(
        self,
        text: str,
        paper_id: str,
        metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Simple chunking by size."""
        chunks = []
        text_chunks = self._chunk_text(text)
        
        for i, (chunk_text, start, end) in enumerate(text_chunks):
            chunk_metadata = {
                **metadata,
                'chunk_index': i,
                'chunk_total': len(text_chunks)
            }
            
            chunk = TextChunk(
                text=chunk_text,
                metadata=chunk_metadata,
                start_char=start,
                end_char=end,
                chunk_id=f"{paper_id}_chunk_{i}"
            )
            chunks.append(chunk)
            
        return chunks
        
    def _chunk_text(
        self,
        text: str,
        start_offset: int = 0
    ) -> List[tuple[str, int, int]]:
        """
        Chunk text by size with overlap.
        
        Returns list of (chunk_text, start_pos, end_pos) tuples.
        """
        if len(text) <= self.config.chunk_size:
            return [(text, start_offset, start_offset + len(text))]
            
        chunks = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find chunk end
            chunk_end = min(current_pos + self.config.chunk_size, len(text))
            
            # Try to break at sentence boundary
            if chunk_end < len(text):
                # Look for sentence end
                sentence_end = text.rfind('.', current_pos, chunk_end)
                if sentence_end > current_pos + self.config.min_chunk_size:
                    chunk_end = sentence_end + 1
                else:
                    # Try to break at word boundary
                    word_end = text.rfind(' ', current_pos, chunk_end)
                    if word_end > current_pos + self.config.min_chunk_size:
                        chunk_end = word_end
                        
            chunk_text = text[current_pos:chunk_end].strip()
            
            if chunk_text:
                chunks.append((
                    chunk_text,
                    start_offset + current_pos,
                    start_offset + chunk_end
                ))
                
            # Move position with overlap
            if chunk_end < len(text):
                current_pos = chunk_end - self.config.chunk_overlap
            else:
                break
                
        return chunks
        
    def chunk_abstract(self, abstract: str, paper_id: str) -> TextChunk:
        """
        Create a single chunk for paper abstract.
        
        Abstracts are typically kept as single chunks since they're concise.
        """
        return TextChunk(
            text=abstract,
            metadata={'section': 'abstract', 'is_abstract': True},
            start_char=0,
            end_char=len(abstract),
            chunk_id=f"{paper_id}_abstract"
        )