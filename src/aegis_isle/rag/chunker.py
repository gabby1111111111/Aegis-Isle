"""
Text chunking strategies for RAG pipeline.
"""

import re
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.config import settings
from ..core.logging import logger
from .document_processor import DocumentChunk, ProcessedDocument


class BaseChunker(ABC):
    """Base class for text chunking strategies."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk a document into smaller pieces."""
        pass

    def _create_chunk(
        self,
        document_id: str,
        content: str,
        chunk_index: int,
        start_pos: int = 0,
        end_pos: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentChunk:
        """Create a DocumentChunk object."""
        return DocumentChunk(
            document_id=document_id,
            content=content.strip(),
            chunk_index=chunk_index,
            start_pos=start_pos,
            end_pos=end_pos,
            metadata=metadata or {}
        )


class RecursiveChunker(BaseChunker):
    """
    Recursive text chunking that tries to preserve semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.separators = separators or [
            "\n\n",  # Double newline (paragraph breaks)
            "\n",    # Single newline
            ". ",    # Sentence endings
            "! ",    # Exclamation sentences
            "? ",    # Question sentences
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Spaces
            ""       # Character level (fallback)
        ]

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document using recursive strategy."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with recursive strategy")

        chunks = []
        text_chunks = self._split_text(document.content)

        for i, chunk_text in enumerate(text_chunks):
            if chunk_text.strip():
                chunk = self._create_chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={
                        "chunking_strategy": "recursive",
                        "chunk_size": self.chunk_size,
                        "overlap": self.chunk_overlap
                    }
                )
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document {document.id}")
        return chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text recursively using different separators."""
        return self._split_text_recursive(text, self.separators)

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using the provided separators."""
        if not separators:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            # Character-level splitting (fallback)
            return self._split_by_character(text)

        splits = text.split(separator)
        chunks = []
        current_chunk = ""

        for split in splits:
            # If this split would make the chunk too large, process current chunk
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    # Current chunk is ready, process it
                    if len(current_chunk) > self.chunk_size:
                        # Current chunk is still too large, split recursively
                        sub_chunks = self._split_text_recursive(
                            current_chunk, remaining_separators
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_chunk)

                # Start new chunk with current split
                current_chunk = split

        # Add remaining chunk
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._split_text_recursive(
                    current_chunk, remaining_separators
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)

        # Add overlap between chunks
        return self._add_overlap(chunks)

    def _split_by_character(self, text: str) -> List[str]:
        """Split text by characters when all other separators fail."""
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or self.chunk_overlap == 0:
            return chunks

        overlapped_chunks = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = overlapped_chunks[-1]
            current_chunk = chunks[i]

            # Add overlap from previous chunk
            if len(prev_chunk) >= self.chunk_overlap:
                overlap = prev_chunk[-self.chunk_overlap:]
                overlapped_chunk = overlap + " " + current_chunk
            else:
                overlapped_chunk = current_chunk

            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks


class SemanticChunker(BaseChunker):
    """
    Semantic-aware chunking that uses sentence embeddings to group
    semantically similar sentences together.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        similarity_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(chunk_size, chunk_overlap, **kwargs)
        self.similarity_threshold = similarity_threshold
        self._sentence_model = None

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document using semantic similarity."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with semantic strategy")

        # Split into sentences
        sentences = self._split_into_sentences(document.content)
        if not sentences:
            return []

        # Group sentences semantically
        semantic_groups = self._group_sentences_semantically(sentences)

        # Create chunks from groups
        chunks = []
        for i, group in enumerate(semantic_groups):
            chunk_text = " ".join(group)
            if chunk_text.strip():
                chunk = self._create_chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    metadata={
                        "chunking_strategy": "semantic",
                        "sentence_count": len(group),
                        "similarity_threshold": self.similarity_threshold
                    }
                )
                chunks.append(chunk)

        logger.info(f"Created {len(chunks)} semantic chunks for document {document.id}")
        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - could be improved with spaCy or NLTK
        sentence_endings = re.compile(r'[.!?]+')
        sentences = sentence_endings.split(text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def _group_sentences_semantically(self, sentences: List[str]) -> List[List[str]]:
        """Group sentences based on semantic similarity."""
        if not self._sentence_model:
            self._load_sentence_model()

        if not self._sentence_model or len(sentences) <= 1:
            # Fallback to size-based grouping
            return self._group_by_size(sentences)

        try:
            # Compute embeddings for all sentences
            embeddings = self._sentence_model.encode(sentences)

            # Group sentences based on similarity
            groups = []
            current_group = [sentences[0]]
            current_length = len(sentences[0])

            for i in range(1, len(sentences)):
                sentence = sentences[i]
                sentence_length = len(sentence)

                # Check if adding this sentence would exceed chunk size
                if current_length + sentence_length > self.chunk_size:
                    groups.append(current_group)
                    current_group = [sentence]
                    current_length = sentence_length
                else:
                    # Check semantic similarity with current group
                    group_embedding = embeddings[i-len(current_group):i].mean(axis=0)
                    sentence_embedding = embeddings[i]

                    similarity = self._cosine_similarity(group_embedding, sentence_embedding)

                    if similarity >= self.similarity_threshold:
                        current_group.append(sentence)
                        current_length += sentence_length
                    else:
                        groups.append(current_group)
                        current_group = [sentence]
                        current_length = sentence_length

            # Add final group
            if current_group:
                groups.append(current_group)

            return groups

        except Exception as e:
            logger.warning(f"Semantic grouping failed: {e}, falling back to size-based")
            return self._group_by_size(sentences)

    def _group_by_size(self, sentences: List[str]) -> List[List[str]]:
        """Fallback grouping by size only."""
        groups = []
        current_group = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length <= self.chunk_size:
                current_group.append(sentence)
                current_length += sentence_length
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_length = sentence_length

        if current_group:
            groups.append(current_group)

        return groups

    def _load_sentence_model(self):
        """Load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model for semantic chunking")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, "
                "semantic chunking will fall back to size-based"
            )
            self._sentence_model = None

    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunking with overlap."""

    def chunk_document(self, document: ProcessedDocument) -> List[DocumentChunk]:
        """Chunk document into fixed-size pieces."""
        if not document.content.strip():
            return []

        logger.debug(f"Chunking document {document.id} with fixed-size strategy")

        chunks = []
        text = document.content
        step_size = self.chunk_size - self.chunk_overlap

        for i, start in enumerate(range(0, len(text), step_size)):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]

            if chunk_text.strip():
                chunk = self._create_chunk(
                    document_id=document.id,
                    content=chunk_text,
                    chunk_index=i,
                    start_pos=start,
                    end_pos=end,
                    metadata={
                        "chunking_strategy": "fixed_size",
                        "chunk_size": self.chunk_size,
                        "overlap": self.chunk_overlap
                    }
                )
                chunks.append(chunk)

            # Break if we've reached the end
            if end >= len(text):
                break

        logger.info(f"Created {len(chunks)} fixed-size chunks for document {document.id}")
        return chunks


def get_chunker(strategy: str = "recursive", **kwargs) -> BaseChunker:
    """Factory function to get a chunker based on strategy."""
    chunkers = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "fixed": FixedSizeChunker,
    }

    if strategy not in chunkers:
        logger.warning(f"Unknown chunking strategy '{strategy}', using recursive")
        strategy = "recursive"

    # Use settings defaults if not provided
    chunk_size = kwargs.get("chunk_size", settings.chunk_size)
    chunk_overlap = kwargs.get("chunk_overlap", settings.chunk_overlap)

    return chunkers[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )