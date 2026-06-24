"""Text chunking with fixed size and overlap.

Strategy is locked across domains per CLAUDE.md hard constraint:
  - Same chunk_size and chunk_overlap for source AND target domains
  - Only the CORPUS changes between domains
"""

from typing import List
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Fixed-size text chunker with overlap.

    Simple whitespace-based splitting. In production, consider
    using the tokenizer's own tokenization for accurate sizing.
    """

    def __init__(self, chunk_size: int = 256, chunk_overlap: int = 32):
        """
        Args:
            chunk_size: Target chunk size in words.
            chunk_overlap: Overlap between adjacent chunks in words.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        assert chunk_overlap < chunk_size, "Overlap must be less than chunk size"

    def chunk_text(self, text: str) -> List[str]:
        """Split a single document into chunks."""
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if not chunk_words:
                break
            chunks.append(" ".join(chunk_words))
            if i + self.chunk_size >= len(words):
                break
        return chunks

    def chunk_documents(self, documents: List[str]) -> List[str]:
        """Chunk a list of documents into a flat list of chunks."""
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
        logger.debug(
            f"Chunked {len(documents)} docs → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    def chunk_with_ids(
        self, documents: List[str], doc_ids: List[str] = None
    ) -> List[tuple]:
        """Chunk documents and return (chunk_text, source_doc_id) pairs."""
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]

        pairs = []
        for doc, doc_id in zip(documents, doc_ids):
            chunks = self.chunk_text(doc)
            for j, chunk in enumerate(chunks):
                pairs.append((chunk, f"{doc_id}_chunk_{j}"))
        return pairs

    def to_config_dict(self) -> dict:
        """Export chunker configuration for consistency checks."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
