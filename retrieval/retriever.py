"""FAISS-based retriever with optional cross-encoder reranking.

The retrieval pipeline is:
  1. Chunk corpus → embed → build FAISS index
  2. For each query: embed → FAISS search (top_k * n_probe) → rerank → top_k
"""

import numpy as np
import faiss
from typing import List, Optional, Tuple
import logging

from data.schemas import Passage, RetrievalConfig
from retrieval.embedder import EmbedderWrapper
from retrieval.chunker import TextChunker
from models.reranker import RerankerWrapper

logger = logging.getLogger(__name__)


class Retriever:
    """End-to-end retriever: chunk → embed → index → search → rerank."""

    def __init__(
        self,
        config: RetrievalConfig,
        embedder: Optional[EmbedderWrapper] = None,
        reranker: Optional[RerankerWrapper] = None,
    ):
        """
        Args:
            config: Retrieval configuration (chunk size, top_k, etc.).
            embedder: Pre-loaded embedder wrapper (shared across retriever instances).
            reranker: Optional pre-loaded reranker.
        """
        self.config = config
        self.embedder = embedder
        self.reranker = reranker
        self.chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        self.index: Optional[faiss.Index] = None
        self.passage_map: List[Passage] = []  # index position → Passage
        self._indexed = False

    def index_corpus(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """Build FAISS index from a document corpus.

        This is called once per domain. The index is rebuilt when switching domains.

        Args:
            documents: List of document texts.
            doc_ids: Optional document IDs.
        """
        if self.embedder is None:
            raise RuntimeError("Embedder must be set before indexing")

        # Chunk
        chunk_pairs = self.chunker.chunk_with_ids(documents, doc_ids)
        chunk_texts = [p[0] for p in chunk_pairs]
        chunk_ids = [p[1] for p in chunk_pairs]

        logger.info(f"Indexing {len(chunk_texts)} chunks...")

        # Embed
        embeddings = self.embedder.encode_passages(chunk_texts)

        # Build FAISS index
        dim = embeddings.shape[1]
        index_type = self.config.faiss_index_type

        if index_type == "Flat":
            if self.config.similarity_metric == "cosine":
                self.index = faiss.IndexFlatIP(dim)  # Inner product on normalized vectors = cosine
            else:
                self.index = faiss.IndexFlatL2(dim)
        elif index_type == "IVF256":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, 256)
            self.index.train(embeddings.astype(np.float32))
        else:
            raise ValueError(f"Unknown FAISS index type: {index_type}")

        self.index.add(embeddings.astype(np.float32))

        # Build passage map
        self.passage_map = [
            Passage(id=chunk_ids[j], text=chunk_texts[j], embedding=embeddings[j].tolist())
            for j in range(len(chunk_texts))
        ]

        self._indexed = True
        logger.info(f"Index built: {self.index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
    ) -> List[Passage]:
        """Retrieve top_k passages for a query.

        Args:
            query: The query text.
            top_k: Number of passages to retrieve (default: config.top_k).
            rerank: Whether to use reranker (default: config.reranker_enabled).

        Returns:
            List of Passage objects with retrieval scores.
        """
        if not self._indexed:
            raise RuntimeError("Index not built. Call index_corpus() first.")
        if self.embedder is None:
            raise RuntimeError("Embedder not set.")

        if top_k is None:
            top_k = self.config.top_k
        if rerank is None:
            rerank = self.config.reranker_enabled

        # Phase 1: Embedding-based retrieval
        query_emb = self.embedder.encode_query(query).reshape(1, -1).astype(np.float32)

        # Retrieve more candidates if reranking
        n_probe = top_k * 3 if rerank and self.reranker is not None else top_k
        n_probe = min(n_probe, self.index.ntotal)

        distances, indices = self.index.search(query_emb, n_probe)
        distances = distances[0]
        indices = indices[0]

        # Build candidate passages
        candidates = []
        for dist, idx in zip(distances, indices):
            if idx < 0 or idx >= len(self.passage_map):
                continue
            p = self.passage_map[idx]
            # Create a copy with the retrieval score
            candidates.append(Passage(
                id=p.id,
                text=p.text,
                embedding=p.embedding,
                source_doc_id=p.source_doc_id,
                retrieval_score=float(dist),
            ))

        # Phase 2: Rerank (optional)
        if rerank and self.reranker is not None and self.reranker.is_loaded():
            ranked = self.reranker.rerank(
                query, [p.text for p in candidates], top_k=top_k
            )
            reranked = []
            for idx, score in ranked:
                p = candidates[idx]
                p.retrieval_score = float(score)
                reranked.append(p)
            return reranked[:top_k]

        # No reranking: use embedding distances
        candidates.sort(key=lambda p: p.retrieval_score or 0, reverse=True)
        return candidates[:top_k]

    def retrieve_for_queries(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
    ) -> List[List[Passage]]:
        """Batch retrieve passages for multiple queries."""
        results = []
        for query in queries:
            passages = self.search(query, top_k=top_k)
            results.append(passages)
        return results

    def is_indexed(self) -> bool:
        return self._indexed

    def get_config_fingerprint(self) -> dict:
        """Return the retrieval config for cross-domain consistency checks."""
        return self.config.to_dict()
