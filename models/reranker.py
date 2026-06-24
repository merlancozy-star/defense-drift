"""Reranker wrapper for Qwen3-Reranker-8B.

For retrieval: embedder → top_k candidates → reranker → final top_k.
Optional — pipeline can run with embedder-only retrieval.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class RerankerWrapper:
    """Cross-encoder reranker for improving retrieval quality.

    Scores (query, passage) pairs with a relevance score.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.device = device

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return

        logger.info(f"Loading reranker from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self._loaded = True
        logger.info(f"Reranker loaded on {self.device}")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        passages: List[str],
        top_k: int = 10,
        batch_size: int = 16,
    ) -> List[Tuple[int, float]]:
        """Rerank passages by relevance to query.

        Args:
            query: The query text.
            passages: Candidate passages to rerank.
            top_k: Number of top passages to return.
            batch_size: Batch size for scoring.

        Returns:
            List of (passage_index, relevance_score) sorted by score descending.
        """
        if not self._loaded:
            self.load()

        scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i:i + batch_size]
            pairs = [(query, p) for p in batch]

            enc = self.tokenizer(
                pairs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**enc)
            batch_scores = outputs.logits.squeeze(-1).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)

        # Sort by score descending
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return ranked

    def is_loaded(self) -> bool:
        return self._loaded
