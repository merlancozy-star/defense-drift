"""Reranker wrapper for Qwen3-Reranker-8B.

For retrieval: embedder → top_k candidates → reranker → final top_k.
Optional — pipeline can run with embedder-only retrieval.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
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
        self._use_cosine = False

    def load(self):
        if self._loaded:
            return

        logger.info(f"Loading reranker from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        # Try sequence classification first, fall back to AutoModel
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
        except Exception:
            logger.warning(
                "AutoModelForSequenceClassification failed — "
                "Qwen3-Reranker may use a custom architecture. "
                "Using AutoModel as fallback (cosine similarity scoring)."
            )
            self.model = AutoModel.from_pretrained(
                self.model_path,
                dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._use_cosine = True
        else:
            self._use_cosine = False
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

            if self._use_cosine:
                # Cosine similarity mode: embed query and passages separately
                q_enc = self.tokenizer(
                    query, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(self.device)
                p_enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
                ).to(self.device)

                q_out = self.model(**q_enc)
                p_out = self.model(**p_enc)

                # Mean pooling
                q_emb = _mean_pool(q_out.last_hidden_state, q_enc["attention_mask"])
                p_emb = _mean_pool(p_out.last_hidden_state, p_enc["attention_mask"])

                # Cosine similarity
                q_emb = F.normalize(q_emb, dim=-1)
                p_emb = F.normalize(p_emb, dim=-1)
                batch_scores = (q_emb @ p_emb.T).squeeze(0).cpu().tolist()
                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]
            else:
                pairs = [(query, p) for p in batch]
                enc = self.tokenizer(
                    pairs, return_tensors="pt", padding=True,
                    truncation=True, max_length=512,
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


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over token dimension, excluding padding."""
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
