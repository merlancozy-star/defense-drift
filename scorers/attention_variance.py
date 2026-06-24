"""AttentionVarianceScorer — cross-passage attention quality.

Signal: when the generator attends to context passages during generation,
poison passages tend to receive anomalous attention patterns (either
abnormally high variance across heads/layers, or consistent under-attention
relative to their retrieval rank).

This scorer requires a GeneratorWrapper with output_attentions=True.

---

IMPORTANT (CLAUDE.md hard constraint):
This scorer requires WHITE-BOX access to model attention weights.
If the serving backend (e.g., vLLM) does not expose per-token attention
across all layers, this scorer will NOT work.

The current implementation assumes HuggingFace Transformers with
output_attentions=True on both the model forward and generate().

TODO (M2):
- [ ] Verify attention extraction works with Qwen3-8B on HF Transformers
- [ ] Validate that attention patterns actually differ for poison vs clean
- [ ] Tune the aggregation: variance? entropy? max-min across heads?
---

HIGHER attention variance → more anomalous attention → more suspicious.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Dict, TYPE_CHECKING
import logging

from scorers.base import BaseScorer

if TYPE_CHECKING:
    from models.generator import GeneratorWrapper

logger = logging.getLogger(__name__)


class AttentionVarianceScorer(BaseScorer):
    """Score passages by variance of attention weights allocated to each passage.

    How it works:
    1. Feed query + passages to the generator
    2. Generate a short answer while recording attention
    3. For each passage, compute the variance of attention weights across:
       - Generation steps (temporal variance)
       - Attention heads (multi-head variance)
       - Layers (depth variance)
    4. Aggregate into a single anomaly score per passage

    Higher variance → model is "uncertain" about this passage → more suspicious.
    """

    def __init__(self, generator: Optional[GeneratorWrapper] = None):
        self._generator = generator

    @property
    def name(self) -> str:
        return "attention_variance"

    def set_generator(self, generator: GeneratorWrapper):
        self._generator = generator

    def score(
        self,
        query: str,
        passages: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Compute attention variance scores.

        Args:
            query: The query text.
            passages: List of retrieved passage texts.
            **kwargs:
                max_new_tokens: Max tokens to generate (default 32).

        Returns:
            np.ndarray of shape (len(passages),) — attention variance per passage.
        """
        if self._generator is None:
            raise RuntimeError(
                "AttentionVarianceScorer requires a GeneratorWrapper. "
                "Call set_generator() before scoring."
            )

        n = len(passages)
        if n == 0:
            return np.array([], dtype=np.float64)

        max_new_tokens = kwargs.get("max_new_tokens", 32)

        try:
            result = self._generator.generate_with_attention(
                prompt=query,
                context_passages=passages,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            logger.warning(f"Attention extraction failed: {e}. Returning zeros.")
            return np.zeros(n, dtype=np.float64)

        attn = result["attention_weights"]
        boundaries = result["passage_boundaries"]

        if not attn:
            logger.warning("No attention weights returned. Returning zeros.")
            return np.zeros(n, dtype=np.float64)

        # Stack all generation steps: [gen_steps, n_layers, n_heads, 1, full_seq_len]
        # We need to aggregate attention to each passage's token span.
        try:
            all_attn = _stack_attention(attn)  # [gen_steps, n_layers, n_heads, full_seq_len]
        except Exception as e:
            logger.warning(f"Failed to stack attention: {e}. Returning zeros.")
            return np.zeros(n, dtype=np.float64)

        # For each passage, compute mean attention across all generation steps,
        # layers, and heads within its token span. Then compute variance.
        scores = np.zeros(n, dtype=np.float64)
        for i, (start, end) in enumerate(boundaries):
            if start >= all_attn.shape[-1]:
                continue
            end = min(end, all_attn.shape[-1])
            if end <= start:
                continue

            # Attention to this passage's tokens
            passage_attn = all_attn[..., start:end]  # [gen, layers, heads, span]

            # Mean attention per generation step → [gen]
            mean_per_step = passage_attn.mean(axis=(1, 2, 3))  # mean over layers, heads, tokens

            # Variance across generation steps
            scores[i] = float(np.var(mean_per_step))

        # Normalize
        smax = scores.max()
        if smax > 0:
            scores = scores / smax

        return scores

    def score_batch_synthetic(
        self,
        clean_passages: List[str],
        poison_passages: List[str],
    ) -> bool:
        """Quick validation: check if attention variance differs for clean vs poison.

        Uses a fixed generic query. Returns True if poison passages have
        higher attention variance on average.

        This is a DEVELOPMENT-ONLY method for verifying signal direction.
        """
        query = "What is the answer to this question?"
        all_passages = clean_passages + poison_passages
        n_clean = len(clean_passages)

        scores = self.score(query, all_passages)
        clean_scores = scores[:n_clean]
        poison_scores = scores[n_clean:]

        clean_mean = np.mean(clean_scores)
        poison_mean = np.mean(poison_scores)

        logger.info(
            f"AttentionVariance sanity check: "
            f"clean_mean={clean_mean:.4f}, poison_mean={poison_mean:.4f}"
        )
        return poison_mean > clean_mean


def _stack_attention(attn_list: List) -> np.ndarray:
    """Stack list of [n_layers, n_heads, 1, full_seq_len] into a single array.

    Returns: [gen_steps, n_layers, n_heads, full_seq_len]
    """
    stacked = []
    for step_attn in attn_list:
        # step_attn: [n_layers, n_heads, 1, full_seq_len]
        stacked.append(step_attn.squeeze(2).numpy())  # [n_layers, n_heads, full_seq_len]
    return np.stack(stacked, axis=0)  # [gen_steps, n_layers, n_heads, full_seq_len]
