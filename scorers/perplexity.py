"""PerplexityScorer — token NLL → PPL via generator LM.

Signal: a passage with unusually high perplexity under a general-domain LM
is likely to be anomalous (poison often contains out-of-distribution text
patterns, fabricated facts, or unnatural phrasing).

HIGHER PPL → more suspicious (aligned with "higher = poison" convention).

Reference: RAGuard (uses PPL + similarity), general PPL-based anomaly detection.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from scorers.base import BaseScorer

if TYPE_CHECKING:
    from models.generator import GeneratorWrapper


class PerplexityScorer(BaseScorer):
    """Score passages by perplexity under the generator LM.

    Perplexity = exp(token-level cross-entropy).
    Higher PPL means the passage is less "natural" under the model's distribution,
    which is a weak signal for poison detection.
    """

    def __init__(self, generator: Optional[GeneratorWrapper] = None):
        """
        Args:
            generator: A loaded GeneratorWrapper. If None, will be set later
                       via set_generator().
        """
        self._generator = generator

    @property
    def name(self) -> str:
        return "perplexity"

    def set_generator(self, generator: GeneratorWrapper):
        """Set or replace the generator (used for lazy loading)."""
        self._generator = generator

    def score(
        self,
        query: str,
        passages: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Compute PPL for each passage.

        The query is NOT used by this scorer — PPL is computed on the
        passage text alone (passage-level anomaly).

        Args:
            query: Query text (unused for PPL scoring).
            passages: List of passage texts.

        Returns:
            np.ndarray of PPL values, shape (len(passages),).
        """
        if self._generator is None:
            raise RuntimeError(
                "PerplexityScorer requires a GeneratorWrapper. "
                "Call set_generator() before scoring."
            )

        batch_size = kwargs.get("batch_size", 8)
        ppls = self._generator.compute_perplexity(passages, batch_size=batch_size)
        return np.array(ppls, dtype=np.float64)

    # Note: PPL direction is naturally "higher = more suspicious",
    # which matches the BaseScorer convention. No sign flip needed.
