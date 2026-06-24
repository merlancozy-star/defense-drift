"""Base class for all separability signal scorers.

Each scorer takes a list of passages (and optionally a query) and returns
a numeric score for each passage. The score direction MUST be:
  HIGHER = more likely to be poison

This convention is enforced by verify_score_direction() in utils/metrics.py.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseScorer(ABC):
    """Abstract base for all signal scorers.

    Subclasses must implement:
      - name: str — unique signal identifier
      - score(query, passages) → np.ndarray

    The returned array must have shape (len(passages),) with higher values
    indicating more suspicious / poison-like passages.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique signal name matching config keys."""
        ...

    @abstractmethod
    def score(
        self,
        query: str,
        passages: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Score each passage for how poison-like it is.

        Args:
            query: The query/question text.
            passages: List of retrieved passage texts.

        Returns:
            np.ndarray of shape (len(passages),) — higher = more suspicious.
        """
        ...

    @property
    def needs_calibration(self) -> bool:
        """Whether this scorer needs a calibration phase before scoring.

        Override to return True for scorers that need to estimate
        clean-distribution parameters (e.g., DRS via PCA).
        """
        return False

    def calibrate(self, clean_embeddings: np.ndarray):
        """Calibrate scorer on clean passage embeddings.

        Called once per domain before scoring. Default is no-op.
        Override for scorers that need distribution estimation.

        Args:
            clean_embeddings: [N, D] array of clean passage embeddings.
        """
        pass

    def batch_score(
        self,
        queries: List[str],
        passages_list: List[List[str]],
        **kwargs,
    ) -> List[np.ndarray]:
        """Batch scoring for efficiency. Override if batch processing is possible.

        Default implementation loops over score().
        """
        return [self.score(q, p, **kwargs) for q, p in zip(queries, passages_list)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
