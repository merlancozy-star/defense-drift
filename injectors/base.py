"""Base class for poison injectors.

Each injector takes a set of clean passages and injects poison passages
according to a specified attack strategy and ratio.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from data.schemas import Passage


class BaseInjector(ABC):
    """Abstract base for poison passage injectors.

    Subclasses implement craft() to generate poison passages.
    The inject() method handles ratio-based mixing with clean passages.
    """

    @property
    @abstractmethod
    def attack_name(self) -> str:
        """Unique attack identifier: 'poisonedrag' or 'corruptrag'."""
        ...

    @abstractmethod
    def craft(
        self,
        query: str,
        clean_passages: List[Passage],
        n_poison: int,
    ) -> List[Passage]:
        """Generate n_poison poison passages for the given query.

        Args:
            query: The query to attack.
            clean_passages: Retrieved clean passages (for context/grounding).
            n_poison: Number of poison passages to generate.

        Returns:
            List of Passage objects with is_poison=True.
        """
        ...

    def inject(
        self,
        query: str,
        clean_passages: List[Passage],
        poison_ratio: float,
        top_k: int,
    ) -> List[Passage]:
        """Inject poison passages at the specified ratio.

        Args:
            query: The query.
            clean_passages: Retrieved clean passages.
            poison_ratio: Fraction of passages that should be poison [0, 1].
            top_k: Total number of passages to return (top_k after injection).

        Returns:
            Mixed list of clean + poison passages, with is_poison flags set.
            The length is top_k.
        """
        n_poison = max(1, int(top_k * poison_ratio))
        n_clean = top_k - n_poison

        # Truncate clean passages
        clean_to_keep = clean_passages[:n_clean]

        # Generate poison passages
        poison_passages = self.craft(query, clean_passages, n_poison)
        for p in poison_passages:
            p.is_poison = True
            p.poison_attack = self.attack_name

        # Mix: poison injected according to injection_position strategy
        mixed = self._interleave(clean_to_keep, poison_passages)

        return mixed

    def _interleave(
        self,
        clean: List[Passage],
        poison: List[Passage],
    ) -> List[Passage]:
        """Interleave poison into clean at the configured position.

        Default: inject poison at top (most visible to retriever).
        Override in subclasses for different strategies.
        """
        # Default: poison first, then clean (worst-case for defense)
        return poison + clean
