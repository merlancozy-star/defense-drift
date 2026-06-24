"""CorruptRAG Injector — stronger, more covert poison attack.

CorruptRAG: unlike PoisonedRAG (which generates entirely new fake passages),
CorruptRAG takes REAL clean passages and CORRUPTS them by subtly altering
key facts, numbers, entities, or relationships while preserving the overall
structure and fluency.

This is harder to detect because:
1. The passage structure and most content is genuine
2. Subtle corruptions (e.g., "increases" → "decreases") are hard to spot
3. Perplexity and embedding-based signals are less effective (passage is
   mostly natural text)

Expected: CorruptRAG should produce LOWER separability (more collapse)
than PoisonedRAG, because the signals are less discriminative.
"""

import random
import re
from typing import List, Optional
import copy

from injectors.base import BaseInjector
from data.schemas import Passage
from models.generator import GeneratorWrapper


class CorruptRAGInjector(BaseInjector):
    """CorruptRAG: corrupt real passages rather than generating new ones.

    Corruption strategies:
    1. NUMERICAL: Flip numbers (e.g., "30%" → "70%", "increased by 5" → "decreased by 5")
    2. ENTITY: Swap key entities with semantically similar but factually wrong ones
    3. RELATION: Invert relational predicates (increases↔decreases, causes↔prevents)
    4. NEGATION: Insert/remove negation (was → was not)
    """

    # Pairs for flipping
    FLIP_PAIRS = [
        ("increase", "decrease"),
        ("increases", "decreases"),
        ("increased", "decreased"),
        ("increasing", "decreasing"),
        ("higher", "lower"),
        ("more", "less"),
        ("above", "below"),
        ("positive", "negative"),
        ("improve", "worsen"),
        ("improved", "worsened"),
        ("gain", "loss"),
        ("gains", "losses"),
        ("benefit", "harm"),
        ("beneficial", "harmful"),
        ("effective", "ineffective"),
        ("causes", "prevents"),
        ("caused", "prevented"),
        ("associated with", "unrelated to"),
        ("correlated with", "independent of"),
    ]

    NEGATION_INSERT = [
        (r"\b(is|are|was|were)\b", r"\1 not"),
        (r"\b(has|have|had)\b", r"\1 not"),
        (r"\b(does|do|did)\b", r"\1 not"),
    ]

    def __init__(
        self,
        corrupt_ratio: float = 0.3,
        preserve_structure: bool = True,
        generator: Optional[GeneratorWrapper] = None,
        random_seed: int = 42,
    ):
        """
        Args:
            corrupt_ratio: Fraction of tokens/phrases to corrupt (0.0-1.0).
            preserve_structure: If True, keep sentence boundaries and structure intact.
            generator: Optional generator for rephrasing corrupted passages.
            random_seed: For reproducibility.
        """
        self.corrupt_ratio = corrupt_ratio
        self.preserve_structure = preserve_structure
        self._generator = generator
        self.rng = random.Random(random_seed)

    @property
    def attack_name(self) -> str:
        return "corruptrag"

    def set_generator(self, generator: GeneratorWrapper):
        self._generator = generator

    def craft(
        self,
        query: str,
        clean_passages: List[Passage],
        n_poison: int,
    ) -> List[Passage]:
        """Generate n_poison corrupted passages from clean ones.

        Strategy:
        1. Select n_poison clean passages to corrupt
        2. Apply multiple corruption strategies to each
        3. The corrupted version is the poison passage
        """
        # Use real passages as the base for corruption
        source_passages = clean_passages[:n_poison]
        if len(source_passages) < n_poison:
            # Cycle through clean passages if not enough
            source_passages = (source_passages * (n_poison // max(1, len(source_passages)) + 1))[:n_poison]

        poison_passages = []
        for i, src in enumerate(source_passages):
            corrupted_text = self._corrupt_text(src.text)
            poison_passages.append(Passage(
                id=f"poison_{self.attack_name}_{i}_from_{src.id}",
                text=corrupted_text,
                is_poison=True,
                poison_attack=self.attack_name,
                retrieval_score=src.retrieval_score,  # Preserve retrieval score
            ))

        return poison_passages

    def _corrupt_text(self, text: str) -> str:
        """Apply corruption strategies to a single passage."""
        corrupted = text

        # Strategy 1: Flip directional words
        corrupted = self._flip_directional(corrupted)

        # Strategy 2: Corrupt numbers
        corrupted = self._corrupt_numbers(corrupted)

        # Strategy 3: Insert/remove negation (probabilistic)
        if self.rng.random() < self.corrupt_ratio:
            corrupted = self._toggle_negation(corrupted)

        return corrupted

    def _flip_directional(self, text: str) -> str:
        """Flip directional/relational word pairs."""
        words = text.split()
        n_to_flip = max(1, int(len(words) * self.corrupt_ratio))

        # Find candidate positions
        candidate_positions = []
        for i, word in enumerate(words):
            clean_word = word.lower().strip(".,;:!?()\"'")
            for a, b in self.FLIP_PAIRS:
                if clean_word == a:
                    candidate_positions.append((i, a, b))
                    break

        # Randomly select positions to flip
        self.rng.shuffle(candidate_positions)
        flipped = set()
        for pos, original, replacement in candidate_positions[:n_to_flip]:
            if pos not in flipped:
                # Preserve capitalization
                if words[pos][0].isupper():
                    replacement = replacement.capitalize()
                # Preserve trailing punctuation
                suffix = ""
                for ch in reversed(words[pos]):
                    if ch in ".,;:!?)\"'":
                        suffix = ch + suffix
                    else:
                        break
                if suffix:
                    words[pos] = replacement + suffix
                else:
                    words[pos] = replacement
                flipped.add(pos)

        return " ".join(words)

    def _corrupt_numbers(self, text: str) -> str:
        """Corrupt numerical values in the text.

        Strategy: multiply numbers by a random factor (0.1-10x) or flip sign.
        """
        def _corrupt_match(match):
            num_str = match.group(0)
            try:
                num = float(num_str.replace(",", ""))
                if self.rng.random() < self.corrupt_ratio:
                    # Corrupt this number
                    factor = self.rng.choice([0.1, 0.5, 2.0, 3.0, 10.0, -1.0])
                    new_num = num * factor
                    if new_num == int(new_num):
                        return str(int(new_num))
                    else:
                        return f"{new_num:.1f}"
                return num_str
            except ValueError:
                return num_str

        # Match numbers (including decimals and commas)
        return re.sub(r'\b\d+(?:[,.]\d+)*\b', _corrupt_match, text)

    def _toggle_negation(self, text: str) -> str:
        """Insert or remove negation from the text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return text

        # Pick a random sentence to negate
        idx = self.rng.randrange(len(sentences))
        sent = sentences[idx]
        for pattern, replacement in self.NEGATION_INSERT:
            # 50% chance: insert negation
            if self.rng.random() < 0.5:
                sent = re.sub(pattern, replacement, sent, count=1)
                break
        sentences[idx] = sent
        return " ".join(sentences)
