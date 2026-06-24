"""Test DRSScorer — Directional Relative Shift with synthetic embeddings.

Tests the placeholder DRS implementation with a mock embedder.
Actual formula calibration against the NeurIPS'24 paper is deferred to M2.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from scorers.drs import DRSScorer
from utils.metrics import compute_auroc, verify_score_direction


class MockEmbedderDRS:
    """Mock embedder that returns controlled embeddings for DRS testing.

    Query at origin. Clean passages near query. Poison passages shifted
    in a consistent direction (the "attack-prone" direction).
    """

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.embedding_dim = dim
        self._loaded = True

        # Fixed attack direction
        self.attack_direction = self.rng.randn(dim)
        self.attack_direction = self.attack_direction / np.linalg.norm(self.attack_direction)

    def encode_query(self, query):
        """Query embedding near origin."""
        return self.rng.randn(self.dim).astype(np.float32) * 0.1

    def encode_passages(self, passages):
        """Clean passages have small shifts; poison passages have large shifts
        along the attack direction."""
        n = len(passages)
        embeddings = np.zeros((n, self.dim), dtype=np.float32)

        for i, text in enumerate(passages):
            if "poison" in text.lower():
                # Strong shift along attack direction
                shift_magnitude = self.rng.uniform(3.0, 8.0)
                embeddings[i] = self.attack_direction * shift_magnitude
                # Small orthogonal noise
                noise = self.rng.randn(self.dim) * 0.3
                noise -= np.dot(noise, self.attack_direction) * self.attack_direction
                embeddings[i] += noise
            else:
                # Small random shift
                embeddings[i] = self.rng.randn(self.dim) * 0.5

        return embeddings.astype(np.float32)

    def is_loaded(self):
        return True


class TestDRSScorer:
    """Test DRS scorer with mock embedder."""

    def test_name(self):
        scorer = DRSScorer()
        assert scorer.name == "drs"

    def test_requires_embedder(self):
        scorer = DRSScorer()
        with pytest.raises(RuntimeError, match="requires an EmbedderWrapper"):
            scorer.score("query", ["a", "b"])

    def test_score_higher_for_poison(self):
        """Poison passages (shifted along attack direction) should score higher."""
        embedder = MockEmbedderDRS(dim=32)
        scorer = DRSScorer(embedder=embedder)

        passages = [
            "clean passage one",
            "poison passage with fake info",
            "clean passage two",
            "poison passage with wrong facts",
            "clean passage three",
        ]

        scores = scorer.score("test query", passages)
        assert len(scores) == len(passages)

        poison_idx = [1, 3]
        clean_idx = [0, 2, 4]
        assert scores[poison_idx].mean() > scores[clean_idx].mean(), (
            f"Poison DRS ({scores[poison_idx].mean():.4f}) should exceed "
            f"clean DRS ({scores[clean_idx].mean():.4f})"
        )

    def test_auroc_direction_correct(self):
        """Verify AUROC performance with clean DRS separation."""
        embedder = MockEmbedderDRS(dim=32)
        scorer = DRSScorer(embedder=embedder)

        n_clean = 25
        n_poison = 15
        passages = (
            [f"clean doc {i}" for i in range(n_clean)] +
            [f"poison doc {i}" for i in range(n_poison)]
        )

        scores = scorer.score("test query", passages)
        labels = np.array([0] * n_clean + [1] * n_poison)
        auroc = compute_auroc(scores, labels)

        assert auroc > 0.6, f"Expected AUROC > 0.6 with clear DRS separation, got {auroc:.4f}"
        assert verify_score_direction(scores, labels, "drs")

    def test_empty_passages(self):
        embedder = MockEmbedderDRS(dim=32)
        scorer = DRSScorer(embedder=embedder)
        scores = scorer.score("query", [])
        assert len(scores) == 0

    def test_single_passage(self):
        embedder = MockEmbedderDRS(dim=32)
        scorer = DRSScorer(embedder=embedder)
        scores = scorer.score("query", ["single passage"])
        assert len(scores) == 1


class TestDRSFormulaNotes:
    """Documentation tests for DRS calibration TODO.

    These tests verify the PLACEHOLDER implementation produces
    expected behavior but are NOT validation of DRS accuracy.
    """

    def test_placeholder_projection_positive(self):
        """Projection onto attack direction should be non-negative."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder)

        passages = [f"passage {i}" for i in range(10)]
        scores = scorer.score("query", passages)

        # All scores should be non-negative (|projection|)
        assert np.all(scores >= 0)

    def test_placeholder_normalized(self):
        """Scores should be normalized to [0, 1] range (approximately)."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder)

        passages = [f"passage {i}" for i in range(20)]
        scores = scorer.score("query", passages)

        assert 0.0 <= scores.max() <= 1.01, f"Max score {scores.max()} out of [0, 1]"
        assert 0.0 <= scores.min() <= 1.01
