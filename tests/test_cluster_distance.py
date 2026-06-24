"""Test ClusterDistanceScorer — 2-means outlier distance with synthetic embeddings.

Does NOT require a real embedder — uses a mock that returns synthetic
embedding vectors.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from scorers.cluster_distance import ClusterDistanceScorer
from utils.metrics import compute_auroc, verify_score_direction


class MockEmbedder:
    """Mock embedder that returns synthetic embeddings.

    Clean passages: clustered tightly around [0, 0]
    Poison passages: scattered far from clean cluster
    """

    def __init__(self, dim: int = 128, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.embedding_dim = dim
        self._loaded = True

    def encode_passages(self, passages):
        """Return embeddings that separate clean from poison."""
        n = len(passages)
        embeddings = np.zeros((n, self.dim), dtype=np.float32)

        for i, text in enumerate(passages):
            if "poison" in text.lower():
                # Poison: far from origin in random direction
                direction = self.rng.randn(self.dim)
                direction = direction / np.linalg.norm(direction)
                embeddings[i] = direction * self.rng.uniform(5.0, 15.0)
            else:
                # Clean: tight cluster near origin
                embeddings[i] = self.rng.randn(self.dim) * 0.5

        return embeddings

    def encode_query(self, query):
        return self.rng.randn(self.dim).astype(np.float32) * 0.1

    def is_loaded(self):
        return True


class TestClusterDistanceScorer:
    """Test ClusterDistanceScorer with a mock embedder."""

    def test_name(self):
        scorer = ClusterDistanceScorer()
        assert scorer.name == "cluster_distance"

    def test_requires_embedder(self):
        scorer = ClusterDistanceScorer()
        with pytest.raises(RuntimeError, match="requires an EmbedderWrapper"):
            scorer.score("query", ["a", "b"])

    def test_score_higher_for_outliers(self):
        """Poison passages (far from cluster) should get higher scores."""
        embedder = MockEmbedder(dim=16)
        scorer = ClusterDistanceScorer(embedder=embedder)

        passages = [
            "clean passage about science",
            "clean passage about history",
            "clean passage about technology",
            "poison passage with fake facts",
            "clean passage about art",
            "poison passage with wrong data",
            "clean passage about music",
            "poison passage with false claims",
            "clean passage about literature",
            "poison passage with contradictions",
        ]

        scores = scorer.score("query", passages)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(passages)

        # Poison passages should have higher scores (more anomalous)
        poison_idx = [i for i, p in enumerate(passages) if "poison" in p.lower()]
        clean_idx = [i for i, p in enumerate(passages) if "poison" not in p.lower()]

        poison_mean = scores[poison_idx].mean()
        clean_mean = scores[clean_idx].mean()

        assert poison_mean > clean_mean, (
            f"Expected poison_mean ({poison_mean:.4f}) > clean_mean ({clean_mean:.4f})"
        )

    def test_auroc_direction_correct(self):
        """Verify AUROC > 0.5 with clear cluster separation."""
        embedder = MockEmbedder(dim=16)
        scorer = ClusterDistanceScorer(embedder=embedder)

        n_clean = 30
        n_poison = 20
        passages = (
            [f"clean passage {i}" for i in range(n_clean)] +
            [f"poison passage {i}" for i in range(n_poison)]
        )

        scores = scorer.score("query", passages)
        labels = np.array([0] * n_clean + [1] * n_poison)

        auroc = compute_auroc(scores, labels)
        assert auroc > 0.5, f"Expected AUROC > 0.5, got {auroc:.4f}"
        assert verify_score_direction(scores, labels, "cluster_distance")

    def test_too_few_passages(self):
        """Should handle < 3 passages gracefully."""
        embedder = MockEmbedder(dim=16)
        scorer = ClusterDistanceScorer(embedder=embedder)

        scores = scorer.score("query", ["only one"])
        assert len(scores) == 1
        assert scores[0] == 0.0
