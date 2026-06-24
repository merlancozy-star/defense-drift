"""Test PerplexityScorer — AUROC direction with synthetic data.

Cannot test actual PPL computation without loading a real LM,
but we verify the scoring interface and AUROC direction convention.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from scorers.perplexity import PerplexityScorer
from utils.metrics import compute_auroc, verify_score_direction


class TestPerplexityScorer:
    """Test PerplexityScorer behavior with a mocked generator."""

    def test_name(self):
        scorer = PerplexityScorer()
        assert scorer.name == "perplexity"

    def test_requires_generator(self):
        """Scorer should raise if generator is not set."""
        scorer = PerplexityScorer()
        with pytest.raises(RuntimeError, match="requires a GeneratorWrapper"):
            scorer.score("test query", ["passage 1", "passage 2"])

    def test_score_returns_correct_shape(self):
        """Score should return array of same length as passages."""
        mock_gen = Mock()
        # Poison passages have higher PPL (more anomalous)
        mock_gen.compute_perplexity.return_value = [10.5, 12.3, 8.9, 25.7, 11.1]

        scorer = PerplexityScorer(generator=mock_gen)
        passages = ["clean text", "poison text", "clean text", "another poison", "clean"]
        scores = scorer.score("test query", passages)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(passages)
        assert scores.dtype == np.float64

    def test_score_direction_higher_is_poison(self):
        """Verify that higher PPL = more suspicious (correct direction)."""
        mock_gen = Mock()
        # Synthetic: poison passages have notably higher PPL
        mock_gen.compute_perplexity.return_value = [
            8.0,   # clean
            45.0,  # poison (anomalously high PPL)
            9.0,   # clean
            52.0,  # poison
            7.5,   # clean
            38.0,  # poison
            10.0,  # clean
            60.0,  # poison
            8.5,   # clean
            41.0,  # poison
        ]

        scorer = PerplexityScorer(generator=mock_gen)
        passages = [f"passage {i}" for i in range(10)]
        scores = scorer.score("query", passages)

        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # 0=clean, 1=poison
        auroc = compute_auroc(scores, labels)

        assert auroc > 0.8, f"Expected AUROC > 0.8 with clear PPL separation, got {auroc:.4f}"
        assert verify_score_direction(scores, labels, "perplexity")

    def test_batch_score(self):
        """Batch scoring should work correctly."""
        mock_gen = Mock()
        mock_gen.compute_perplexity.side_effect = [
            [10.0, 20.0],
            [15.0, 25.0],
            [12.0, 18.0],
        ]

        scorer = PerplexityScorer(generator=mock_gen)
        queries = ["q1", "q2", "q3"]
        passages_list = [["a", "b"], ["c", "d"], ["e", "f"]]

        results = scorer.batch_score(queries, passages_list)
        assert len(results) == 3
        for r in results:
            assert len(r) == 2


class TestPPLAUROCConvention:
    """Standalone tests for the 'higher = poison' convention."""

    def test_auroc_perfect_separation(self):
        """Perfect separation should give AUROC = 1.0."""
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        auroc = compute_auroc(scores, labels)
        assert auroc == 1.0

    def test_auroc_random(self):
        """Random scores should give AUROC ≈ 0.5."""
        rng = np.random.RandomState(42)
        scores = rng.rand(100)
        labels = rng.randint(0, 2, 100)
        auroc = compute_auroc(scores, labels)
        assert 0.3 < auroc < 0.7, f"Expected near-random AUROC, got {auroc:.4f}"

    def test_auroc_reversed_direction(self):
        """If clean has higher scores, AUROC should be < 0.5."""
        scores = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.15])  # clean higher
        labels = np.array([0, 0, 0, 1, 1, 1])                  # 1 = poison
        auroc = compute_auroc(scores, labels)
        assert auroc < 0.5, f"Expected AUROC < 0.5 for reversed direction, got {auroc:.4f}"
