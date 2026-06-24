"""Test AttentionVarianceScorer — structure and interface only.

Full functional testing requires a real LM with output_attentions=True.
These tests verify:
1. The scorer has the correct name and interface
2. It handles edge cases (empty passages, generator not loaded)
3. The attention stacking utility works correctly
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock

from scorers.attention_variance import AttentionVarianceScorer, _stack_attention


class TestAttentionVarianceScorer:
    """Interface and edge-case tests for AttentionVarianceScorer."""

    def test_name(self):
        scorer = AttentionVarianceScorer()
        assert scorer.name == "attention_variance"

    def test_requires_generator(self):
        scorer = AttentionVarianceScorer()
        with pytest.raises(RuntimeError, match="requires a GeneratorWrapper"):
            scorer.score("query", ["passage 1"])

    def test_empty_passages(self):
        mock_gen = Mock()
        scorer = AttentionVarianceScorer(generator=mock_gen)

        scores = scorer.score("query", [])
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 0

    def test_attention_extraction_failure(self):
        """If generator fails to return attention, scorer should return zeros."""
        mock_gen = Mock()
        mock_gen.generate_with_attention.side_effect = RuntimeError("No attention available")

        scorer = AttentionVarianceScorer(generator=mock_gen)
        passages = ["passage 1", "passage 2", "passage 3"]

        scores = scorer.score("query", passages)
        assert len(scores) == len(passages)
        assert np.all(scores == 0.0), "Failed attention should return zeros"

    def test_attention_weights_none(self):
        """If attention_weights is empty, should return zeros."""
        mock_gen = Mock()
        mock_gen.generate_with_attention.return_value = {
            "generated_text": "test",
            "attention_weights": [],  # empty
            "passage_boundaries": [(0, 5), (5, 10), (10, 15)],
            "input_len": 15,
        }

        scorer = AttentionVarianceScorer(generator=mock_gen)
        passages = ["a", "b", "c"]
        scores = scorer.score("query", passages)

        assert len(scores) == len(passages)
        assert np.all(scores == 0.0)


class TestAttentionStacking:
    """Test the _stack_attention utility."""

    def test_stack_attention(self):
        """Verify stacking of attention tensors."""
        n_layers = 4
        n_heads = 8
        seq_len = 20
        n_gen_steps = 5

        # Build mock attention list
        attn_list = []
        for _ in range(n_gen_steps):
            # [n_layers, n_heads, 1, full_seq_len] per step
            step = torch.randn(n_layers, n_heads, 1, seq_len)
            attn_list.append(step)

        stacked = _stack_attention(attn_list)
        assert stacked.shape == (n_gen_steps, n_layers, n_heads, seq_len)

    def test_stack_single_step(self):
        """Single generation step should work."""
        step = torch.randn(12, 16, 1, 50)
        stacked = _stack_attention([step])
        assert stacked.shape == (1, 12, 16, 50)


@pytest.mark.skip(reason="Requires real LM with output_attentions=True — run on GPU server")
class TestAttentionVarianceFunctional:
    """Functional tests requiring a real generator — skipped in CI."""

    def test_poison_higher_variance(self):
        """Verify poison passages get higher attention variance."""
        pass  # To be implemented when GPU is available
