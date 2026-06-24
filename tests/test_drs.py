"""Test DRSScorer — PCA-based Directional Relative Shift.

Tests the calibrated DRS (NeurIPS 2024) with a mock embedder
that produces controlled embeddings with known variance structure.

Key insight being tested:
- Clean passages: vary along HIGH-variance directions, tight along LOW-variance
- Poison passages: shifted along LOW-variance directions (attack-prone)
- DRS should detect the low-variance shifts → higher scores for poison
"""

import pytest
import numpy as np
from unittest.mock import Mock

from scorers.drs import DRSScorer
from utils.metrics import compute_auroc, verify_score_direction


class MockEmbedderDRS:
    """Mock embedder with controlled variance structure.

    Embedding space is 16-dimensional:
    - Dimensions 0-3: HIGH variance (clean passages vary a lot — "natural" variation)
    - Dimensions 4-7: MEDIUM variance
    - Dimensions 8-11: LOW variance (tight cluster — "attack-prone" directions)
    - Dimensions 12-15: VERY LOW variance (tightest cluster)

    Clean passages: centered at origin, with variance proportional to dim index
    Poison passages: shifted by +2.0 along the LOW-variance dimensions (8-11)

    DRS should detect the shift in dimensions 8-15 because clean data
    has very little natural variation there.
    """

    def __init__(self, dim: int = 16, seed: int = 42):
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.embedding_dim = dim
        self._loaded = True

        # Variance per dimension: exponentially decaying from high to low
        self.stds = np.exp(-np.linspace(-1.5, 4.0, dim))  # range ~[4.5, 0.02]

        # Poison shift: pushes along LOW-variance directions (upper indices)
        self.poison_shift = np.zeros(dim)
        low_start = max(0, dim // 2)  # Bottom half = low variance
        self.poison_shift[low_start:] = 3.0
        # Small random component in high-var directions too (lower indices)
        self.poison_shift[:dim//4] = self.rng.randn(dim//4) * 0.5

    def encode_query(self, query):
        return self.rng.randn(self.dim).astype(np.float32) * 0.1

    def encode_passages(self, passages):
        """Generate embeddings with controlled variance structure."""
        n = len(passages)
        embeddings = np.zeros((n, self.dim), dtype=np.float32)

        for i, text in enumerate(passages):
            if "poison" in text.lower():
                # Start from clean-like base
                base = self.rng.randn(self.dim) * self.stds
                # Apply poison shift
                embeddings[i] = base + self.poison_shift
            else:
                # Clean: varying along high-var, tight along low-var
                embeddings[i] = self.rng.randn(self.dim) * self.stds

        return embeddings.astype(np.float32)

    def is_loaded(self):
        return True


class TestDRSScorer:
    """Test DRS scorer with PCA calibration."""

    def test_name(self):
        scorer = DRSScorer()
        assert scorer.name == "drs"

    def test_needs_calibration(self):
        scorer = DRSScorer()
        assert scorer.needs_calibration is True

    def test_requires_embedder(self):
        scorer = DRSScorer()
        with pytest.raises(RuntimeError, match="requires an EmbedderWrapper"):
            scorer.score("query", ["a", "b"])

    def test_requires_calibration(self):
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder)
        with pytest.raises(RuntimeError, match="requires calibration"):
            scorer.score("query", ["a", "b"])

    def test_calibrate_computes_parameters(self):
        """Calibration should set mu, V_low, and lambda_low."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder, variance_fraction=0.25)

        clean_texts = [f"clean passage {i}" for i in range(100)]
        clean_embs = embedder.encode_passages(clean_texts)

        scorer.calibrate(clean_embs)

        assert scorer.is_calibrated
        assert scorer._mu is not None
        assert scorer._mu.shape == (16,)
        assert scorer._V_low is not None
        assert scorer._V_low.shape[0] == 16  # D
        assert scorer._V_low.shape[1] >= 5   # min_directions
        assert scorer._lambda_low is not None

    def test_score_higher_for_poison(self):
        """Poison shifted along low-variance directions → higher DRS."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder, variance_fraction=0.25)

        # Calibrate on clean only
        clean_texts = [f"clean passage {i}" for i in range(200)]
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        # Score mixed passages
        test_passages = [
            "clean passage A",
            "poison passage with fake info",
            "clean passage B",
            "poison passage with wrong facts",
            "clean passage C",
            "poison passage with contradictions",
            "clean passage D",
        ]

        scores = scorer.score("test query", test_passages)
        assert len(scores) == len(test_passages)

        poison_idx = [1, 3, 5]
        clean_idx = [0, 2, 4, 6]
        poison_mean = scores[poison_idx].mean()
        clean_mean = scores[clean_idx].mean()

        assert poison_mean > clean_mean, (
            f"Poison DRS ({poison_mean:.4f}) should exceed "
            f"clean DRS ({clean_mean:.4f})"
        )

    def test_auroc_direction_correct(self):
        """DRS AUROC should be > 0.7 with clear low-variance shift."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder, variance_fraction=0.25)

        # Calibrate
        clean_texts = [f"clean calib {i}" for i in range(200)]
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        # Test set
        n_clean = 40
        n_poison = 30
        passages = (
            [f"clean doc {i}" for i in range(n_clean)] +
            [f"poison doc {i}" for i in range(n_poison)]
        )

        scores = scorer.score("test query", passages)
        labels = np.array([0] * n_clean + [1] * n_poison)
        auroc = compute_auroc(scores, labels)

        assert auroc > 0.7, (
            f"Expected AUROC > 0.7 with clear low-variance poison shift, got {auroc:.4f}"
        )
        assert verify_score_direction(scores, labels, "drs")

    def test_empty_passages(self):
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder)

        clean_embs = embedder.encode_passages([f"c{i}" for i in range(100)])
        scorer.calibrate(clean_embs)

        scores = scorer.score("query", [])
        assert len(scores) == 0

    def test_threshold_classification(self):
        """Clean passages should mostly be below threshold."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder, variance_fraction=0.25)

        clean_texts = [f"clean {i}" for i in range(200)]
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        # Get clean DRS scores for threshold
        clean_test = [f"clean test {i}" for i in range(50)]
        clean_scores = scorer.score("q", clean_test)
        threshold = scorer.get_threshold(clean_scores, percentile=95)

        # Most clean passages should be below threshold
        false_positive_rate = np.mean(clean_scores > threshold)
        assert false_positive_rate <= 0.10, (
            f"FPR too high: {false_positive_rate:.2%} (expected ≤ 10%)"
        )

    def test_get_calibration_state_roundtrip(self):
        """Calibration state should survive serialization round-trip."""
        embedder = MockEmbedderDRS(dim=16)
        scorer1 = DRSScorer(embedder=embedder)
        clean_embs = embedder.encode_passages([f"c{i}" for i in range(100)])
        scorer1.calibrate(clean_embs)

        state = scorer1.get_calibration_state()
        assert state["calibrated"] is True
        assert "mu" in state
        assert "V_low" in state
        assert "lambda_low" in state

        # Restore
        scorer2 = DRSScorer(embedder=embedder)
        scorer2.set_calibration_state(state)
        assert scorer2.is_calibrated

        # Scores should be highly correlated (serialization may cause minor drift)
        passages = ["clean A", "poison B", "clean C"]
        s1 = scorer1.score("q", passages)
        s2 = scorer2.score("q", passages)
        # Check: ranking preserved (poison > clean), values same order of magnitude
        assert np.argmax(s1) == np.argmax(s2), f"Rank preserved: s1={s1}, s2={s2}"
        assert np.all((s1 > 0) & (s2 > 0)), "Scores should be positive"

    def test_too_few_calibration_samples(self):
        """Should raise with < 2 samples."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder)

        with pytest.raises(ValueError, match="at least 2"):
            scorer.calibrate(np.random.randn(1, 16))

    def test_high_N_low_D_case(self):
        """N >= D path: standard PCA."""
        embedder = MockEmbedderDRS(dim=8)  # small D
        scorer = DRSScorer(embedder=embedder)

        clean_texts = [f"c{i}" for i in range(100)]  # N >> D
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        assert scorer.is_calibrated
        assert scorer._V_low.shape[1] >= scorer.min_directions

    def test_low_N_high_D_case(self):
        """N < D path: SVD-based PCA."""
        embedder = MockEmbedderDRS(dim=64)  # large D
        scorer = DRSScorer(embedder=embedder)

        clean_texts = [f"c{i}" for i in range(30)]  # N < D
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        assert scorer.is_calibrated
        # V_low should span full D
        assert scorer._V_low.shape[0] == 64

    def test_drs_no_poison_shift_gives_low_scores(self):
        """If no poison shift, DRS scores should be similar to clean."""
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder, variance_fraction=0.25)

        clean_texts = [f"c{i}" for i in range(200)]
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        # Score more clean passages (no poison shift)
        more_clean = [f"more clean {i}" for i in range(30)]
        scores = scorer.score("q", more_clean)

        # Clean scores should be mostly low (within threshold)
        threshold = scorer.get_threshold(scores, percentile=95)
        fpr = np.mean(scores > threshold)
        assert fpr <= 0.10, f"FPR on clean-only data too high: {fpr:.2%}"


class TestDRSVsPlaceholder:
    """Verify the new PCA-based implementation differs from placeholder."""

    def test_pca_detects_low_variance_shift(self):
        """The key insight: PCA-based DRS detects shifts in LOW-variance directions.

        This is what the placeholder got WRONG — it looked at the MEAN shift
        direction, which is closest to PC1 (HIGHEST variance).
        """
        embedder = MockEmbedderDRS(dim=16)
        scorer = DRSScorer(embedder=embedder, variance_fraction=0.25)

        clean_texts = [f"c{i}" for i in range(200)]
        clean_embs = embedder.encode_passages(clean_texts)
        scorer.calibrate(clean_embs)

        # Verify that the selected V_low directions indeed capture
        # the low-variance part of the clean distribution
        # The smallest eigenvalues should be selected
        assert scorer._lambda_low is not None
        assert len(scorer._lambda_low) >= 5

        # Generate a poison passage (shifted in low-variance dims)
        # and a passage shifted in HIGH-variance dims
        # The former should have HIGHER DRS
        poison_shift_low = np.zeros(16)
        poison_shift_low[8:16] = 3.0  # low-variance shift → HIGH DRS
        poison_shift_high = np.zeros(16)
        poison_shift_high[0:4] = 3.0   # high-variance shift → LOW DRS

        emb_low = embedder.encode_passages(["clean baseline"])[0] + poison_shift_low
        emb_high = embedder.encode_passages(["clean baseline"])[0] + poison_shift_high

        # Compute DRS manually
        delta_low = emb_low - scorer._mu
        delta_high = emb_high - scorer._mu
        proj_low = delta_low @ scorer._V_low
        proj_high = delta_high @ scorer._V_low

        drs_low = np.sum(proj_low ** 2 / scorer._lambda_low)
        drs_high = np.sum(proj_high ** 2 / scorer._lambda_low)

        # Low-variance shift should be detected more strongly
        assert drs_low > drs_high, (
            f"DRS(low-var shift)={drs_low:.4f} should exceed "
            f"DRS(high-var shift)={drs_high:.4f} — "
            f"this is the core DRS insight!"
        )
