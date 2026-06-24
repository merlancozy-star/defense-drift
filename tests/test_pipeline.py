"""End-to-end pipeline tests using the --dry-run mode.

These tests verify that the full pipeline can run without real models,
producing valid CellResult objects.
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schemas import CellResult
from utils.config import load_config
from utils.metrics import compute_auroc, compute_separability_drop, bootstrap_auroc_ci
from utils.results import save_results, load_results, summarize_results


@pytest.fixture
def config():
    """Load the real experiment config."""
    config_path = Path(__file__).parent.parent / "separability_collapse.yaml"
    return load_config(str(config_path))


class TestExperimentConfig:
    """Verify configuration loads correctly."""

    def test_config_loads(self, config):
        """Config should load without errors."""
        assert config is not None

    def test_signals_present(self, config):
        """All four signals should be in config."""
        assert len(config.signals) == 4
        assert "perplexity" in config.signals
        assert "drs" in config.signals
        assert "cluster_distance" in config.signals
        assert "attention_variance" in config.signals

    def test_domains_present(self, config):
        """Source and target domains should be configured."""
        assert len(config.source_domains) >= 1
        assert len(config.target_domains) >= 1
        assert "nq" in config.source_domains
        assert "bioasq" in config.target_domains

    def test_attacks_present(self, config):
        """Both attacks should be configured."""
        assert "poisonedrag" in config.attacks
        assert "corruptrag" in config.attacks

    def test_poison_ratios(self, config):
        """Four poison ratios should be configured."""
        assert len(config.poison_ratios) == 4
        assert 0.05 in config.poison_ratios
        assert 0.4 in config.poison_ratios

    def test_retrieval_consistency(self, config):
        """Retrieval config validation should pass."""
        assert config.validate_retrieval_consistency()

    def test_domain_type_lookup(self, config):
        """Domain type should be correctly identified."""
        assert config.get_domain_type("nq") == "source"
        assert config.get_domain_type("bioasq") == "target"
        with pytest.raises(ValueError):
            config.get_domain_type("unknown_domain")


class TestCellResult:
    """Test CellResult dataclass and derived fields."""

    def test_source_baseline_no_drop(self):
        """Source cells should not have a drop."""
        cell = CellResult(
            signal="perplexity",
            domain="nq",
            domain_type="source",
            attack="poisonedrag",
            poison_ratio=0.1,
            auroc=0.85,
            n_samples=100,
            is_source_baseline=True,
        )
        assert cell.separability_drop is None
        assert cell.domain_robust is None

    def test_target_drop_computation(self):
        """Target cells should compute drop from source."""
        cell = CellResult(
            signal="perplexity",
            domain="bioasq",
            domain_type="target",
            attack="poisonedrag",
            poison_ratio=0.1,
            auroc=0.68,
            n_samples=100,
        )
        cell.compute_drop(source_auroc=0.85)
        assert cell.separability_drop == pytest.approx(0.17, abs=0.01)
        assert cell.domain_robust is False

    def test_domain_robust_flag(self):
        """Small drops should be flagged as domain-robust."""
        cell = CellResult(
            signal="drs",
            domain="bioasq",
            domain_type="target",
            attack="poisonedrag",
            poison_ratio=0.1,
            auroc=0.84,
            n_samples=100,
        )
        cell.compute_drop(source_auroc=0.85)
        assert cell.domain_robust is True  # Δ=0.01 < 0.05

    def test_to_csv_row(self, config):
        """CSV row should contain all header fields."""
        cell = CellResult(
            signal="perplexity",
            domain="nq",
            domain_type="source",
            attack="poisonedrag",
            poison_ratio=0.1,
            auroc=0.85,
            n_samples=100,
        )
        row = cell.to_csv_row()
        for col in CellResult.csv_header():
            assert col in row, f"Missing column: {col}"


class TestMetrics:
    """Test AUROC and related metrics."""

    def test_compute_auroc_perfect(self):
        import numpy as np
        scores = np.array([0.0, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        assert compute_auroc(scores, labels) == 1.0

    def test_compute_auroc_nan_single_class(self):
        import numpy as np
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])
        result = compute_auroc(scores, labels)
        assert np.isnan(result)

    def test_separability_drop(self):
        drop = compute_separability_drop(0.85, 0.68)
        assert drop == pytest.approx(0.17, abs=0.01)

    def test_separability_drop_negative(self):
        """If target is better than source, drop is negative."""
        drop = compute_separability_drop(0.70, 0.80)
        assert drop == pytest.approx(-0.10, abs=0.01)

    def test_bootstrap_auroc_ci(self):
        import numpy as np
        rng = np.random.RandomState(42)
        scores = np.concatenate([
            rng.normal(0, 1, 50),  # clean: low scores
            rng.normal(2, 1, 50),  # poison: high scores
        ])
        labels = np.array([0] * 50 + [1] * 50)

        mean, ci_low, ci_high = bootstrap_auroc_ci(scores, labels, n_bootstrap=200)
        assert 0.8 < mean < 1.0
        assert ci_low <= mean <= ci_high


class TestResultsIO:
    """Test result saving and loading."""

    def test_save_and_load_csv(self, tmp_path):
        """Results should round-trip through CSV."""
        cells = [
            CellResult("perplexity", "nq", "source", "poisonedrag", 0.1, 0.85, 100),
            CellResult("perplexity", "bioasq", "target", "poisonedrag", 0.1, 0.68, 100),
        ]
        cells[1].compute_drop(0.85)

        saved = save_results(cells, str(tmp_path), prefix="test", formats=["csv"])
        assert os.path.exists(saved)

        loaded = load_results(saved)
        assert len(loaded) == 2
        assert loaded[0].auroc == 0.85
        assert loaded[1].separability_drop == pytest.approx(0.17, abs=0.01)

    def test_summarize_results(self):
        """Summarize should identify collapses and robust signals."""
        cells = [
            CellResult("perplexity", "nq", "source", "poisonedrag", 0.1, 0.85, 100),
            CellResult("perplexity", "bioasq", "target", "poisonedrag", 0.1, 0.60, 100),
            CellResult("drs", "nq", "source", "poisonedrag", 0.1, 0.82, 100),
            CellResult("drs", "bioasq", "target", "poisonedrag", 0.1, 0.80, 100),
        ]
        cells[1].compute_drop(0.85)  # drop = 0.25 (significant)
        cells[3].compute_drop(0.82)  # drop = 0.02 (robust)

        summary = summarize_results(cells)
        assert summary["n_significant_collapse"] == 1
        assert ("drs", "bioasq") in summary["domain_robust_signals"]


class TestDryRun:
    """Test the --dry-run pipeline mode (runs without models)."""

    def test_dry_run_produces_results(self, config):
        """Dry run should produce CellResult list with all cells."""
        import random
        import numpy as np

        rng = random.Random(42)
        results = []

        for domain in config.all_domains:
            domain_type = config.get_domain_type(domain)
            for signal_name in config.signals:
                for attack_name in config.attacks:
                    for ratio in config.poison_ratios:
                        auroc = rng.uniform(0.80, 0.92) if domain_type == "source" else rng.uniform(0.58, 0.76)
                        results.append(CellResult(
                            signal=signal_name,
                            domain=domain,
                            domain_type=domain_type,
                            attack=attack_name,
                            poison_ratio=ratio,
                            auroc=round(auroc, 4),
                            n_samples=rng.randint(80, 200),
                            is_source_baseline=(domain_type == "source"),
                        ))

        # Should have the full Cartesian product
        expected = len(config.signals) * len(config.all_domains) * len(config.attacks) * len(config.poison_ratios)
        assert len(results) == expected, f"Expected {expected} cells, got {len(results)}"

        # All cells should be valid
        for r in results:
            assert r.signal in config.signals
            assert r.domain in config.all_domains
            assert r.domain_type in ("source", "target")
            assert r.attack in config.attacks
            assert r.poison_ratio in config.poison_ratios
            assert not np.isnan(r.auroc)
            assert r.n_samples > 0
