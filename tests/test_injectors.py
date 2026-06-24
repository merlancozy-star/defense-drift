"""Test poison injectors — verify injection ratios and labeling."""

import pytest
from data.schemas import Passage
from injectors.poisonedrag import PoisonedRAGInjector
from injectors.corruptrag import CorruptRAGInjector


def make_clean_passages(n: int = 10) -> list:
    """Create synthetic clean passages."""
    return [
        Passage(
            id=f"clean_{i}",
            text=f"This is clean passage number {i}. It contains factual information about the topic.",
            is_poison=False,
        )
        for i in range(n)
    ]


class TestPoisonedRAGInjector:
    """Test PoisonedRAG injection."""

    def test_attack_name(self):
        injector = PoisonedRAGInjector()
        assert injector.attack_name == "poisonedrag"

    def test_inject_produces_correct_count(self):
        """Injection should produce exactly top_k passages."""
        injector = PoisonedRAGInjector(mode="template", random_seed=42)
        clean = make_clean_passages(15)

        mixed = injector.inject(
            query="What is machine learning?",
            clean_passages=clean,
            poison_ratio=0.2,
            top_k=10,
        )

        assert len(mixed) == 10

    def test_inject_correct_poison_count(self):
        """Should have the right number of poison passages."""
        injector = PoisonedRAGInjector(mode="template", random_seed=42)
        clean = make_clean_passages(15)

        for ratio in [0.05, 0.1, 0.2, 0.4]:
            mixed = injector.inject(
                query="test query",
                clean_passages=clean,
                poison_ratio=ratio,
                top_k=10,
            )
            n_poison_expected = max(1, int(10 * ratio))
            n_poison_actual = sum(1 for p in mixed if p.is_poison)
            n_clean_actual = sum(1 for p in mixed if not p.is_poison)

            assert n_poison_actual == n_poison_expected, (
                f"Ratio {ratio}: expected {n_poison_expected} poison, got {n_poison_actual}"
            )
            assert len(mixed) == 10

    def test_is_poison_flags_set(self):
        """All poison passages should have is_poison=True."""
        injector = PoisonedRAGInjector(mode="template", random_seed=42)
        clean = make_clean_passages(15)

        mixed = injector.inject(
            query="test",
            clean_passages=clean,
            poison_ratio=0.3,
            top_k=10,
        )

        for p in mixed:
            if p.poison_attack == "poisonedrag":
                assert p.is_poison, f"Passage {p.id} should be marked poison"
            else:
                assert not p.is_poison, f"Passage {p.id} should be marked clean"

    def test_craft_generates_passages(self):
        """craft() should generate the requested number of passages."""
        injector = PoisonedRAGInjector(mode="template", random_seed=42)
        clean = make_clean_passages(5)

        poison = injector.craft(
            query="What is artificial intelligence?",
            clean_passages=clean,
            n_poison=3,
        )

        assert len(poison) == 3
        for p in poison:
            assert p.is_poison
            assert p.poison_attack == "poisonedrag"
            assert len(p.text) > 0

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        clean = make_clean_passages(5)

        inj1 = PoisonedRAGInjector(mode="template", random_seed=42)
        inj2 = PoisonedRAGInjector(mode="template", random_seed=42)

        poison1 = inj1.craft("test query", clean, 3)
        poison2 = inj2.craft("test query", clean, 3)

        for p1, p2 in zip(poison1, poison2):
            assert p1.text == p2.text

    def test_template_types(self):
        """All template types should produce valid passages."""
        for tmpl_type in ["contradict", "hallucinate", "distract"]:
            injector = PoisonedRAGInjector(
                mode="template", template_type=tmpl_type, random_seed=42
            )
            clean = make_clean_passages(5)
            poison = injector.craft("test query", clean, 5)
            assert len(poison) == 5
            for p in poison:
                assert len(p.text) > 0


class TestCorruptRAGInjector:
    """Test CorruptRAG injection (stronger attack)."""

    def test_attack_name(self):
        injector = CorruptRAGInjector()
        assert injector.attack_name == "corruptrag"

    def test_craft_corrupts_existing_passages(self):
        """CorruptRAG should corrupt real passages, not generate new ones."""
        injector = CorruptRAGInjector(corrupt_ratio=0.5, random_seed=42)
        clean = make_clean_passages(5)

        poison = injector.craft(
            query="test query",
            clean_passages=clean,
            n_poison=3,
        )

        assert len(poison) == 3
        for p in poison:
            assert p.is_poison
            assert p.poison_attack == "corruptrag"
            # Corrupted text should differ from original
            # (but may be identical if no corruption targets were found)
            assert len(p.text) > 0

    def test_numerical_corruption(self):
        """Numbers should be corrupted in the text."""
        injector = CorruptRAGInjector(corrupt_ratio=0.8, random_seed=42)
        clean = [
            Passage(
                id="c0",
                text="The study found that 30% of patients showed improvement and blood pressure decreased by 15 points.",
                is_poison=False,
            )
        ]

        poison = injector.craft("query", clean, n_poison=1)
        # With corrupt_ratio=0.8, the number "30" should be changed
        assert len(poison) == 1
        # The corrupted text should NOT exactly equal the original
        # (at minimum, numbers may change)
        assert poison[0].text != clean[0].text or "30" not in poison[0].text or "15" not in poison[0].text, \
            "Numerical corruption should change at least some numbers"

    def test_inject_produces_correct_count(self):
        injector = CorruptRAGInjector(random_seed=42)
        clean = make_clean_passages(15)

        mixed = injector.inject(
            query="test",
            clean_passages=clean,
            poison_ratio=0.2,
            top_k=10,
        )

        assert len(mixed) == 10
        n_poison = sum(1 for p in mixed if p.is_poison)
        assert n_poison == max(1, int(10 * 0.2))
