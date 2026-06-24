"""Build the full collapse matrix (Table 1).

Orchestrates the Cartesian product:
    signals × domains × attacks × poison_ratios

Each cell is computed independently and can be parallelized.
Intermediate results are saved to disk to survive interruptions.
"""

import os
import logging
import itertools
import numpy as np
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

from data.schemas import CellResult, RetrievalConfig
from data.loader import DatasetLoader
from retrieval.embedder import EmbedderWrapper
from retrieval.retriever import Retriever
from models.generator import GeneratorWrapper
from models.reranker import RerankerWrapper
from scorers.base import BaseScorer
from scorers.perplexity import PerplexityScorer
from scorers.drs import DRSScorer
from scorers.cluster_distance import ClusterDistanceScorer
from scorers.attention_variance import AttentionVarianceScorer
from injectors.base import BaseInjector
from injectors.poisonedrag import PoisonedRAGInjector
from injectors.corruptrag import CorruptRAGInjector
from pipeline.assemble_units import UnitAssembler
from utils.config import ExperimentConfig, ModelConfig
from utils.results import save_results, summarize_results

logger = logging.getLogger(__name__)

# Signal registry: name → (factory, requires_embedder, requires_generator)
SIGNAL_REGISTRY = {
    "perplexity": (PerplexityScorer, False, True),
    "drs": (DRSScorer, True, False),
    "cluster_distance": (ClusterDistanceScorer, True, False),
    "attention_variance": (AttentionVarianceScorer, False, True),
}

# Attack registry: name → factory
ATTACK_REGISTRY = {
    "poisonedrag": PoisonedRAGInjector,
    "corruptrag": CorruptRAGInjector,
}


class CollapseMatrixBuilder:
    """Builds the full separability collapse matrix."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._embedder: Optional[EmbedderWrapper] = None
        self._generator: Optional[GeneratorWrapper] = None
        self._reranker: Optional[RerankerWrapper] = None

    def initialize_models(self):
        """Load all models needed for the experiment."""
        logger.info("Initializing models...")

        # Embedder (needed by most signals + retrieval)
        self._embedder = EmbedderWrapper(
            model_path=self.config.embedder.path,
            device=self.config.embedder.device,
            torch_dtype=self.config.embedder.torch_dtype,
            batch_size=self.config.embedder.batch_size,
        )
        self._embedder.load()

        # Generator (needed by perplexity + attention_variance)
        self._generator = GeneratorWrapper(
            model_path=self.config.generator.path,
            device=self.config.generator.device,
            torch_dtype=self.config.generator.torch_dtype,
            max_length=self.config.generator.max_length,
            output_attentions=self.config.generator.output_attentions,
        )
        self._generator.load()

        # Reranker (optional)
        if self.config.reranker.enabled:
            self._reranker = RerankerWrapper(
                model_path=self.config.reranker.path,
                device=self.config.reranker.device,
                torch_dtype=self.config.reranker.torch_dtype,
            )
            self._reranker.load()

        logger.info("All models loaded")

    def shutdown_models(self):
        """Release GPU memory."""
        if self._embedder:
            self._embedder.unload()
        if self._generator:
            self._generator.unload()
        if self._reranker:
            self._reranker.unload()

    def _make_scorer(self, signal_name: str) -> BaseScorer:
        """Instantiate a scorer by name."""
        if signal_name not in SIGNAL_REGISTRY:
            raise ValueError(f"Unknown signal: {signal_name}. Known: {list(SIGNAL_REGISTRY)}")

        factory, needs_embedder, needs_generator = SIGNAL_REGISTRY[signal_name]
        kwargs = {}
        if needs_embedder:
            kwargs["embedder"] = self._embedder
        if needs_generator:
            kwargs["generator"] = self._generator

        return factory(**kwargs)

    def _make_injector(self, attack_name: str) -> BaseInjector:
        """Instantiate an injector by name."""
        if attack_name not in ATTACK_REGISTRY:
            raise ValueError(f"Unknown attack: {attack_name}. Known: {list(ATTACK_REGISTRY)}")

        factory = ATTACK_REGISTRY[attack_name]
        kwargs = {}
        if attack_name == "poisonedrag":
            kwargs["mode"] = self.config.poison["poisonedrag"].template_mode
        elif attack_name == "corruptrag":
            kwargs["corrupt_ratio"] = self.config.poison["corruptrag"].corrupt_ratio

        if attack_name in ("poisonedrag", "corruptrag"):
            kwargs["generator"] = self._generator

        kwargs["random_seed"] = self.config.diagnostic.random_seed
        return factory(**kwargs)

    def _build_retriever(self) -> Retriever:
        """Build a retriever with the locked retrieval config."""
        ret_config = RetrievalConfig(**self.config.to_retrieval_config_dict())
        return Retriever(
            config=ret_config,
            embedder=self._embedder,
            reranker=self._reranker,
        )

    def compute_cell(
        self,
        signal: str,
        domain: str,
        attack: str,
        ratio: float,
        max_queries: Optional[int] = None,
    ) -> CellResult:
        """Compute one cell of the collapse matrix.

        Returns a CellResult with AUROC for this condition.
        """
        domain_type = self.config.get_domain_type(domain)
        domain_cfg = self.config.get_domain_config(domain)

        # Build scorer, injector, retriever for this cell
        scorer = self._make_scorer(signal)
        injector = self._make_injector(attack)
        retriever = self._build_retriever()

        # Load corpus and build index for this domain
        loader = DatasetLoader(self.config)
        corpus = loader.load_corpus(domain)
        if corpus:
            retriever.index_corpus(corpus)
        else:
            # If no corpus available, use a synthetic one for testing
            logger.warning(
                f"No corpus found for {domain}, using synthetic passages. "
                f"Results will NOT be valid for publication."
            )
            synthetic_docs = [
                f"This is a synthetic document about {domain} topic. It contains general knowledge about the subject matter.",
                f"Research in {domain} has advanced significantly in recent years with many new discoveries.",
                f"The field of {domain} encompasses many sub-disciplines and methodologies.",
                f"Recent findings in {domain} challenge previously held assumptions about key mechanisms.",
                f"Understanding {domain} requires interdisciplinary approaches combining theory and practice.",
                f"Historical context is essential for understanding modern developments in {domain}.",
                f"Several meta-analyses in {domain} have identified consistent patterns across studies.",
                f"Methodological challenges in {domain} research include sample size and reproducibility.",
                f"Emerging technologies are transforming how researchers approach problems in {domain}.",
                f"International collaboration has accelerated progress in understanding {domain} phenomena.",
            ] * 20  # 200 synthetic docs
            retriever.index_corpus(synthetic_docs)

        # Assemble units and compute AUROC
        assembler = UnitAssembler(
            config=self.config,
            retriever=retriever,
            scorer=scorer,
            injector=injector,
        )

        units, auroc = assembler.assemble(
            domain=domain,
            poison_ratio=ratio,
            max_queries=max_queries,
        )

        if not units:
            return CellResult(
                signal=signal,
                domain=domain,
                domain_type=domain_type,
                attack=attack,
                poison_ratio=ratio,
                auroc=float("nan"),
                n_samples=0,
                is_source_baseline=(domain_type == "source"),
            )

        return assembler.get_cell_result(
            domain=domain,
            domain_type=domain_type,
            poison_ratio=ratio,
            auroc=auroc,
            n_samples=len(units),
        )

    def build_full_matrix(
        self,
        signals: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        attacks: Optional[List[str]] = None,
        ratios: Optional[List[float]] = None,
        max_queries: Optional[int] = None,
        parallel: bool = True,
    ) -> List[CellResult]:
        """Build the complete collapse matrix.

        Computes AUROC for every (signal, domain, attack, ratio) combination,
        then computes separability_drop for target-domain cells relative to
        their source-domain baselines.

        Args:
            signals: Signals to evaluate (default: all in config).
            domains: Domains to evaluate (default: all in config).
            attacks: Attacks to evaluate (default: all in config).
            ratios: Poison ratios (default: all in config).
            max_queries: Max queries per cell (default: from domain config).
            parallel: Use parallel processing.

        Returns:
            List of CellResult, one per matrix cell.
        """
        if signals is None:
            signals = self.config.signals
        if domains is None:
            domains = self.config.all_domains
        if attacks is None:
            attacks = self.config.attacks
        if ratios is None:
            ratios = self.config.poison_ratios

        # Generate Cartesian product
        cells = list(itertools.product(signals, domains, attacks, ratios))
        logger.info(
            f"Building collapse matrix: "
            f"{len(signals)} signals × {len(domains)} domains × "
            f"{len(attacks)} attacks × {len(ratios)} ratios = {len(cells)} cells"
        )

        results = []
        output_dir = self.config.output.results_dir
        os.makedirs(output_dir, exist_ok=True)
        intermediate_path = os.path.join(
            output_dir,
            f"intermediate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        if parallel and len(cells) > 1:
            # Sequential for now — GPU sharing across processes is tricky
            # In production, use joblib with loky backend and per-process model loading
            logger.info("Running cells sequentially (GPU memory constraint)")
            results = self._run_sequential(
                cells, max_queries, intermediate_path
            )
        else:
            results = self._run_sequential(cells, max_queries, intermediate_path)

        # Compute separability drops
        self._compute_drops(results, signals, attacks, ratios)

        # Run acceptance checks
        self._check_acceptance(results)

        return results

    def _run_sequential(
        self,
        cells: List[Tuple],
        max_queries: Optional[int],
        intermediate_path: str,
    ) -> List[CellResult]:
        """Run cells sequentially, saving intermediate results."""
        results = []
        for i, (signal, domain, attack, ratio) in enumerate(cells):
            logger.info(
                f"[{i+1}/{len(cells)}] signal={signal} domain={domain} "
                f"attack={attack} ratio={ratio}"
            )

            try:
                cell = self.compute_cell(signal, domain, attack, ratio, max_queries)
                results.append(cell)

                # Save intermediate
                if self.config.output.save_intermediate and len(results) % 4 == 0:
                    save_results(results, os.path.dirname(intermediate_path),
                                 prefix="intermediate", formats=["csv"])
            except Exception as e:
                logger.error(f"Cell failed: {e}", exc_info=True)
                # Still record a NaN cell so the matrix isn't missing entries
                results.append(CellResult(
                    signal=signal,
                    domain=domain,
                    domain_type=self.config.get_domain_type(domain),
                    attack=attack,
                    poison_ratio=ratio,
                    auroc=float("nan"),
                    n_samples=0,
                ))

        return results

    def _compute_drops(
        self,
        results: List[CellResult],
        signals: List[str],
        attacks: List[str],
        ratios: List[float],
    ):
        """Compute separability_drop for each target-domain cell.

        Uses the matching source-domain cell as baseline.
        For multiple source domains, uses the average source AUROC.
        """
        source_domains = self.config.source_domains

        # Build lookup: (signal, attack, ratio) → avg source AUROC
        source_aurocs: Dict[Tuple[str, str, float], List[float]] = {}
        for r in results:
            if r.domain_type == "source" and not np.isnan(r.auroc):
                key = (r.signal, r.attack, r.poison_ratio)
                source_aurocs.setdefault(key, []).append(r.auroc)

        # Average across source domains
        source_avg = {}
        for key, aurocs in source_aurocs.items():
            source_avg[key] = sum(aurocs) / len(aurocs) if aurocs else float("nan")

        # Compute drops
        for r in results:
            if r.domain_type == "target" and not np.isnan(r.auroc):
                key = (r.signal, r.attack, r.poison_ratio)
                ref = source_avg.get(key, float("nan"))
                if not np.isnan(ref):
                    r.compute_drop(ref)

    def _check_acceptance(self, results: List[CellResult]):
        """Verify results against acceptance criteria from config."""
        criteria = self.config.acceptance

        # Check source AUROC
        source_results = [r for r in results if r.domain_type == "source"]
        signal_aurocs = {}
        for r in source_results:
            if not np.isnan(r.auroc):
                signal_aurocs.setdefault(r.signal, []).append(r.auroc)

        ok_signals = 0
        for signal, aurocs in signal_aurocs.items():
            avg = sum(aurocs) / len(aurocs)
            passed = avg >= criteria.source_auroc_min
            status = "✓" if passed else "✗"
            logger.info(
                f"[ACCEPTANCE] {status} {signal}: avg source AUROC = {avg:.4f} "
                f"(min required: {criteria.source_auroc_min})"
            )
            if passed:
                ok_signals += 1

        if ok_signals < len(signal_aurocs):
            logger.warning(
                f"[ACCEPTANCE] Only {ok_signals}/{len(signal_aurocs)} signals "
                f"meet source AUROC threshold. Check scorer implementations."
            )

        # Check collapse
        drop_results = [r for r in results if r.separability_drop is not None
                        and not np.isnan(r.separability_drop)]
        significant = [r for r in drop_results if r.separability_drop > criteria.collapse_threshold]
        signals_with_collapse = set(r.signal for r in significant)

        logger.info(
            f"[ACCEPTANCE] Signals with significant collapse: "
            f"{signals_with_collapse} (need ≥{criteria.min_collapse_signals})"
        )
