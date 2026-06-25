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

        # Reranker (optional — graceful degradation on failure)
        if self.config.reranker.enabled:
            try:
                self._reranker = RerankerWrapper(
                    model_path=self.config.reranker.path,
                    device=self.config.reranker.device,
                    torch_dtype=self.config.reranker.torch_dtype,
                )
                self._reranker.load()
            except Exception as e:
                logger.warning(f"Reranker failed to load ({e}). Disabling reranker.")
                self.config.reranker.enabled = False
                self._reranker = None

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
            poison_cfg = self.config.poison["poisonedrag"]
            kwargs["mode"] = poison_cfg.generation_model  # "template" or "generator"
            kwargs["template_type"] = poison_cfg.template_mode  # "contradict" etc.
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
        loader = DatasetLoader(self.config, offline=getattr(self.config, 'offline', False))
        corpus = loader.load_corpus(domain)
        if corpus:
            retriever.index_corpus(corpus)
        else:
            # If no corpus available, use a synthetic one for testing
            logger.warning(
                f"No corpus found for {domain}, using synthetic passages. "
                f"Results will NOT be valid for publication."
            )
            synthetic_docs = _get_synthetic_corpus(domain)
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


def _get_synthetic_corpus(domain: str) -> List[str]:
    """Domain-specific synthetic corpus for offline/testing use.

    These are varied enough in vocabulary and structure to exercise
    the embedder and PPL scorer meaningfully, but they are NOT real
    domain documents — results with synthetic corpus are for pipeline
    validation only.
    """
    corpora = {
        "nq": [
            "The Declaration of Independence was adopted by the Continental Congress on July 4, 1776. Thomas Jefferson served as the primary author, with contributions from John Adams and Benjamin Franklin. The document outlined grievances against King George III and articulated the colonies' right to self-governance based on natural rights philosophy.",
            "Paris is the capital and largest city of France, situated on the river Seine in northern France. Known as the City of Light, Paris is a global center for art, fashion, gastronomy, and culture. The city is home to landmarks such as the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum.",
            "The Solar System consists of eight planets orbiting the Sun: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune. Pluto was reclassified as a dwarf planet in 2006 by the International Astronomical Union. The inner planets are rocky, while the outer planets are gas giants.",
            "The United Nations was founded on October 24, 1945, following the end of World War II. The organization's primary goals include maintaining international peace and security, promoting human rights, and fostering social and economic development. Its headquarters is located in New York City.",
            "Climate change is primarily driven by the greenhouse effect, where gases like carbon dioxide and methane trap heat in Earth's atmosphere. Human activities including fossil fuel combustion, deforestation, and industrial agriculture have accelerated this process. The result is rising global temperatures and increasingly extreme weather events.",
            "Penicillin was discovered by Alexander Fleming in 1928 at St. Mary's Hospital in London. Fleming observed that a mold contaminant (Penicillium notatum) inhibited the growth of Staphylococcus bacteria. This accidental discovery revolutionized medicine by introducing the first true antibiotic.",
            "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy. Using chlorophyll, they combine carbon dioxide and water to produce glucose and oxygen. The overall equation is 6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂, powered by sunlight.",
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. Key approaches include supervised learning, unsupervised learning, and reinforcement learning. Deep neural networks have achieved remarkable results in image recognition and natural language processing.",
            "The Mona Lisa was painted by Leonardo da Vinci in the early 16th century, likely between 1503 and 1519. The portrait, believed to depict Lisa Gherardini, is renowned for its enigmatic smile and innovative sfumato technique. It is displayed at the Louvre Museum in Paris.",
            "The speed of light in vacuum is exactly 299,792,458 meters per second, a fundamental constant of physics denoted by c. According to Einstein's theory of special relativity, nothing with mass can reach or exceed this speed. Light travels approximately 9.46 trillion kilometers in one year, a distance known as a light-year.",
        ],
        "hotpotqa": [
            "The Beatles were an English rock band formed in Liverpool in 1960, comprising John Lennon, Paul McCartney, George Harrison, and Ringo Starr. After the band's breakup in 1970, John Lennon pursued a successful solo career and collaborated with artist Yoko Ono, whom he married in 1969. Lennon's solo work includes the iconic album 'Imagine'.",
            "SpaceX, founded by Elon Musk in 2002, revolutionized space travel by developing reusable rocket technology. Musk also co-founded Tesla Inc., which manufactures electric vehicles and clean energy products. Tesla's Model 3 became the world's best-selling electric vehicle.",
            "Leonardo DiCaprio is an American actor who starred in James Cameron's Titanic (1997) alongside Kate Winslet, and later in Christopher Nolan's Inception (2010). Both films were major critical and commercial successes. DiCaprio won his first Academy Award for Best Actor for his role in The Revenant (2015).",
            "Switzerland is a landlocked country that borders France to the west and Germany to the north, among other neighboring nations. Its de facto capital is Bern, while Zurich is its largest city and a major global financial hub.",
        ],
        "bioasq": [
            "BRCA1 is a tumor suppressor gene located on chromosome 17q21. Mutations in BRCA1 significantly increase the risk of developing breast and ovarian cancers. The gene produces a protein involved in DNA repair, cell cycle checkpoint control, and maintenance of genomic stability.",
            "Metformin is a first-line oral medication for type 2 diabetes mellitus. It works primarily by reducing hepatic glucose production through inhibition of gluconeogenesis, while also increasing peripheral insulin sensitivity. Common side effects include gastrointestinal disturbances such as nausea and diarrhea.",
            "ACE inhibitors are a class of medications used primarily for hypertension and heart failure. They work by inhibiting the angiotensin-converting enzyme, reducing the production of angiotensin II. Common side effects include a persistent dry cough, hyperkalemia, and in rare cases, angioedema.",
            "Alzheimer's disease is associated with several genetic risk factors including the APOE ε4 allele. Mutations in APP, PSEN1, and PSEN2 genes cause early-onset familial Alzheimer's disease. The pathological hallmarks include extracellular amyloid-beta plaques and intracellular neurofibrillary tangles composed of hyperphosphorylated tau protein.",
            "Immunotherapy has emerged as a promising approach for melanoma treatment, particularly immune checkpoint inhibitors targeting CTLA-4 and PD-1. These agents work by releasing the brakes on T cells, allowing the immune system to recognize and attack tumor cells. Combination therapy with nivolumab and ipilimumab has shown improved survival rates.",
        ],
        "finance": [
            "Interest rate changes have a significant inverse relationship with bond prices. When interest rates rise, existing bonds with lower coupon rates become less attractive, causing their market prices to fall. The duration of a bond measures its sensitivity to interest rate changes.",
            "Quantitative easing (QE) is an unconventional monetary policy tool used by central banks to stimulate the economy when standard policy becomes ineffective. By purchasing large quantities of government bonds and other securities, central banks increase the money supply and lower long-term interest rates. This can lead to higher inflation expectations.",
            "A call option gives the holder the right, but not the obligation, to buy an underlying asset at a specified strike price before expiration. A put option gives the right to sell. Options are priced using models such as Black-Scholes, which account for factors including volatility, time to expiration, and risk-free interest rate.",
            "The Weighted Average Cost of Capital (WACC) represents a firm's average cost of financing from all sources, weighted by their proportion in the capital structure. WACC = (E/V)×Re + (D/V)×Rd×(1-Tc), where E is equity value, D is debt value, V is total value, Re is cost of equity, Rd is cost of debt, and Tc is the corporate tax rate.",
        ],
    }
    default = corpora.get("nq", list(corpora.values())[0])
    docs = corpora.get(domain, default)
    # Repeat to reach sufficient document count for FAISS indexing
    return docs * 8  # ~80 varied passages
