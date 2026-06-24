"""Assemble scoring units: retrieve → inject → score → label → package.

This is the core pipeline that produces ScoringUnit objects for one
(signal, domain, attack, ratio) combination. Each unit is a single
(query, passage) pair with all signal scores and the ground-truth is_poison label.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Tuple

from data.schemas import Passage, Query, ScoringUnit, CellResult, RetrievalConfig
from data.loader import DatasetLoader
from retrieval.retriever import Retriever
from scorers.base import BaseScorer
from injectors.base import BaseInjector
from utils.config import ExperimentConfig
from utils.metrics import compute_auroc

logger = logging.getLogger(__name__)


class UnitAssembler:
    """Assembles ScoringUnits for one experimental condition.

    Lifecycle per condition:
    1. Load queries for the domain
    2. For each query: retrieve clean passages
    3. Inject poison at specified ratio
    4. Score all passages with the signal
    5. Package as ScoringUnits with ground-truth labels
    """

    def __init__(
        self,
        config: ExperimentConfig,
        retriever: Retriever,
        scorer: BaseScorer,
        injector: BaseInjector,
    ):
        self.config = config
        self.retriever = retriever
        self.scorer = scorer
        self.injector = injector
        self.loader = DatasetLoader(config)

    def assemble(
        self,
        domain: str,
        poison_ratio: float,
        max_queries: Optional[int] = None,
    ) -> Tuple[List[ScoringUnit], float]:
        """Run the full assemble pipeline for one condition.

        Args:
            domain: Domain name (nq, hotpotqa, bioasq, finance).
            poison_ratio: Fraction of passages that are poison.
            max_queries: Override number of queries to process.

        Returns:
            (scoring_units, auroc): List of ScoringUnits and the AUROC
            for this condition.
        """
        # 1. Load queries
        queries = self.loader.load_queries(domain, max_queries=max_queries)

        # 2. Retrieve + inject + score per query
        all_units = []
        top_k = self.config.retrieval.top_k

        for query in queries:
            try:
                # Retrieve clean passages
                clean_passages = self.retriever.search(query.text, top_k=top_k)

                # Inject poison
                mixed_passages = self.injector.inject(
                    query=query.text,
                    clean_passages=clean_passages,
                    poison_ratio=poison_ratio,
                    top_k=top_k,
                )

                # Score all passages with the signal
                passage_texts = [p.text for p in mixed_passages]
                scores = self.scorer.score(query.text, passage_texts)

                # Package as ScoringUnits
                for passage, score in zip(mixed_passages, scores):
                    unit = ScoringUnit(
                        query_id=query.id,
                        passage_id=passage.id,
                        domain=domain,
                        is_poison=passage.is_poison,
                        attack=passage.poison_attack,
                        poison_ratio=poison_ratio,
                        signals={self.scorer.name: float(score)},
                    )
                    all_units.append(unit)

            except Exception as e:
                logger.warning(f"Failed to process query {query.id}: {e}")
                continue

        if not all_units:
            logger.error(f"No units assembled for domain={domain}, ratio={poison_ratio}")
            return [], float("nan")

        # 3. Compute AUROC
        scores_arr = np.array([u.get_signal(self.scorer.name) for u in all_units])
        labels_arr = np.array([1 if u.is_poison else 0 for u in all_units])

        auroc = compute_auroc(scores_arr, labels_arr)
        logger.info(
            f"Assembled {len(all_units)} units | "
            f"domain={domain} signal={self.scorer.name} "
            f"attack={self.injector.attack_name} ratio={poison_ratio} | "
            f"AUROC={auroc:.4f}"
        )

        return all_units, auroc

    def get_cell_result(
        self,
        domain: str,
        domain_type: str,
        poison_ratio: float,
        auroc: float,
        n_samples: int,
        source_auroc: Optional[float] = None,
    ) -> CellResult:
        """Create a CellResult for this condition."""
        is_source = (domain_type == "source")
        result = CellResult(
            signal=self.scorer.name,
            domain=domain,
            domain_type=domain_type,
            attack=self.injector.attack_name,
            poison_ratio=poison_ratio,
            auroc=round(auroc, 4),
            n_samples=n_samples,
            is_source_baseline=is_source,
        )
        if source_auroc is not None and not is_source:
            result.compute_drop(source_auroc)
        return result
