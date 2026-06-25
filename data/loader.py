"""Dataset loader for RAG-QA Arena datasets.

Supports: NQ, HotpotQA (source domains), BioASQ, Finance (target domains).
All datasets are loaded via HuggingFace datasets with a unified interface.
"""

import json
import os
import random
import logging
from typing import List, Optional
from pathlib import Path

from data.schemas import Query
from utils.config import ExperimentConfig, DomainConfig

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Unified loader for all domain datasets.

    Handles:
    - HuggingFace datasets as primary source
    - Local JSONL/CSV fallback
    - Synthetic data for testing
    """

    def __init__(self, config: ExperimentConfig, offline: bool = False):
        self.config = config
        self.offline = offline
        self.rng = random.Random(config.diagnostic.random_seed)

    def load_queries(self, domain: str, max_queries: Optional[int] = None) -> List[Query]:
        """Load queries for a given domain.

        Args:
            domain: Domain name (nq, hotpotqa, bioasq, finance).
            max_queries: Override max queries from config.

        Returns:
            List of Query objects with id, text, domain, and golden_answer.
        """
        domain_cfg = self.config.get_domain_config(domain)
        if max_queries is None:
            max_queries = domain_cfg.max_queries

        # 1. Try local JSONL file first (fastest, works offline)
        local_dir = getattr(self.config, 'local_dir', None)
        if local_dir:
            queries = self._load_from_local(local_dir, domain, max_queries)
            if queries:
                logger.info(f"Loaded {len(queries)} queries from local JSONL for {domain}")
                return queries

        # 2. Try HuggingFace (requires network)
        if not self.offline:
            try:
                queries = self._load_from_huggingface(domain, domain_cfg, max_queries)
                if queries:
                    return queries
            except Exception as e:
                logger.warning(f"Failed to load {domain} from HuggingFace ({e})")

        # 3. Fallback: synthetic queries
        logger.info(f"Using synthetic queries for {domain} (no local/HF data available)")
        return self._generate_synthetic_queries(domain, max_queries)

        return queries

    def _load_from_local(
        self, local_dir: str, domain: str, max_queries: int
    ) -> Optional[List[Query]]:
        """Load queries from a local JSONL file.

        Expected path: {local_dir}/{domain}_queries.jsonl
        Each line: {"id": "...", "question": "...", "answer": "...", "domain": "..."}
        """
        import json
        path = os.path.join(local_dir, f"{domain}_queries.jsonl")
        if not os.path.exists(path):
            return None

        queries = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_queries:
                    break
                try:
                    item = json.loads(line)
                    queries.append(Query(
                        id=item.get("id", f"{domain}_{i}"),
                        text=item["question"].strip(),
                        domain=domain,
                        golden_answer=item.get("answer", ""),
                    ))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipping malformed line {i} in {path}: {e}")
                    continue

        return queries if queries else None

    def _load_from_huggingface(
        self, domain: str, cfg: DomainConfig, max_queries: int
    ) -> List[Query]:
        """Load queries from HuggingFace datasets."""
        from datasets import load_dataset  # Lazy import (heavy dependency)
        # Handle special HF paths
        hf_path = cfg.hf_path
        hf_subset = cfg.hf_subset

        # Load the dataset
        if hf_subset:
            dataset = load_dataset(hf_path, hf_subset, split="train", )
        else:
            try:
                dataset = load_dataset(hf_path, split="train", )
            except Exception:
                # Try without split specification
                dataset = load_dataset(hf_path, )
                if isinstance(dataset, dict):
                    dataset = dataset.get("train", list(dataset.values())[0])

        # Extract queries
        queries = []
        for i, item in enumerate(dataset):
            if i >= max_queries:
                break

            query_text = self._extract_field(item, cfg.query_field)
            if not query_text:
                continue

            # Try to extract answer — first from config, then common field names
            answer = None
            candidate_fields = []
            if cfg.answer_field:
                candidate_fields.append(cfg.answer_field)
            candidate_fields.extend(["answer", "answers", "golden_answer", "label", "ideal_answer"])
            for ans_field in candidate_fields:
                ans = self._extract_field(item, ans_field)
                if ans is not None:
                    if isinstance(ans, list) and len(ans) > 0:
                        answer = str(ans[0])
                    else:
                        answer = str(ans)
                    break

            queries.append(Query(
                id=f"{domain}_{i}",
                text=str(query_text).strip(),
                domain=domain,
                golden_answer=answer,
            ))

        logger.info(f"Loaded {len(queries)} queries from {domain} (HF: {hf_path})")
        return queries

    def _extract_field(self, item: dict, field: str) -> Optional[str]:
        """Extract a field from a dataset item, handling nested keys."""
        if field is None:
            return None
        if field in item:
            return item[field]
        # Try nested: "answers.text"
        if "." in field:
            parts = field.split(".")
            val = item
            for p in parts:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and p in val[0]:
                    val = [v[p] for v in val]
                else:
                    return None
            return val
        return None

    def _generate_synthetic_queries(self, domain: str, n: int) -> List[Query]:
        """Generate synthetic queries for testing when real data is unavailable.

        These are semantically plausible for the domain but are NOT real data.
        Used only for pipeline testing — not for results.
        """
        templates = {
            "nq": [
                "who wrote the declaration of independence",
                "what is the capital of france",
                "how many planets are in the solar system",
                "when was the united nations founded",
                "what causes climate change",
                "who discovered penicillin",
                "what is the speed of light",
                "how does photosynthesis work",
                "what is machine learning",
                "who painted the mona lisa",
            ],
            "hotpotqa": [
                "which band member of the beatles also had a successful solo career and was married to a japanese artist",
                "what company did the founder of spacex also start, and what is its primary product",
                "which actor starred in both titanic and inception",
                "what is the capital of the country that borders both france and germany",
            ],
            "bioasq": [
                "what is the role of BRCA1 in breast cancer",
                "how does metformin affect blood glucose levels",
                "what are the side effects of ACE inhibitors",
                "which genes are associated with Alzheimer's disease",
                "how does immunotherapy work for melanoma",
                "what is the mechanism of action of penicillin",
                "describe the pathology of Parkinson's disease",
            ],
            "finance": [
                "what is the impact of interest rate changes on bond prices",
                "how does quantitative easing affect inflation",
                "what is the difference between a put and a call option",
                "how do you calculate the weighted average cost of capital",
                "what factors influence exchange rate fluctuations",
                "what is the efficient market hypothesis",
            ],
        }

        domain_templates = templates.get(domain, templates["nq"])
        # Repeat/cycle if needed
        all_qs = (domain_templates * (n // len(domain_templates) + 1))[:n]

        queries = []
        for i, q_text in enumerate(all_qs):
            queries.append(Query(
                id=f"{domain}_synth_{i}",
                text=q_text,
                domain=domain,
                golden_answer="[synthetic — not real data]",
            ))
        logger.info(f"Generated {len(queries)} synthetic queries for {domain} (FALLBACK MODE)")
        return queries

    def load_corpus(
        self, domain: str, corpus_path: Optional[str] = None
    ) -> List[str]:
        """Load the corpus (document collection) for a domain.

        For RAG-QA Arena datasets, the corpus is typically the set of all
        passages/documents that will be indexed.

        Args:
            domain: Domain name.
            corpus_path: Optional path to a local corpus file (JSONL/CSV).

        Returns:
            List of document texts.
        """
        if corpus_path and Path(corpus_path).exists():
            return self._load_local_corpus(corpus_path)

        # For HuggingFace datasets, we use their passage collections
        domain_cfg = self.config.get_domain_config(domain)
        try:
            from datasets import load_dataset  # Lazy import
            dataset = load_dataset(domain_cfg.hf_path, )
            if isinstance(dataset, dict):
                dataset = dataset.get("corpus", dataset.get("train", list(dataset.values())[0]))

            texts = []
            for item in dataset:
                text = self._extract_field(item, "text") or self._extract_field(item, "passage")
                if text:
                    texts.append(str(text))
            return texts
        except Exception:
            logger.warning(f"Could not load corpus for {domain}, will use retrieved passages only")
            return []

    def _load_local_corpus(self, path: str) -> List[str]:
        """Load corpus from local JSONL or CSV file."""
        texts = []
        if path.endswith(".jsonl"):
            import json
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    text = item.get("text") or item.get("passage") or item.get("content")
                    if text:
                        texts.append(str(text))
        elif path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(path)
            for col in ["text", "passage", "content"]:
                if col in df.columns:
                    texts = df[col].astype(str).tolist()
                    break
        return texts


def load_domain_queries(
    config: ExperimentConfig,
    domain: str,
    max_queries: Optional[int] = None,
) -> List[Query]:
    """Convenience function: load queries for a domain."""
    loader = DatasetLoader(config)
    return loader.load_queries(domain, max_queries)
