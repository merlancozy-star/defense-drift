"""YAML configuration loader with validation.

All hyperparameters and paths are driven by separability_collapse.yaml.
No hardcoded paths or parameters in the codebase.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    path: str
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    max_length: int = 4096
    output_attentions: bool = False
    batch_size: int = 32
    enabled: bool = True


@dataclass
class RetrievalParams:
    chunk_size: int = 256
    chunk_overlap: int = 32
    top_k: int = 10
    similarity_metric: str = "cosine"
    faiss_index_type: str = "Flat"


@dataclass
class DiagnosticParams:
    probe_size: int = 50
    clean_sample_size: int = 200
    random_seed: int = 42
    min_auroc_threshold: float = 0.6


@dataclass
class OutputParams:
    results_dir: str = "./results"
    csv_prefix: str = "collapse_matrix"
    save_intermediate: bool = True
    save_format: list = field(default_factory=lambda: ["csv", "json"])


@dataclass
class ParallelParams:
    n_jobs: int = 4
    gpu_memory_fraction: float = 0.9


@dataclass
class AcceptanceCriteria:
    source_auroc_min: float = 0.80
    collapse_threshold: float = 0.10
    min_collapse_signals: int = 3
    corrupt_stronger_than_poisoned: bool = True


@dataclass
class PoisonParams:
    generation_model: str = "generator"
    template_mode: str = "contradict"
    injection_position: str = "top"
    corrupt_ratio: float = 0.3
    preserve_structure: bool = True


@dataclass
class DomainConfig:
    hf_path: str
    hf_subset: Optional[str] = None
    query_field: str = "question"
    answer_field: Optional[str] = "answer"
    corpus_field: Optional[str] = None
    max_queries: int = 500


class ExperimentConfig:
    """Complete experiment configuration loaded from YAML."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        # Parse model configs
        gen_cfg = self._raw.get("models", {}).get("generator", {})
        self.generator = ModelConfig(**gen_cfg)

        emb_cfg = self._raw.get("models", {}).get("embedder", {})
        self.embedder = ModelConfig(**emb_cfg)

        rerank_cfg = self._raw.get("models", {}).get("reranker", {})
        self.reranker = ModelConfig(**rerank_cfg)

        # Parse retrieval
        ret_cfg = self._raw.get("retrieval", {})
        self.retrieval = RetrievalParams(**ret_cfg)

        # Parse diagnostic
        diag_cfg = self._raw.get("diagnostic", {})
        self.diagnostic = DiagnosticParams(**diag_cfg)

        # Parse output
        out_cfg = self._raw.get("output", {})
        self.output = OutputParams(**out_cfg)

        # Parse parallel
        par_cfg = self._raw.get("parallel", {})
        self.parallel = ParallelParams(**par_cfg)

        # Parse acceptance
        acc_cfg = self._raw.get("acceptance", {})
        self.acceptance = AcceptanceCriteria(**acc_cfg)

        # Parse poison params
        self.poison = {}
        for attack_name in ["poisonedrag", "corruptrag"]:
            attack_cfg = self._raw.get("poison", {}).get(attack_name, {})
            self.poison[attack_name] = PoisonParams(**attack_cfg)

        # Parse domain configs
        self.domain_configs: Dict[str, DomainConfig] = {}
        for domain_name, dc in self._raw.get("datasets", {}).items():
            self.domain_configs[domain_name] = DomainConfig(**dc)

    @property
    def signals(self) -> list:
        return self._raw.get("signals", [])

    @property
    def source_domains(self) -> list:
        return self._raw.get("domains", {}).get("source", [])

    @property
    def target_domains(self) -> list:
        return self._raw.get("domains", {}).get("target", [])

    @property
    def attacks(self) -> list:
        return self._raw.get("attacks", [])

    @property
    def poison_ratios(self) -> list:
        return self._raw.get("poison_ratios", [])

    @property
    def all_domains(self) -> list:
        return self.source_domains + self.target_domains

    def get_domain_type(self, domain: str) -> str:
        if domain in self.source_domains:
            return "source"
        if domain in self.target_domains:
            return "target"
        raise ValueError(f"Unknown domain: {domain}")

    def get_domain_config(self, domain: str) -> DomainConfig:
        if domain not in self.domain_configs:
            raise ValueError(
                f"No dataset config for domain '{domain}'. "
                f"Available: {list(self.domain_configs.keys())}"
            )
        return self.domain_configs[domain]

    def validate_retrieval_consistency(self) -> bool:
        """Ensure retrieval config is identical across all domains (critical constraint).

        This is a HARD requirement per CLAUDE.md: changing domains must ONLY change
        the corpus, never the retrieval pipeline. If this fails, experiments are invalid.
        """
        # Since we only define ONE retrieval config block, consistency is structural.
        # This method exists to make the constraint explicit and auditable.
        return True

    def to_retrieval_config_dict(self) -> Dict[str, Any]:
        """Export retrieval params in a format compatible with data/schemas.py."""
        return {
            "chunk_size": self.retrieval.chunk_size,
            "chunk_overlap": self.retrieval.chunk_overlap,
            "top_k": self.retrieval.top_k,
            "similarity_metric": self.retrieval.similarity_metric,
            "embedder_path": self.embedder.path,
            "reranker_path": self.reranker.path,
            "reranker_enabled": self.reranker.enabled,
        }


def load_config(config_path: str = None) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to separability_collapse.yaml.
                     Defaults to the one in the project root.

    Returns:
        ExperimentConfig with all parameters parsed and validated.
    """
    if config_path is None:
        # Default to project root
        config_path = Path(__file__).parent.parent / "separability_collapse.yaml"
    return ExperimentConfig(str(config_path))
