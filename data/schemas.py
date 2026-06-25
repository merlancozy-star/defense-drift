"""Core data structures for the Defense Drift pipeline."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import json


@dataclass
class RetrievalConfig:
    """Immutable retrieval configuration — locked across source/target domains."""
    chunk_size: int = 256
    chunk_overlap: int = 32
    top_k: int = 10
    similarity_metric: str = "cosine"
    faiss_index_type: str = "Flat"
    embedder_path: str = ""
    reranker_path: str = ""
    reranker_enabled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __hash__(self):
        return hash((
            self.chunk_size, self.chunk_overlap, self.top_k,
            self.similarity_metric, self.embedder_path
        ))


@dataclass
class Passage:
    """A single passage (chunk) from the retrieval corpus."""
    id: str
    text: str
    embedding: Optional[List[float]] = None
    source_doc_id: Optional[str] = None
    is_poison: bool = False              # Known during diagnostic, hidden during inference
    poison_attack: Optional[str] = None  # "poisonedrag" | "corruptrag" | None
    retrieval_score: Optional[float] = None  # Similarity score from retriever


@dataclass
class Query:
    """A query with its golden answer and metadata."""
    id: str
    text: str
    domain: str                          # "nq" | "hotpotqa" | "bioasq" | "finance"
    golden_answer: Optional[str] = None
    retrieved_passages: List[Passage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoringUnit:
    """A single evaluation unit: one query × one passage × all signal scores.

    This is the atomic unit fed into AUROC calculation.
    """
    query_id: str
    passage_id: str
    domain: str
    is_poison: bool                      # Ground truth (known during diagnostic)
    attack: Optional[str]                # Attack type if poisoned, else None
    poison_ratio: float                  # Overall poison ratio in this batch
    signals: Dict[str, float] = field(default_factory=dict)  # signal_name → score

    def get_signal(self, name: str) -> Optional[float]:
        return self.signals.get(name)

    def set_signal(self, name: str, score: float):
        self.signals[name] = score


@dataclass
class CellResult:
    """One cell in Table 1 — the collapse matrix.

    Represents: signal × domain × attack × poison_ratio
    """
    signal: str
    domain: str
    domain_type: str                     # "source" | "target"
    attack: str
    poison_ratio: float
    auroc: float
    n_samples: int                       # Number of ScoringUnits used
    # Derived
    separability_drop: Optional[float] = None  # Δ_sep = AUROC(source) − AUROC(target)
    # Metadata
    source_auroc: Optional[float] = None  # Reference source-domain AUROC for this signal
    is_source_baseline: bool = False      # True if this cell IS the source baseline
    domain_robust: Optional[bool] = None  # True if Δ_sep ≈ 0

    def compute_drop(self, source_auroc: float):
        """Compute separability drop relative to source."""
        if not self.is_source_baseline:
            self.source_auroc = source_auroc
            self.separability_drop = source_auroc - self.auroc
            self.domain_robust = (self.separability_drop is not None
                                  and abs(self.separability_drop) < 0.05)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_row(self) -> Dict[str, Any]:
        """Flatten for CSV output — matches paper Table 1 columns."""
        return {
            "signal": self.signal,
            "domain": self.domain,
            "domain_type": self.domain_type,
            "attack": self.attack,
            "poison_ratio": self.poison_ratio,
            "auroc": round(self.auroc, 4),
            "separability_drop": round(self.separability_drop, 4) if self.separability_drop is not None else "",
            "n_samples": self.n_samples,
            "source_auroc": round(self.source_auroc, 4) if self.source_auroc is not None else "",
            "domain_robust": self.domain_robust if self.domain_robust is not None else "",
        }

    @staticmethod
    def csv_header() -> List[str]:
        return [
            "signal", "domain", "domain_type", "attack", "poison_ratio",
            "auroc", "separability_drop", "n_samples", "source_auroc", "domain_robust"
        ]
