# data — dataset loading and schemas
from .schemas import Passage, Query, ScoringUnit, CellResult, RetrievalConfig

# Lazy imports for heavy dependencies (datasets, transformers)
# Import loader functions only when needed — they require `datasets` package.

__all__ = [
    "Passage",
    "Query",
    "ScoringUnit",
    "CellResult",
    "RetrievalConfig",
]
