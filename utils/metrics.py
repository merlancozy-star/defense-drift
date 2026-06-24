"""AUROC and related metrics for separability measurement."""

import numpy as np
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Optional


def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    poison_is_positive: bool = True,
) -> float:
    """Compute AUROC for a signal's ability to separate clean vs poison passages.

    Args:
        scores: Signal scores for each passage (higher = more suspicious).
        labels: Ground truth labels (1 = poison, 0 = clean).
        poison_is_positive: If True, poison=1 is the positive class (standard).

    Returns:
        AUROC score in [0, 1]. NaN if only one class present.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    # Filter NaN scores
    valid = ~np.isnan(scores)
    scores = scores[valid]
    labels = labels[valid]

    if len(np.unique(labels)) < 2:
        return float("nan")

    try:
        if poison_is_positive:
            return float(roc_auc_score(labels, scores))
        else:
            return float(roc_auc_score(labels, -scores))
    except ValueError:
        return float("nan")


def compute_separability_drop(
    source_auroc: float,
    target_auroc: float,
) -> float:
    """Compute separability collapse: Δ_sep = AUROC(source) − AUROC(target).

    Positive values indicate collapse (target-domain separability is worse).
    Negative values indicate improvement (unexpected but recorded).
    """
    if np.isnan(source_auroc) or np.isnan(target_auroc):
        return float("nan")
    return source_auroc - target_auroc


def bootstrap_auroc_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap AUROC confidence interval.

    Returns:
        (auroc_mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)

    aurocs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        auroc = compute_auroc(scores[idx], labels[idx])
        if not np.isnan(auroc):
            aurocs.append(auroc)

    if not aurocs:
        return float("nan"), float("nan"), float("nan")

    aurocs = np.array(auroc)
    lower = np.percentile(aurocs, (1 - ci) / 2 * 100)
    upper = np.percentile(aurocs, (1 + ci) / 2 * 100)
    return float(np.mean(aurocs)), float(lower), float(upper)


def verify_score_direction(
    scores: np.ndarray,
    labels: np.ndarray,
    signal_name: str = "unknown",
) -> bool:
    """Verify that higher scores correspond to poison (positive class).

    Returns True if direction is correct (AUROC > 0.5 when poison is positive).
    Prints a warning if direction appears reversed.
    """
    auroc = compute_auroc(scores, labels, poison_is_positive=True)
    if np.isnan(auroc):
        print(f"[WARN] {signal_name}: cannot verify direction (only one class)")
        return False

    if auroc < 0.5:
        print(
            f"[WARN] {signal_name}: AUROC={auroc:.4f} < 0.5 — "
            f"scores may be reversed! Higher scores should indicate poison."
        )
        return False
    return True
