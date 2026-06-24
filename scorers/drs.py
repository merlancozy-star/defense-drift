"""DRSScorer — Directional Relative Shift (NeurIPS 2024).

Signal: effective poisoning attacks occur along directions where the clean
data distribution has SMALL variance. DRS measures each passage's deviation
from the clean distribution specifically along these "attack-prone" directions.

---

FORMULA (aligned with NeurIPS 2024 paper: "Understanding Data Poisoning
Attacks for RAG: Insights and Algorithms", OpenReview 2aL6gcFX7q):

Calibration (per domain):
  1. Collect N clean passage embeddings X = {e_1, ..., e_N} ⊂ R^d
  2. Compute empirical mean: μ = (1/N) Σ e_i
  3. Center: X_c = X - μ
  4. Eigendecompose covariance: Σ = (1/N) X_c^T X_c = V Λ V^T
     with eigenvalues λ_1 ≥ λ_2 ≥ ... ≥ λ_d (descending)
  5. Select low-variance directions:
     V_low = [v_{d-k+1}, ..., v_d]  (k eigenvectors with smallest eigenvalues)
     where k is determined by `variance_fraction` (bottom fraction of total variance)
  6. Store: μ (mean), V_low (attack-prone directions), λ_low (eigenvalues)

Scoring (per passage):
  1. Embed passage: e = embed(text)
  2. Center: Δ = e - μ
  3. Project onto attack-prone directions:
     DRS(x) = Σ_{j=1..k} (Δ · v_j)² / λ_j
     This is the Mahalanobis distance restricted to the low-variance subspace.

Direction: HIGHER DRS = passage deviates more from clean distribution
along attack-prone (low-variance) directions = more suspicious ✓

---

Key differences from the old PLACEHOLDER:
  OLD: shift = emb(p) - emb(query), direction = mean(shift)  → WRONG
  NEW: shift = emb(p) - μ_clean, direction = low-variance PCA eigenvectors  → CORRECT
  OLD: mean shift ≈ PC1 (HIGHEST variance)  → WRONG direction
  NEW: low-variance eigenvectors  → CORRECT attack-prone directions

Reference: "Understanding Data Poisoning Attacks for RAG: Insights and Algorithms"
NeurIPS 2024, OpenReview: https://openreview.net/forum?id=2aL6gcFX7q
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING
import logging

from scorers.base import BaseScorer

if TYPE_CHECKING:
    from retrieval.embedder import EmbedderWrapper

logger = logging.getLogger(__name__)


class DRSScorer(BaseScorer):
    """Score passages by Directional Relative Shift from clean distribution.

    Requires calibration on clean passage embeddings before scoring.
    Calibration computes the clean distribution's mean and low-variance
    PCA directions — the "attack-prone" subspace.

    Usage:
        scorer = DRSScorer(embedder=embedder)
        # Calibration phase (once per domain)
        clean_embs = embedder.encode_passages(clean_passages)
        scorer.calibrate(clean_embs)
        # Scoring phase
        scores = scorer.score(query, passages)
    """

    def __init__(
        self,
        embedder: Optional[EmbedderWrapper] = None,
        variance_fraction: float = 0.20,
        min_directions: int = 5,
        max_directions: int = 50,
        regularization: float = 1e-6,
    ):
        """
        Args:
            embedder: EmbedderWrapper for encoding passages.
            variance_fraction: Fraction of total variance to capture in
                the LOW-variance subspace (default 0.20 = bottom 20%).
            min_directions: Minimum number of low-variance directions.
            max_directions: Maximum number of low-variance directions.
            regularization: Small constant to prevent division by zero
                for near-zero eigenvalues (λ_reg = λ + ε).
        """
        self._embedder = embedder
        self.variance_fraction = variance_fraction
        self.min_directions = min_directions
        self.max_directions = max_directions
        self.regularization = regularization

        # Calibrated parameters (set by calibrate())
        self._mu: Optional[np.ndarray] = None          # [D] clean mean
        self._V_low: Optional[np.ndarray] = None       # [D, k] attack-prone directions
        self._lambda_low: Optional[np.ndarray] = None  # [k] eigenvalues
        self._calibrated: bool = False

    @property
    def name(self) -> str:
        return "drs"

    @property
    def needs_calibration(self) -> bool:
        """Whether this scorer needs calibration before scoring."""
        return True

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def set_embedder(self, embedder: EmbedderWrapper):
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, clean_embeddings: np.ndarray):
        """Calibrate DRS on clean passage embeddings.

        Computes PCA on the clean covariance matrix and identifies
        the low-variance (attack-prone) directions.

        Args:
            clean_embeddings: [N, D] array of clean passage embeddings.
                N should be >> D for stable covariance estimation.
                If N < D, uses a shrinkage estimator automatically.

        Raises:
            ValueError: If N < 2 (need at least 2 samples).
        """
        clean_embeddings = np.asarray(clean_embeddings, dtype=np.float64)
        N, D = clean_embeddings.shape

        if N < 2:
            raise ValueError(
                f"DRS calibration needs at least 2 clean samples, got {N}"
            )

        logger.info(f"DRS calibration: {N} samples, {D} dimensions")

        # 1. Compute mean
        self._mu = clean_embeddings.mean(axis=0)  # [D]

        # 2. Center
        X_c = clean_embeddings - self._mu  # [N, D]

        # 3. Eigendecompose covariance
        #    Σ = (1/N) X_c^T X_c
        #    Use SVD of X_c for numerical stability when N < D
        if N >= D:
            # Standard PCA: eigendecompose covariance
            cov = (X_c.T @ X_c) / N  # [D, D]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
        else:
            # N < D: use SVD of centered data (more stable)
            # X_c = U S V^T → eigenvalues = S² / N, eigenvectors = V
            _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            eigenvalues = (S ** 2) / N  # [min(N,D)]
            eigenvectors = Vt.T  # [D, min(N,D)]
            # Pad eigenvalues/eigenvectors to D if needed
            if len(eigenvalues) < D:
                eigenvalues = np.pad(eigenvalues, (0, D - len(eigenvalues)))
                # The remaining eigenvectors are arbitrary (zero eigenvalues)
                # Complete the basis with random orthonormal vectors
                remainder = D - eigenvectors.shape[1]
                if remainder > 0:
                    Q, _ = np.linalg.qr(
                        np.hstack([eigenvectors, np.eye(D)[:, :remainder]])
                    )
                    eigenvectors = Q

        # Sort descending (standard for PCA)
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # 4. Select low-variance directions
        k = self._select_k(eigenvalues)
        self._V_low = eigenvectors[:, -k:]    # [D, k] — last k = lowest variance
        self._lambda_low = eigenvalues[-k:]   # [k]

        # Regularize eigenvalues (prevent division by zero)
        self._lambda_low = self._lambda_low + self.regularization

        self._calibrated = True

        logger.info(
            f"DRS calibrated: μ shape={self._mu.shape}, "
            f"V_low shape={self._V_low.shape}, "
            f"λ_low range=[{self._lambda_low.min():.4e}, {self._lambda_low.max():.4e}], "
            f"variance captured in low subspace: "
            f"{self._lambda_low.sum() / eigenvalues.sum():.2%}"
        )

    def _select_k(self, eigenvalues: np.ndarray) -> int:
        """Select number of low-variance directions.

        Uses cumulative variance from the BOTTOM: sum of k smallest eigenvalues
        should equal `variance_fraction` of total variance.

        Clamped to [min_directions, max_directions].
        """
        total_var = eigenvalues.sum()
        if total_var < 1e-12:
            return self.min_directions

        # Cumulative from smallest upward
        cumsum_asc = np.cumsum(eigenvalues[::-1])  # smallest first
        target = total_var * self.variance_fraction

        k = int(np.searchsorted(cumsum_asc, target) + 1)
        k = max(self.min_directions, min(k, self.max_directions, len(eigenvalues)))
        return k

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        passages: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Compute DRS scores for passages.

        Requires prior calibration via `calibrate()`.

        Args:
            query: Query text (NOT used in DRS scoring — only passage embeddings
                   relative to clean distribution).
            passages: List of passage texts.

        Returns:
            np.ndarray of DRS scores, shape (len(passages),).
            Higher = more deviation along attack-prone directions = more suspicious.
        """
        if self._embedder is None:
            raise RuntimeError(
                "DRSScorer requires an EmbedderWrapper. Call set_embedder() first."
            )
        if not self._calibrated:
            raise RuntimeError(
                "DRSScorer requires calibration before scoring. "
                "Call calibrate(clean_embeddings) first."
            )

        n = len(passages)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Embed passages
        E = self._embedder.encode_passages(passages)  # [N, D]

        # Center relative to clean distribution mean
        delta = E - self._mu  # [N, D]

        # Project onto low-variance directions
        # proj shape: [N, k] = delta @ V_low
        proj = delta @ self._V_low  # [N, k]

        # DRS = Σ (proj_j)² / λ_j   (Mahalanobis in low-variance subspace)
        # Shape: [N]
        drs_scores = np.sum(proj ** 2 / self._lambda_low, axis=1)

        return drs_scores.astype(np.float64)

    def get_threshold(self, clean_drs_scores: np.ndarray, percentile: float = 95.0) -> float:
        """Compute classification threshold from clean DRS scores.

        Passages with DRS > threshold are flagged as potential poison.

        Args:
            clean_drs_scores: DRS scores of clean passages (after calibration).
            percentile: Percentile for threshold (default 95).

        Returns:
            Threshold value.
        """
        return float(np.percentile(clean_drs_scores, percentile))

    def classify(
        self,
        query: str,
        passages: List[str],
        threshold: Optional[float] = None,
        clean_drs_scores: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score and classify passages.

        Args:
            query: Query text.
            passages: List of passage texts.
            threshold: Explicit threshold. If None, computed from clean_drs_scores.
            clean_drs_scores: Clean DRS scores for threshold estimation.

        Returns:
            (scores, is_poison_pred): scores are DRS values, is_poison_pred is
            boolean array where True = predicted poison.
        """
        scores = self.score(query, passages)
        if threshold is None:
            if clean_drs_scores is None:
                threshold = 0.0  # fallback
            else:
                threshold = self.get_threshold(clean_drs_scores)
        is_poison = scores > threshold
        return scores, is_poison

    # ------------------------------------------------------------------
    # Serialization (for cross-domain reuse — not currently used,
    # but preserves calibration state across runs)
    # ------------------------------------------------------------------

    def get_calibration_state(self) -> dict:
        """Export calibration state for serialization."""
        if not self._calibrated:
            return {"calibrated": False}
        return {
            "calibrated": True,
            "mu": self._mu.tolist(),
            "V_low": self._V_low.tolist(),
            "lambda_low": self._lambda_low.tolist(),
            "variance_fraction": self.variance_fraction,
        }

    def set_calibration_state(self, state: dict):
        """Restore calibration state from serialized dict."""
        if not state.get("calibrated"):
            self._calibrated = False
            return
        self._mu = np.array(state["mu"], dtype=np.float64)
        self._V_low = np.array(state["V_low"], dtype=np.float64)
        self._lambda_low = np.array(state["lambda_low"], dtype=np.float64)
        self._calibrated = True
