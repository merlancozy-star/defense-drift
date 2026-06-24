"""ClusterDistanceScorer — 2-means outlier distance via embedder.

Signal: embed poison passages → they tend to form a distinct cluster from
clean passages. Distance to the NEAREST cluster centroid (clean or poison)
serves as an anomaly score.

Implementation: K-means with K=2 on the passage embeddings.
The intuition from TrustRAG: poison passages cluster together in embedding
space, and their distance to cluster centroids can be used for filtering.

HIGHER distance → more anomalous → more suspicious.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from scorers.base import BaseScorer

if TYPE_CHECKING:
    from retrieval.embedder import EmbedderWrapper


class ClusterDistanceScorer(BaseScorer):
    """Score passages by distance to nearest 2-means cluster centroid.

    Uses the embedder to compute passage embeddings, then runs K-means (K=2).
    The score for each passage is its Euclidean distance to the NEARER centroid.

    Intuition (from TrustRAG):
    - Clean passages cluster tightly around one centroid
    - Poison passages are outliers relative to the clean cluster
    - The distance to the nearest centroid captures this anomalousness
    """

    def __init__(self, embedder: Optional[EmbedderWrapper] = None):
        """
        Args:
            embedder: A loaded EmbedderWrapper. If None, set later via set_embedder().
        """
        self._embedder = embedder

    @property
    def name(self) -> str:
        return "cluster_distance"

    def set_embedder(self, embedder: EmbedderWrapper):
        """Set or replace the embedder (used for lazy loading)."""
        self._embedder = embedder

    def score(
        self,
        query: str,
        passages: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Compute cluster-distance scores for passages.

        Steps:
        1. Embed all passages
        2. Run K-means (K=2)
        3. For each passage, compute distance to NEARER centroid
        4. Return these distances as suspicion scores

        Note: The query is NOT used in clustering — only passage embeddings matter.
        """
        if self._embedder is None:
            raise RuntimeError(
                "ClusterDistanceScorer requires an EmbedderWrapper. "
                "Call set_embedder() before scoring."
            )

        n = len(passages)
        if n < 3:
            # Too few passages for meaningful clustering
            return np.zeros(n, dtype=np.float64)

        # Lazy import sklearn (heavy dependency)
        from sklearn.cluster import KMeans

        # Embed
        embeddings = self._embedder.encode_passages(passages)

        # K-means with K=2
        k = min(2, n - 1)  # Ensure k < n
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)

        # Distance to NEARER centroid
        centroids = kmeans.cluster_centers_  # [k, dim]
        distances = np.zeros(n, dtype=np.float64)
        for i in range(n):
            dists = np.linalg.norm(embeddings[i] - centroids, axis=1)
            distances[i] = np.min(dists)  # Nearest centroid distance

        # Normalize to [0, 1] for stability across domains
        d_max = distances.max()
        if d_max > 0:
            distances = distances / d_max

        return distances

    # Direction: higher distance = more anomalous = more suspicious ✓
