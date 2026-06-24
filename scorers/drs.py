"""DRSScorer — Directional Relative Shift.

Signal: embed the query and each passage, then measure how far each passage's
embedding deviates from the query in a specific "attack-prone direction."

---

IMPORTANT (CLAUDE.md hard constraint):
This is a PLACEHOLDER implementation. The exact DRS formula from the NeurIPS'24
paper must be verified and calibrated before this scorer is used for results.

Do NOT rely on this implementation without first verifying against the original
paper's definition of Directional Relative Shift.

Current placeholder logic:
1. Embed query q and each passage p_i
2. Compute shift vector: s_i = emb(p_i) - emb(q)
3. Compute "attack-prone direction" d as the mean shift over all passages
4. DRS score = |s_i · d| / |d| — projection magnitude onto the attack direction

TODO (M2):
- [ ] Look up exact DRS formula from the NeurIPS'24 paper
- [ ] Verify direction normalization matches paper
- [ ] Verify score direction: higher = more poison-like (may need sign flip)
- [ ] Calibrate against published AUROC numbers on NQ
---

Reference: DRS (Directional Relative Shift), NeurIPS 2024.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, TYPE_CHECKING

from scorers.base import BaseScorer

if TYPE_CHECKING:
    from retrieval.embedder import EmbedderWrapper


class DRSScorer(BaseScorer):
    """Score passages by Directional Relative Shift from query embedding.

    PLACEHOLDER — see module docstring for calibration TODO.
    """

    def __init__(self, embedder: Optional[EmbedderWrapper] = None):
        self._embedder = embedder

    @property
    def name(self) -> str:
        return "drs"

    def set_embedder(self, embedder: EmbedderWrapper):
        self._embedder = embedder

    def score(
        self,
        query: str,
        passages: List[str],
        **kwargs,
    ) -> np.ndarray:
        """Compute DRS scores for passages relative to query.

        Placeholder algorithm:
        1. Embed query → q
        2. Embed each passage → p_i
        3. Shift vectors: s_i = p_i - q
        4. Attack-prone direction d = mean(s_i)  (normalized)
        5. DRS_i = |projection of s_i onto d|
        """
        if self._embedder is None:
            raise RuntimeError(
                "DRSScorer requires an EmbedderWrapper. Call set_embedder() before scoring."
            )

        n = len(passages)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Embed query and passages
        q = self._embedder.encode_query(query)  # [D]
        P = self._embedder.encode_passages(passages)  # [N, D]

        # Shift vectors
        shifts = P - q  # [N, D]

        # Attack-prone direction: mean shift (unit vector)
        d = shifts.mean(axis=0)  # [D]
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-12:
            return np.zeros(n, dtype=np.float64)
        d_unit = d / d_norm

        # Projection magnitude onto attack direction
        drs_scores = np.abs(np.dot(shifts, d_unit))  # [N]

        # Normalize to [0, 1]
        smax = drs_scores.max()
        if smax > 0:
            drs_scores = drs_scores / smax

        return drs_scores.astype(np.float64)

    # Direction check: higher projection onto attack direction = more suspicious
    # This MAY need sign adjustment after calibrating against paper.
