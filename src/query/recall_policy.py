from dataclasses import dataclass
from typing import Optional


@dataclass
class PolicyConfig:
    # Staleness score weights (b1..b5)
    w_version_lag: float = 0.158    # b1: version drift
    w_mutation: float = 0.032       # b2: (updates + deletes) / partition_size since cache
    w_update: float = 0.180         # b3: updates / partition_size since cache
    w_delete: float = 0.180         # b4: deletes / partition_size since cache
    w_reconstruction: float = 0.450 # b5: structural drift via reconstruction error

    # Fetch if risk >= this threshold
    fetch_threshold: float = 0.0046

    # Normalization caps
    max_version_lag_for_norm: int = 10
    max_reconstruction_error_for_norm: float = 500.0  # tuned for SIFT1M RMSE (~300-400 typical)

    # n_probe needed for importance normalization
    n_probe: int = 10

    eps: float = 1e-9


class RecallAwarePolicy:
    """
    Stateless scorer — takes the freshness dict entry for a partition and
    decides whether to fetch from MinIO or use the cached copy.

    Decision:
        risk = staleness_score(partition) * importance(probe_rank)
        risk >= fetch_threshold → fetch from MinIO
        risk <  fetch_threshold → use cache
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        self.config = config or PolicyConfig()

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def _importance(self, probe_rank: int) -> float:
        """Rank 1 = closest centroid = lowest importance. Farther partitions matter more for recall."""
        return probe_rank / max(self.config.n_probe, 1)

    def compute_staleness_score(self, f: dict) -> float:
        """
        Computes staleness in [0, 1] from a freshness dict entry.
        All signals normalized before weighting.
        """
        cfg = self.config

        if f["cached_version"] == f["latest_known_version"]:
            return 0.0  # no versions have arrived since cache — fully fresh

        denom = max(float(f["latest_partition_size"] or 0), cfg.eps)

        mutation_fraction = (f["cumulative_updates_since_cache"] + f["cumulative_deletes_since_cache"]) / denom
        update_fraction   = f["cumulative_updates_since_cache"] / denom
        delete_fraction   = f["cumulative_deletes_since_cache"] / denom
        version_lag       = max(0, f["latest_known_version"] - f["cached_version"])
        re                = f["latest_reconstruction_error"] or 0.0

        norm_version_lag = self._clamp01(version_lag / max(cfg.max_version_lag_for_norm, 1))
        norm_re          = self._clamp01(re / max(cfg.max_reconstruction_error_for_norm, cfg.eps))

        score = (
            cfg.w_version_lag    * norm_version_lag +
            cfg.w_mutation       * self._clamp01(mutation_fraction) +
            cfg.w_update         * self._clamp01(update_fraction) +
            cfg.w_delete         * self._clamp01(delete_fraction) +
            cfg.w_reconstruction * norm_re
        )
        return self._clamp01(score)

    def should_fetch(self, f: dict, probe_rank: int) -> tuple[bool, dict]:
        """
        f         — freshness dict entry for this partition
        probe_rank — 1-based rank by centroid distance (1 = closest)

        Returns (fetch: bool, diagnostics: dict)
        Only called for partitions already in the LRU cache.
        """
        if f["cached_version"] == f["latest_known_version"]:
            return False, {"reason": "fresh_in_cache", "staleness_score": 0.0, "risk": 0.0}

        staleness = self.compute_staleness_score(f)
        risk = self._clamp01(staleness * self._importance(probe_rank))
        fetch = risk >= self.config.fetch_threshold

        return fetch, {
            "reason": "risk_threshold",
            "staleness_score": round(staleness, 4),
            "risk": round(risk, 4),
            "cached_version": f["cached_version"],
            "latest_version": f["latest_known_version"],
        }
