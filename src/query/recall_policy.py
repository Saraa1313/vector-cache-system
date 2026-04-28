from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


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

    # Weight for candidate_fraction in importance score.
    # candidate_fraction is typically 1/n_probe ≈ 0.03 for equal-sized partitions;
    # large partitions can reach 0.1+. A weight of 2.0 adds 0.06–0.2 to importance.
    w_candidate_fraction: float = 2.0

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

    def _importance(self, probe_rank: int, candidate_fraction: float = 0.0) -> float:
        """
        Rank 1 = closest centroid = lowest importance. Farther partitions matter more for recall.
        candidate_fraction amplifies importance for partitions holding a large share of the
        candidate pool — these are more likely to contain true neighbors regardless of rank.
        """
        rank_importance = probe_rank / max(self.config.n_probe, 1)
        frac_importance = self.config.w_candidate_fraction * candidate_fraction
        return self._clamp01(rank_importance + frac_importance)

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

    def should_fetch(self, f: dict, probe_rank: int, candidate_fraction: float = 0.0,
                     centroid_distances: list[float] | None = None) -> tuple[bool, dict]:
        """
        f                  — freshness dict entry for this partition
        probe_rank         — 1-based rank by centroid distance (1 = closest)
        candidate_fraction — partition_size / total candidates across all probed partitions

        Returns (fetch: bool, diagnostics: dict)
        Only called for partitions already in the LRU cache.
        """
        if f["cached_version"] == f["latest_known_version"]:
            return False, {"reason": "fresh_in_cache", "staleness_score": 0.0, "risk": 0.0}

        staleness = self.compute_staleness_score(f)
        risk = self._clamp01(staleness * self._importance(probe_rank, candidate_fraction))
        fetch = risk >= self.config.fetch_threshold

        return fetch, {
            "reason": "risk_threshold",
            "staleness_score": round(staleness, 4),
            "risk": round(risk, 4),
            "candidate_fraction": round(candidate_fraction, 4),
            "cached_version": f["cached_version"],
            "latest_version": f["latest_known_version"],
        }


# Feature column order must match FEATURE_COLS in train_model.py / feature_extractor.py exactly.
_FEATURE_COLS = [
    "version_lag", "partition_size", "update_fraction", "delete_fraction",
    "recon_error_stale", "recon_error_rel_delta",
    "normalized_probe_rank", "centroid_dist",
    "rel_gap_to_prev", "rel_gap_to_next", "candidate_fraction",
    "historical_hit_rate",
]


class LearnedPolicy:
    """
    XGBoost-backed recall-drop predictor.

    Loads a saved .ubj model and predicts recall_drop for a cached partition.
    Exposes the same should_fetch() interface as RecallAwarePolicy so it can be
    swapped in without changing query_node.py call sites.

    Designed to run in shadow mode (called alongside RecallAwarePolicy but not
    acting on decisions) until v7 OOD metrics validate it for production.
    """

    def __init__(self, model_path: str, unsafe_threshold: float | None = None,
                 n_probe: int = 32, is_classifier: bool = False) -> None:
        import xgboost as xgb  # deferred — only needed when LearnedPolicy is instantiated
        import json, os
        if is_classifier:
            self.model = xgb.XGBClassifier()
        else:
            self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.n_probe = n_probe
        self.is_classifier = is_classifier

        # Load best_threshold from companion metrics JSON if not explicitly provided.
        if unsafe_threshold is not None:
            self.unsafe_threshold = unsafe_threshold
        else:
            metrics_path = model_path.replace(".ubj", "_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as fh:
                    self.unsafe_threshold = float(json.load(fh).get("best_threshold", 0.85))
            else:
                self.unsafe_threshold = 0.85
        print(f"LearnedPolicy loaded: {model_path} ({'classifier' if is_classifier else 'regressor'})"
              f" threshold={self.unsafe_threshold}", flush=True)

    def _assemble_features(
        self,
        f: dict,
        probe_rank: int,
        candidate_fraction: float,
        centroid_distances: list[float] | None,
    ) -> np.ndarray:
        partition_size = int(f.get("latest_partition_size") or 0)
        cached_size = int(f.get("cached_partition_size") or partition_size)
        denom = max(float(cached_size), 1e-9)

        updates = f.get("cumulative_updates_since_cache") or 0
        deletes = f.get("cumulative_deletes_since_cache") or 0
        inserts = f.get("cumulative_inserts_since_cache") or 0
        version_lag = max(0, (f.get("latest_known_version") or 0) - (f.get("cached_version") or 0))

        recon_stale = f.get("cached_reconstruction_error") or 0.0
        recon_latest = f.get("latest_reconstruction_error")
        recon_delta = (recon_latest - recon_stale) if recon_latest is not None else 0.0
        recon_rel_delta = recon_delta / max(recon_stale, 1e-6)

        rank_idx = probe_rank - 1
        if centroid_distances and rank_idx < len(centroid_distances):
            centroid_dist = centroid_distances[rank_idx]
            gap_prev = (centroid_dist - centroid_distances[rank_idx - 1]) if rank_idx > 0 else 0.0
            gap_next = (centroid_distances[rank_idx + 1] - centroid_dist) if (rank_idx + 1) < len(centroid_distances) else 0.0
            _cd = max(centroid_dist, 1e-6)
            rel_gap_prev = gap_prev / _cd
            rel_gap_next = gap_next / _cd
        else:
            centroid_dist = 0.0
            rel_gap_prev = 0.0
            rel_gap_next = 0.0

        historical_hit_rate = float(f.get("historical_hit_rate") or 0.0)

        feat = np.array([[
            version_lag,
            partition_size,
            updates / denom,
            deletes / denom,
            recon_stale,
            recon_rel_delta,
            probe_rank / max(self.n_probe, 1),
            centroid_dist,
            rel_gap_prev,
            rel_gap_next,
            candidate_fraction,
            historical_hit_rate,
        ]], dtype=np.float32)
        return feat

    def predict(
        self,
        f: dict,
        probe_rank: int,
        candidate_fraction: float = 0.0,
        centroid_distances: list[float] | None = None,
    ) -> float:
        """Return predicted score in [0, 1]: recall_drop (regressor) or P(unsafe) (classifier)."""
        X = self._assemble_features(f, probe_rank, candidate_fraction, centroid_distances)
        if self.is_classifier:
            return float(self.model.predict_proba(X)[0, 1])
        return float(np.clip(self.model.predict(X)[0], 0.0, 1.0))

    def should_fetch(
        self,
        f: dict,
        probe_rank: int,
        candidate_fraction: float = 0.0,
        centroid_distances: list[float] | None = None,
    ) -> tuple[bool, dict]:
        if f.get("cached_version") == f.get("latest_known_version"):
            return False, {"reason": "fresh_in_cache", "predicted_recall_drop": 0.0}
        pred = self.predict(f, probe_rank, candidate_fraction, centroid_distances)
        fetch = pred >= self.unsafe_threshold
        return fetch, {
            "reason": "learned_policy",
            "predicted_recall_drop": round(pred, 4),
            "unsafe_threshold": self.unsafe_threshold,
            "cached_version": f.get("cached_version"),
            "latest_version": f.get("latest_known_version"),
        }
