"""
Feature extraction for one (query, partition) training sample.

Assembles a flat feature dict covering:
  - partition staleness signals  (version lag, mutation counts/fractions, drift)
  - query-partition relevance    (probe rank, centroid distance, gaps)
  - workload context             (template name, mutation type, drift level)
  - label targets                (fresh_recall, stale_recall, recall_drop)

Primary regression target: recall_drop  (continuous ∈ [0, 1])
The reconstruction-error computation mirrors worker_node._reconstruction_error
exactly (mean L2 distance from vectors to centroid), inlined here to avoid
instantiating WorkerNode (which connects to DynamoDB/MinIO/gRPC).
"""

from __future__ import annotations

import numpy as np

from offline_training.snapshot_builder import MutatedSnapshot


# ── Column groups (used by dataset_writer metadata and training scripts) ──────

ID_COLS = ["query_idx", "partition_id"]

FEATURE_COLS = [
    # partition staleness — derivable from WAL + cached partition data at runtime
    "version_lag",
    "partition_size",
    "update_fraction",
    "delete_fraction",
    "size_reduction_fraction",   # (stale_size - fresh_size) / stale_size: net size shrink (0 for updates, >0 for deletes)
    "recon_error_stale",         # mean L2 dist of cached vectors to centroid
    "recon_error_rel_delta",     # (recon_error_fresh - recon_error_stale) / recon_error_stale: proportional drift
    # query-partition relevance — derivable from centroid search at runtime
    "normalized_probe_rank",
    "centroid_dist",
    "rel_gap_to_prev",           # gap_to_prev_centroid / centroid_dist: scale-invariant margin to prev partition
    "rel_gap_to_next",           # gap_to_next_centroid / centroid_dist: scale-invariant margin to next partition
    "candidate_fraction",        # partition_size / total candidates across all probed partitions
    "historical_hit_rate",       # fraction of batch queries where this partition contributed ≥1 top-k result
]

REGRESSION_TARGET = "recall_drop"   # primary supervised target (continuous ∈ [0, 1])

# Not usable as model features: either requires fresh vectors (data leakage)
# or is a training-time label not observable at serve time.
# Kept in the dataset for analysis / stratified evaluation.
AUX_COLS = [
    # recall signals — useful for evaluation / error analysis
    "fresh_recall",
    "stale_recall",
    # raw mutation counts — fractions already in FEATURE_COLS; counts useful for debugging
    "update_count",
    "delete_count",
    "insert_count",
    "mutation_fraction",         # update_fraction + delete_fraction; kept for analysis
    "insert_fraction",           # always 0 (no insert mutations in templates) — kept for audit
    # recon error — absolute delta and fresh value kept for post-hoc analysis
    "recon_error_fresh",
    "recon_error_delta",         # absolute delta; rel version (recon_error_rel_delta) is the feature
    # absolute probe rank and gap values — relative/normalised versions are the features
    "probe_rank",
    "gap_to_prev_centroid",
    "gap_to_next_centroid",
    # training-time template metadata — not observable at runtime
    "workload_name",
    "mutation_type",
    "workload_mutation_fraction",
    "drift_level",
    "spatial_pattern",
    "query_type",
]

CATEGORICAL_FEATURES = []   # no categorical features after near_far_bucket removal


def _recon_error(centroid_vec: np.ndarray, vecs: np.ndarray) -> float:
    """Mean L2 distance from each vector to the centroid (mirrors worker_node logic)."""
    if len(vecs) == 0:
        return 0.0
    diffs = vecs.astype(np.float32) - centroid_vec
    return float(np.mean(np.sqrt(np.sum(diffs ** 2, axis=1))))


def extract_features_for_qp(
    query_idx: int,
    partition_id: int,
    probe_rank: int,
    n_probe: int,
    centroid_distances: list[float],
    snapshot: MutatedSnapshot,
    fresh_recall: float,
    stale_recall: float,
    recall_drop: float,
    probe_ids: list[int] | None = None,
    partition_hit_rates: dict[int, float] | None = None,
) -> dict:
    """
    Build one feature row for the (query_idx, partition_id) pair.
    All values are scalar or string — ready for pandas / parquet.
    """
    stale_entry = snapshot.stale_partitions.get(partition_id)
    fresh_entry = snapshot.fresh_partitions.get(partition_id)
    log         = snapshot.mutation_log.get(partition_id)

    if stale_entry is not None:
        # Affected partition: use the pre-mutation copy for staleness features
        stale_ids, stale_vecs, stale_version = stale_entry
        fresh_ids,  fresh_vecs, fresh_version = fresh_entry
    else:
        # Unaffected partition: stale = fresh = current state in all_partitions
        actual_ids, actual_vecs, actual_version = snapshot.all_partitions[partition_id]
        stale_ids, stale_vecs, stale_version = actual_ids, actual_vecs, actual_version
        fresh_ids, fresh_vecs, fresh_version  = actual_ids, actual_vecs, actual_version

    partition_size = int(len(stale_ids))
    fresh_size = int(len(fresh_ids))
    safe_size = max(partition_size, 1)

    updates  = log.updates  if log else 0
    deletes  = log.deletes  if log else 0
    inserts  = log.inserts  if log else 0
    version_lag = max(0, fresh_version - stale_version)

    update_fraction        = updates  / safe_size
    delete_fraction        = deletes  / safe_size
    insert_fraction        = inserts  / safe_size
    mutation_fraction      = (updates + deletes) / safe_size
    size_reduction_fraction = max(0.0, (partition_size - fresh_size) / safe_size)

    centroid_vec     = snapshot.centroids[partition_id]
    recon_stale      = _recon_error(centroid_vec, stale_vecs)
    recon_fresh      = _recon_error(centroid_vec, fresh_vecs)
    recon_delta      = recon_fresh - recon_stale

    # ── Query-partition relevance ────────────────────────────────────────────
    rank_idx      = probe_rank - 1
    centroid_dist = centroid_distances[rank_idx] if rank_idx < len(centroid_distances) else None
    dist_prev     = centroid_distances[rank_idx - 1] if rank_idx > 0 else None
    dist_next     = centroid_distances[rank_idx + 1] if rank_idx + 1 < len(centroid_distances) else None

    gap_to_prev = (centroid_dist - dist_prev)   if (centroid_dist is not None and dist_prev is not None) else None
    gap_to_next = (dist_next    - centroid_dist) if (centroid_dist is not None and dist_next is not None) else None
    _cd = max(centroid_dist, 1e-6) if centroid_dist is not None else None
    rel_gap_to_prev = round(float(gap_to_prev / _cd), 6) if (gap_to_prev is not None and _cd is not None) else None
    rel_gap_to_next = round(float(gap_to_next / _cd), 6) if (gap_to_next is not None and _cd is not None) else None

    # candidate_fraction: share of the probed candidate pool that lives in this partition.
    # Computed from stale partition sizes (what the cache holds at decision time).
    if probe_ids is not None:
        total_probe_size = sum(
            len(snapshot.all_partitions[c][0])
            for c in probe_ids
            if c in snapshot.all_partitions
        )
        candidate_fraction = round(partition_size / max(total_probe_size, 1), 6)
    else:
        candidate_fraction = None

    historical_hit_rate = round(partition_hit_rates[partition_id], 6) if partition_hit_rates and partition_id in partition_hit_rates else 0.0

    return {
        # identifiers
        "query_idx":                    query_idx,
        "partition_id":                 partition_id,

        # partition staleness
        "version_lag":                  version_lag,
        "partition_size":               partition_size,
        "update_count":                 updates,
        "delete_count":                 deletes,
        "insert_count":                 inserts,
        "update_fraction":              round(update_fraction, 6),
        "delete_fraction":              round(delete_fraction, 6),
        "size_reduction_fraction":      round(size_reduction_fraction, 6),
        "insert_fraction":              round(insert_fraction, 6),
        "mutation_fraction":            round(mutation_fraction, 6),
        "recon_error_stale":            round(recon_stale, 4),
        "recon_error_fresh":            round(recon_fresh, 4),
        "recon_error_delta":            round(recon_delta, 4),
        "recon_error_rel_delta":        round(recon_delta / max(recon_stale, 1e-6), 6),

        # query-partition relevance
        "probe_rank":                   probe_rank,
        "normalized_probe_rank":        round(probe_rank / max(n_probe, 1), 6),
        "centroid_dist":                round(float(centroid_dist), 4) if centroid_dist is not None else None,
        "gap_to_prev_centroid":         round(float(gap_to_prev), 4)  if gap_to_prev  is not None else None,
        "gap_to_next_centroid":         round(float(gap_to_next), 4)  if gap_to_next  is not None else None,
        "rel_gap_to_prev":              rel_gap_to_prev,
        "rel_gap_to_next":              rel_gap_to_next,
        "candidate_fraction":           candidate_fraction,
        "historical_hit_rate":          historical_hit_rate,
        # aux: training-time template metadata
        "workload_name":                snapshot.template.name,
        "mutation_type":                snapshot.template.mutation_type,
        "workload_mutation_fraction":   snapshot.template.mutation_fraction,
        "drift_level":                  snapshot.template.drift_magnitude,
        "spatial_pattern":              snapshot.template.spatial_pattern,

        # targets
        "fresh_recall":                 round(fresh_recall, 6),
        "stale_recall":                 round(stale_recall, 6),
        "recall_drop":                  round(recall_drop,  6),
    }
