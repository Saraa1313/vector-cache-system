"""
Builds mutated in-memory snapshots from clean IVF partitions.

Applies a WorkloadTemplate purely in memory and returns both the
pre-mutation (stale) and post-mutation (fresh) copies of affected partitions.
No writes to MinIO — this is entirely offline.

Spatial patterns
----------------
uniform             : mutate a random subset across all non-empty partitions
concentrated        : mutate the n_target_partitions largest partitions
top_k_contributor   : mutate the partitions that contributed most top-k results
                      to the actual labeling query batch (requires contribution_scores)
marginal_contributor: mutate partitions in the p50–p80 contribution band — these
                      sometimes contribute to top-k and produce mid-range recall_drop
                      (requires contribution_scores)

Contribution scores are computed externally via
common.partition_utils.compute_partition_contributions and passed in to
build_mutated_snapshot.  This aligns partition targeting with the exact query
batch used for labeling, eliminating the reference-query mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import faiss


# ── Drift scale per magnitude label ──────────────────────────────────────────

_DRIFT_SCALE: dict[str, float] = {
    "low":    10.0,
    "medium": 50.0,
    "high":  200.0,
}


# ── Workload template ─────────────────────────────────────────────────────────

@dataclass
class WorkloadTemplate:
    name: str
    mutation_type: str          # "update" | "delete" | "mixed"
    mutation_fraction: float    # fraction of each target partition to mutate per round
    drift_magnitude: str        # "low" | "medium" | "high"  (updates only)
    spatial_pattern: str        # "uniform" | "concentrated" | "top_k_contributor" | "rank_targeted"
    n_target_partitions: int = 12  # for concentrated/contributor patterns
    target_partition_ids: Optional[list[int]] = None  # hard override (skips auto-selection)
    rank_range: Optional[tuple[int, int]] = None  # (start, end) 1-based inclusive, for rank_targeted
    n_rounds: int = 1           # mutation rounds applied; version_lag = n_rounds in output
    rng_seed: int = 42
    query_types: tuple = ("random", "hot", "boundary")  # which query types to pair with


# ── Full template set ─────────────────────────────────────────────────────────
#
# Covers all required dimensions:
#   mutation_type    : update / delete / mixed
#   mutation_fraction: 5% / 10% / 20% / 30% / 50%
#   drift_magnitude  : low / medium / high  (for updates)
#   spatial_pattern  : uniform / concentrated / top_k_contributor / marginal_contributor
#
# Design principle: mutation fraction alone does not determine recall impact.
# Recall drop depends on fraction × where mutations land × how strongly the query
# depends on that partition.  Lower fractions (5/10%) cover near-boundary
# transitions; higher fractions (20/30/50%) on top_k_contributor / concentrated
# reliably produce non-zero recall drop signal.

TEMPLATES: list[WorkloadTemplate] = [

    # ── UNIFORM — mutations spread across all partitions ──────────────────────
    # All query types valid: no partition targeting, so no mismatch risk.
    # 5pct variants removed: with mutations spread across 1000+ partitions at 5%,
    # zero positives at the >=0.10 threshold — pure noise rows. affected_zero_sample_rate
    # in the pipeline keeps remaining uniform row counts in line with other patterns.

    WorkloadTemplate("update_uniform_10pct",
                     mutation_type="update", mutation_fraction=0.10,
                     drift_magnitude="medium", spatial_pattern="uniform",
                     query_types=("random", "hot", "boundary"),              rng_seed=11),

    WorkloadTemplate("update_uniform_20pct",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="uniform",
                     query_types=("random", "hot", "boundary"),              rng_seed=12),

    WorkloadTemplate("delete_uniform_10pct",
                     mutation_type="delete", mutation_fraction=0.10,
                     drift_magnitude="low",    spatial_pattern="uniform",
                     query_types=("random", "hot", "boundary"),              rng_seed=21),

    WorkloadTemplate("mixed_uniform_10pct",
                     mutation_type="mixed",  mutation_fraction=0.10,
                     drift_magnitude="medium", spatial_pattern="uniform",
                     query_types=("random", "hot", "boundary"),              rng_seed=30),

    # ── CONCENTRATED — largest partitions ────────────────────────────────────
    # Large partitions hold many vectors → more top-k contributions → stable
    # high-signal baseline that doesn't require query-batch alignment.

    WorkloadTemplate("update_concentrated_10pct",
                     mutation_type="update", mutation_fraction=0.10,
                     drift_magnitude="medium", spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=13),

    WorkloadTemplate("update_concentrated_20pct",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=14),

    WorkloadTemplate("update_concentrated_30pct",
                     mutation_type="update", mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=40),

    WorkloadTemplate("update_concentrated_50pct",
                     mutation_type="update", mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=51),

    WorkloadTemplate("delete_concentrated_10pct",
                     mutation_type="delete", mutation_fraction=0.10,
                     drift_magnitude="low",    spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=22),

    WorkloadTemplate("delete_concentrated_20pct",
                     mutation_type="delete", mutation_fraction=0.20,
                     drift_magnitude="low",    spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=23),

    WorkloadTemplate("delete_concentrated_30pct",
                     mutation_type="delete", mutation_fraction=0.30,
                     drift_magnitude="low",    spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=43),

    WorkloadTemplate("mixed_concentrated_20pct",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=31),

    WorkloadTemplate("mixed_concentrated_30pct",
                     mutation_type="mixed",  mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=45),

    # ── TOP-K CONTRIBUTOR — highest top-k contribution in the labeling batch ──
    # Partition selection uses contribution_scores computed from the same query
    # batch used for labeling → directly targets partitions containing true
    # neighbors → highest recall_drop signal.

    WorkloadTemplate("update_top_contributor_10pct",
                     mutation_type="update", mutation_fraction=0.10,
                     drift_magnitude="medium", spatial_pattern="top_k_contributor",
                     query_types=("hot", "boundary"),                        rng_seed=60),

    WorkloadTemplate("update_top_contributor_20pct",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     query_types=("hot", "boundary"),                        rng_seed=61),

    WorkloadTemplate("update_top_contributor_30pct",
                     mutation_type="update", mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     query_types=("hot", "boundary"),                        rng_seed=62),

    WorkloadTemplate("update_top_contributor_50pct",
                     mutation_type="update", mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     query_types=("hot",),                                   rng_seed=63),

    WorkloadTemplate("delete_top_contributor_10pct",
                     mutation_type="delete", mutation_fraction=0.10,
                     drift_magnitude="low",    spatial_pattern="top_k_contributor",
                     query_types=("hot", "boundary"),                        rng_seed=64),

    WorkloadTemplate("delete_top_contributor_20pct",
                     mutation_type="delete", mutation_fraction=0.20,
                     drift_magnitude="low",    spatial_pattern="top_k_contributor",
                     query_types=("hot", "boundary"),                        rng_seed=65),

    WorkloadTemplate("delete_top_contributor_30pct",
                     mutation_type="delete", mutation_fraction=0.30,
                     drift_magnitude="low",    spatial_pattern="top_k_contributor",
                     query_types=("hot",),                                   rng_seed=66),

    WorkloadTemplate("mixed_top_contributor_20pct",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     query_types=("hot", "boundary"),                        rng_seed=67),

    WorkloadTemplate("mixed_top_contributor_30pct",
                     mutation_type="mixed",  mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     query_types=("hot",),                                   rng_seed=68),

    WorkloadTemplate("mixed_top_contributor_50pct",
                     mutation_type="mixed",  mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     query_types=("hot",),                                   rng_seed=69),

    # High-fraction concentrated: fills delete/mixed at 50pct
    WorkloadTemplate("delete_concentrated_50pct",
                     mutation_type="delete", mutation_fraction=0.50,
                     drift_magnitude="low",    spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=52),

    WorkloadTemplate("mixed_concentrated_50pct",
                     mutation_type="mixed",  mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     query_types=("hot", "boundary"),                        rng_seed=53),

    # High-fraction top_k: fills delete gap at 40/50pct
    WorkloadTemplate("delete_top_contributor_40pct",
                     mutation_type="delete", mutation_fraction=0.40,
                     drift_magnitude="low",    spatial_pattern="top_k_contributor",
                     query_types=("hot",),                                   rng_seed=80),

    WorkloadTemplate("delete_top_contributor_50pct",
                     mutation_type="delete", mutation_fraction=0.50,
                     drift_magnitude="low",    spatial_pattern="top_k_contributor",
                     query_types=("hot",),                                   rng_seed=81),
]

# ── Multi-round templates (version_lag > 1) ───────────────────────────────────

MULTI_ROUND_TEMPLATES: list[WorkloadTemplate] = [

    # uniform lag2/lag3
    WorkloadTemplate("update_uniform_10pct_lag2",
                     mutation_type="update", mutation_fraction=0.10,
                     drift_magnitude="medium", spatial_pattern="uniform",
                     n_rounds=2, query_types=("random", "hot", "boundary"),  rng_seed=110),

    WorkloadTemplate("update_uniform_10pct_lag3",
                     mutation_type="update", mutation_fraction=0.10,
                     drift_magnitude="medium", spatial_pattern="uniform",
                     n_rounds=3, query_types=("random", "hot", "boundary"),  rng_seed=111),

    # concentrated lag2/lag3
    WorkloadTemplate("update_concentrated_20pct_lag2",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     n_rounds=2, query_types=("hot", "boundary"),            rng_seed=112),

    WorkloadTemplate("update_concentrated_20pct_lag3",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     n_rounds=3, query_types=("hot", "boundary"),            rng_seed=113),

    WorkloadTemplate("delete_concentrated_20pct_lag2",
                     mutation_type="delete", mutation_fraction=0.20,
                     drift_magnitude="low",    spatial_pattern="concentrated",
                     n_rounds=2, query_types=("hot", "boundary"),            rng_seed=114),

    WorkloadTemplate("mixed_concentrated_20pct_lag2",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="concentrated",
                     n_rounds=2, query_types=("hot", "boundary"),            rng_seed=115),

    # top_k_contributor lag2
    WorkloadTemplate("update_top_contributor_20pct_lag2",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     n_rounds=2, query_types=("hot", "boundary"),            rng_seed=116),

    WorkloadTemplate("update_top_contributor_30pct_lag2",
                     mutation_type="update", mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     n_rounds=2, query_types=("hot",),                       rng_seed=117),

    WorkloadTemplate("delete_top_contributor_20pct_lag2",
                     mutation_type="delete", mutation_fraction=0.20,
                     drift_magnitude="low",    spatial_pattern="top_k_contributor",
                     n_rounds=2, query_types=("hot", "boundary"),            rng_seed=118),

    WorkloadTemplate("mixed_top_contributor_20pct_lag2",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="top_k_contributor",
                     n_rounds=2, query_types=("hot", "boundary"),            rng_seed=119),

]

# ── Rank-targeted templates (probe_rank decorrelation) ───────────────────────
#
# Mutate partitions selected purely by their probe rank position across a query
# batch, independent of partition size or top-k contribution.  Covers three
# rank windows at two mutation fractions × two mutation types = 12 templates.
#
# Purpose: teach the model the effect of probe_rank independently of other
# features (size, contribution score), which are correlated with rank in all
# other spatial patterns.
#
# n_target_partitions:
#   close (1–4):  8 — small window but partitions recur heavily across queries
#   mid   (9–16): 8 — wider window, moderate recurrence
#   late (25–32): 8 — widest window, most spread across queries

RANK_TARGETED_TEMPLATES: list[WorkloadTemplate] = [

    # ── CLOSE ranks 1–4 ──────────────────────────────────────────────────────
    # These are the nearest centroids — most likely to hold true neighbors.
    # Current training data already has signal here (concentrated / top_k patterns
    # land here naturally), but rank is confounded with size and contribution.

    WorkloadTemplate("update_close_rank_20pct",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="medium", spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(1, 4),
                     query_types=("hot", "boundary"),                          rng_seed=200),

    WorkloadTemplate("update_close_rank_30pct",
                     mutation_type="update", mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(1, 4),
                     query_types=("hot", "boundary"),                          rng_seed=201),

    WorkloadTemplate("mixed_close_rank_20pct",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="medium", spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(1, 4),
                     query_types=("hot", "boundary"),                          rng_seed=202),

    WorkloadTemplate("mixed_close_rank_30pct",
                     mutation_type="mixed",  mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(1, 4),
                     query_types=("hot", "boundary"),                          rng_seed=203),

    # ── MID ranks 9–16 ───────────────────────────────────────────────────────
    # Middle of the probe list — moderate geometric relevance.
    # Underrepresented in current training: concentrated/top_k land at close
    # ranks; marginal_contributor labels are noisy.

    WorkloadTemplate("update_mid_rank_20pct",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="medium", spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=210),

    WorkloadTemplate("update_mid_rank_30pct",
                     mutation_type="update", mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=211),

    WorkloadTemplate("mixed_mid_rank_20pct",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="medium", spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=212),

    WorkloadTemplate("mixed_mid_rank_30pct",
                     mutation_type="mixed",  mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=213),

    WorkloadTemplate("update_mid_rank_40pct",
                     mutation_type="update", mutation_fraction=0.40,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=214),

    WorkloadTemplate("mixed_mid_rank_40pct",
                     mutation_type="mixed",  mutation_fraction=0.40,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=215),

    WorkloadTemplate("update_mid_rank_50pct",
                     mutation_type="update", mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=216),

    WorkloadTemplate("mixed_mid_rank_50pct",
                     mutation_type="mixed",  mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(9, 16),
                     query_types=("hot", "boundary"),                          rng_seed=217),

    # ── LATE ranks 25–32 ─────────────────────────────────────────────────────
    # Far end of the probe list — geometrically distant, rarely contain true
    # neighbors. Model currently sees almost no training signal here since
    # concentrated/top_k patterns never target these partitions.
    # High mutation fractions needed to produce any recall_drop signal.

    WorkloadTemplate("update_late_rank_20pct",
                     mutation_type="update", mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=220),

    WorkloadTemplate("update_late_rank_30pct",
                     mutation_type="update", mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=221),

    WorkloadTemplate("mixed_late_rank_20pct",
                     mutation_type="mixed",  mutation_fraction=0.20,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=222),

    WorkloadTemplate("mixed_late_rank_30pct",
                     mutation_type="mixed",  mutation_fraction=0.30,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=223),

    WorkloadTemplate("update_late_rank_40pct",
                     mutation_type="update", mutation_fraction=0.40,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=224),

    WorkloadTemplate("mixed_late_rank_40pct",
                     mutation_type="mixed",  mutation_fraction=0.40,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=225),

    WorkloadTemplate("update_late_rank_50pct",
                     mutation_type="update", mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=226),

    WorkloadTemplate("mixed_late_rank_50pct",
                     mutation_type="mixed",  mutation_fraction=0.50,
                     drift_magnitude="high",   spatial_pattern="rank_targeted",
                     n_target_partitions=8, rank_range=(25, 32),
                     query_types=("hot", "boundary"),                          rng_seed=227),
]

# Full set used by the pipeline
TEMPLATES = TEMPLATES + MULTI_ROUND_TEMPLATES + RANK_TARGETED_TEMPLATES

# Convenience alias: lightweight smoke-test templates
FIRST_VERSION_TEMPLATES: list[WorkloadTemplate] = [
    next(t for t in TEMPLATES if t.name == "update_uniform_10pct"),
    next(t for t in TEMPLATES if t.name == "update_concentrated_20pct"),
    next(t for t in TEMPLATES if t.name == "delete_concentrated_20pct"),
]


# ── Per-partition mutation bookkeeping ────────────────────────────────────────

@dataclass
class MutationLog:
    inserts: int = 0
    updates: int = 0
    deletes: int = 0
    version_lag: int = 1   # always 1 per single-batch snapshot


# ── Snapshot result ───────────────────────────────────────────────────────────

@dataclass
class MutatedSnapshot:
    template: WorkloadTemplate
    stale_partitions: dict[int, tuple]   # pre-mutation copies of affected partitions
    fresh_partitions: dict[int, tuple]   # post-mutation copies of affected partitions
    all_partitions: dict[int, tuple]     # full index: fresh for affected, orig for rest
    affected_cids: list[int]
    mutation_log: dict[int, MutationLog]
    centroids: np.ndarray


# ── Partition selection ───────────────────────────────────────────────────────

def _select_target_partitions(
    partitions: dict[int, tuple],
    template: WorkloadTemplate,
    rng: np.random.Generator,
    contribution_scores: Optional[dict[int, int]] = None,
) -> list[int]:
    """
    Determine which partition cids to mutate based on the template's spatial_pattern.

    contribution_scores: {partition_id: n_topk_results_contributed} computed from
    the actual labeling query batch via compute_partition_contributions().
    Required for top_k_contributor and marginal_contributor patterns.
    """
    if template.target_partition_ids is not None:
        return list(template.target_partition_ids)

    non_empty = [cid for cid, (ids, _, _) in partitions.items() if len(ids) > 0]
    n = template.n_target_partitions

    if template.spatial_pattern == "uniform":
        return non_empty

    if template.spatial_pattern == "concentrated":
        return sorted(non_empty, key=lambda c: len(partitions[c][0]), reverse=True)[:n]

    # Contribution-based patterns — fall back to partition size if scores missing
    if contribution_scores is None:
        return sorted(non_empty, key=lambda c: len(partitions[c][0]), reverse=True)[:n]

    if template.spatial_pattern == "top_k_contributor":
        # Partitions that contributed the most top-k results to the query batch
        return sorted(non_empty, key=lambda c: contribution_scores.get(c, 0), reverse=True)[:n]

    if template.spatial_pattern == "marginal_contributor":
        # Partitions in the p50–p80 contribution band: sometimes contribute
        # to top-k → produces mid-range recall_drop (boundary cases)
        scored = sorted(non_empty, key=lambda c: contribution_scores.get(c, 0))
        total  = len(scored)
        p50    = scored[total // 2 :]          # upper half by contribution
        p80    = scored[: int(total * 0.8)]    # lower 80%
        mid_band = [c for c in p50 if c in set(p80)]  # intersection = p50–p80
        if len(mid_band) < n:
            mid_band = scored[total // 2 :]    # fallback: upper half
        # Within the band, pick the highest contributors (most signal)
        return sorted(mid_band, key=lambda c: contribution_scores.get(c, 0), reverse=True)[:n]

    if template.spatial_pattern == "rank_targeted":
        # Partitions that most frequently appear at the target probe rank positions
        # across the query batch. contribution_scores here is rank appearance counts
        # computed via compute_rank_targeted_scores(), same {cid: count} structure.
        if contribution_scores is None:
            return sorted(non_empty, key=lambda c: len(partitions[c][0]), reverse=True)[:n]
        return sorted(non_empty, key=lambda c: contribution_scores.get(c, 0), reverse=True)[:n]

    raise ValueError(f"Unknown spatial_pattern: {template.spatial_pattern!r}")


# ── Mutation application ──────────────────────────────────────────────────────

def _apply_update(
    ids: np.ndarray, vecs: np.ndarray,
    n_mutate: int, drift_scale: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_mutate = min(n_mutate, len(ids))
    if n_mutate == 0:
        return ids, vecs, 0
    chosen   = rng.choice(len(ids), size=n_mutate, replace=False)
    noise    = rng.standard_normal((n_mutate, vecs.shape[1])).astype(np.float32) * drift_scale
    new_vecs = vecs.copy()
    new_vecs[chosen] += noise
    return ids, new_vecs, n_mutate


def _apply_delete(
    ids: np.ndarray, vecs: np.ndarray,
    n_delete: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_delete = min(n_delete, len(ids))
    if n_delete == 0:
        return ids, vecs, 0
    chosen = rng.choice(len(ids), size=n_delete, replace=False)
    mask   = np.ones(len(ids), dtype=bool)
    mask[chosen] = False
    return ids[mask], vecs[mask], n_delete


# ── Public API ────────────────────────────────────────────────────────────────

def build_mutated_snapshot(
    partitions: dict[int, tuple],
    centroids: np.ndarray,
    template: WorkloadTemplate,
    contribution_scores: Optional[dict[int, int]] = None,
) -> MutatedSnapshot:
    """
    Apply template.n_rounds rounds of mutations to in-memory partitions.

    partitions          : {cid: (ids, vecs, version)} from ObjectStore.load_centroid
    contribution_scores : {partition_id: n_topk_results_contributed} — required for
                          top_k_contributor and marginal_contributor spatial patterns.
                          Computed externally via compute_partition_contributions()
                          using the same query batch that will be used for labeling.

    Each round applies mutation_fraction of the *current* partition size and
    increments the version by 1, so after n_rounds:
        version_lag      = n_rounds
        stale_partitions = pre-round-1 state  (what the cache holds)
        fresh_partitions = post-round-N state  (authoritative current state)
        mutation_log     = accumulated counts across all rounds

    Returns MutatedSnapshot; does not write to MinIO.
    """
    rng         = np.random.default_rng(template.rng_seed)
    drift_scale = _DRIFT_SCALE.get(template.drift_magnitude, 50.0)
    target_cids = _select_target_partitions(
        partitions, template, rng, contribution_scores
    )

    # ── Freeze stale copies before any mutations ──────────────────────────────
    stale_partitions: dict[int, tuple] = {}
    for cid in target_cids:
        ids, vecs, version = partitions[cid]
        if len(ids) > 0:
            stale_partitions[cid] = (ids.copy(), vecs.copy(), version)

    # Working state: evolves through all rounds; starts from original partitions
    working: dict[int, tuple] = dict(partitions)

    # Accumulated mutation counts across all rounds
    accumulated_log: dict[int, MutationLog] = {
        cid: MutationLog() for cid in stale_partitions
    }

    # ── Apply n_rounds rounds ─────────────────────────────────────────────────
    for _round in range(template.n_rounds):
        for cid in stale_partitions:          # only mutate partitions we froze
            ids, vecs, version = working[cid]

            if len(ids) == 0:
                continue                      # partition emptied by prior deletes

            n_mutate = max(1, int(len(ids) * template.mutation_fraction))
            log      = accumulated_log[cid]

            if template.mutation_type == "update":
                new_ids, new_vecs, n_done = _apply_update(ids, vecs, n_mutate, drift_scale, rng)
                log.updates += n_done

            elif template.mutation_type == "delete":
                new_ids, new_vecs, n_done = _apply_delete(ids, vecs, n_mutate, rng)
                log.deletes += n_done

            else:  # mixed: half update, half delete
                n_upd = n_mutate // 2
                n_del = n_mutate - n_upd
                tmp_ids, tmp_vecs, n_upd_done = _apply_update(ids, vecs, n_upd, drift_scale, rng)
                new_ids, new_vecs, n_del_done = _apply_delete(tmp_ids, tmp_vecs, n_del, rng)
                log.updates += n_upd_done
                log.deletes += n_del_done

            # Version increments once per round — after n_rounds: version = original + n_rounds
            working[cid] = (new_ids, new_vecs, version + 1)

    # ── Stamp version_lag on every log entry ──────────────────────────────────
    for log in accumulated_log.values():
        log.version_lag = template.n_rounds

    # fresh_partitions = state after all rounds
    fresh_partitions: dict[int, tuple] = {cid: working[cid] for cid in stale_partitions}

    all_partitions = dict(partitions)
    for cid, entry in fresh_partitions.items():
        all_partitions[cid] = entry

    return MutatedSnapshot(
        template=template,
        stale_partitions=stale_partitions,
        fresh_partitions=fresh_partitions,
        all_partitions=all_partitions,
        affected_cids=list(fresh_partitions.keys()),
        mutation_log=accumulated_log,
        centroids=centroids,
    )
