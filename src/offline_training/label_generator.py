"""
Stale-intervention labeling.

For each (query q, probed partition p) pair:

  label(q, p) = fresh_recall(q) - recall(q | only p is served stale)

Protocol:
  1. Run query with ALL probed partitions fresh → fresh_recall
  2. For each affected partition p in the probe set:
       - replace only p with its pre-mutation stale copy
       - re-run the same query
       - compute recall against post-mutation GT
  3. recall_drop = fresh_recall - stale_recall_p

Labels near 0 → safe to serve p from cache.
Large labels → stale p threatens recall and should be fetched.

Reuses:
  - common.partition_utils.probe_centroids, ivf_search  (new helpers)
  - client.client.compute_recall                        (existing batch recall fn)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from common.partition_utils import probe_centroids, ivf_search
from client.client import compute_recall as _batch_recall


def _recall_at_k(pred_ids: list[int], gt_ids: list[int], k: int) -> float:
    """Single-query wrapper around the existing batch recall function."""
    return _batch_recall([pred_ids], [gt_ids], k)


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class FreshResult:
    query_idx: int
    query_vec: np.ndarray
    probe_ids: list[int]
    centroid_distances: list[float]
    fresh_topk_ids: list[int]
    fresh_recall: float


@dataclass
class LabelRow:
    query_idx: int
    partition_id: int
    probe_rank: int          # 1-based, ascending distance
    fresh_recall: float
    stale_recall: float
    recall_drop: float


# ── Core functions ────────────────────────────────────────────────────────────

def run_fresh_query_batch(
    queries_with_idx: list[tuple[int, np.ndarray]],
    snapshot,                          # MutatedSnapshot
    centroid_index,
    n_probe: int,
    topk: int,
    gt_lookup: Optional[dict[int, np.ndarray]] = None,
) -> list[FreshResult]:
    """
    Run every query against the fully-fresh partition set (snapshot.all_partitions).

    gt_lookup: {query_idx: ground_truth_ids_array}  — if None, fresh_recall = 1.0
               (relative recall drop is still valid without absolute GT).
    """
    results = []
    for q_idx, q_vec in queries_with_idx:
        probe_ids, dists = probe_centroids(centroid_index, q_vec, n_probe)
        topk_ids = ivf_search(q_vec, snapshot.all_partitions, probe_ids, topk)

        if gt_lookup is not None and q_idx in gt_lookup:
            recall = _recall_at_k(topk_ids, gt_lookup[q_idx].tolist(), topk)
        else:
            recall = 1.0

        results.append(FreshResult(
            query_idx=q_idx,
            query_vec=q_vec,
            probe_ids=probe_ids,
            centroid_distances=dists,
            fresh_topk_ids=topk_ids,
            fresh_recall=recall,
        ))
    return results


def run_stale_intervention_for_partition(
    fresh_result: FreshResult,
    snapshot,
    stale_cid: int,
    topk: int,
    gt_ids_for_query: Optional[list[int]],
) -> float:
    """
    Replace only partition stale_cid with its pre-mutation copy, re-run query.
    Returns stale recall for this single-partition intervention.
    """
    if stale_cid not in snapshot.stale_partitions:
        return fresh_result.fresh_recall  # partition not mutated — no effect

    temp_partitions = dict(snapshot.all_partitions)
    temp_partitions[stale_cid] = snapshot.stale_partitions[stale_cid]

    topk_ids = ivf_search(
        fresh_result.query_vec, temp_partitions, fresh_result.probe_ids, topk
    )

    if gt_ids_for_query is not None:
        return _recall_at_k(topk_ids, gt_ids_for_query, topk)

    # No GT: measure overlap with fresh top-k (relative recall drop)
    gt_set = set(fresh_result.fresh_topk_ids[:topk])
    found = sum(1 for pid in topk_ids[:topk] if pid in gt_set)
    return found / topk if topk > 0 else 0.0


def compute_all_labels(
    queries_with_idx: list[tuple[int, np.ndarray]],
    snapshot,
    centroid_index,
    n_probe: int,
    topk: int,
    gt_lookup: Optional[dict[int, np.ndarray]] = None,
    neg_sample_rate: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    filter_non_contributing: bool = False,
    marginal_neg_sample_rate: float = 0.35,
    affected_zero_sample_rate: float = 1.0,
) -> tuple[list[FreshResult], list[LabelRow], dict[int, float]]:
    """
    Full label generation for one snapshot.

    Emits LabelRows for:
      - ALL affected probed partitions  (positive / risky examples)
      - A sampled subset of unaffected probed partitions  (negative / safe examples)

    Unaffected partitions have recall_drop = 0 by definition (stale copy IS the
    fresh copy), but they are critical for an unbiased model — at runtime the
    policy sees both risky and safe partitions, so training must cover both.

    neg_sample_rate: fraction of unaffected probed partitions to include per query.
                     1.0 = keep all, 0.0 = keep none (original biased behaviour).

    filter_non_contributing: when True (marginal_contributor templates only), skip
        the stale intervention for affected partitions where the query found no top-k
        results. These rows would produce guaranteed-zero labels that create noise —
        the model sees identical features with wildly different labels depending on
        whether the query happened to need the partition.
        A fraction (marginal_neg_sample_rate) are kept as explicit zero-label negatives
        so the model still sees the "marginal partition probed but not needed → safe" case.

    marginal_neg_sample_rate: fraction of filtered-out non-contributing affected rows
        to retain as explicit zero-label negatives. Set to 0.35 to keep the
        nonzero:zero ratio in marginal training data roughly calibrated to reality
        (~1:1.4 vs the true ~1:4), avoiding over-fetching marginal partitions at
        serve time without reintroducing label noise.

    affected_zero_sample_rate: fraction of affected rows to keep when the stale
        intervention ran but recall_drop == 0.0. Use < 1.0 for uniform templates
        where mutations spread thin across all partitions, producing huge numbers of
        zero-drop affected rows that overwhelm the positives and skew class balance.
        Does not apply to filter_non_contributing shortcut rows (those use
        marginal_neg_sample_rate instead).

    Returns (fresh_results, label_rows).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    affected_set = set(snapshot.affected_cids)

    # Build vec→partition map (used for hit rate and optional filter_non_contributing)
    vec_to_partition: dict[int, int] = {}
    for cid, (ids, _, _) in snapshot.all_partitions.items():
        for vid in ids:
            vec_to_partition[int(vid)] = cid

    fresh_results = run_fresh_query_batch(
        queries_with_idx, snapshot, centroid_index, n_probe, topk, gt_lookup
    )

    # Compute per-partition hit rate: fraction of queries where this partition
    # contributed at least one vector to the fresh top-k result.
    partition_hit_counts: dict[int, int] = {}
    for fr in fresh_results:
        seen: set[int] = set()
        for vid in fr.fresh_topk_ids:
            cid = vec_to_partition.get(int(vid))
            if cid is not None and cid not in seen:
                partition_hit_counts[cid] = partition_hit_counts.get(cid, 0) + 1
                seen.add(cid)
    n_queries = max(len(fresh_results), 1)
    partition_hit_rates = {cid: count / n_queries for cid, count in partition_hit_counts.items()}

    label_rows: list[LabelRow] = []
    for fr in fresh_results:
        gt_q = gt_lookup[fr.query_idx].tolist() if (gt_lookup and fr.query_idx in gt_lookup) else None

        # Partitions that contributed at least one top-k result for this query
        contributing_partitions: set[int] = set()
        if filter_non_contributing:
            for vid in fr.fresh_topk_ids:
                cid = vec_to_partition.get(int(vid))
                if cid is not None:
                    contributing_partitions.add(cid)

        for rank, cid in enumerate(fr.probe_ids, start=1):
            if cid in affected_set:
                if filter_non_contributing and cid not in contributing_partitions:
                    # Query found no top-k results here — intervention would produce
                    # a noisy zero label. Keep marginal_neg_sample_rate fraction as
                    # explicit negatives to preserve the "probed but not needed" signal.
                    if rng.random() > marginal_neg_sample_rate:
                        continue
                    stale_recall = fr.fresh_recall
                    recall_drop  = 0.0
                else:
                    # Run actual stale intervention — query depends on this partition
                    stale_recall = run_stale_intervention_for_partition(
                        fr, snapshot, cid, topk, gt_q
                    )
                    recall_drop = max(0.0, fr.fresh_recall - stale_recall)
                    # Uniform templates produce many affected rows where mutation spread
                    # thin → stale intervention runs but recall_drop stays 0. Sample
                    # these down so zero-signal rows don't swamp the positives.
                    if recall_drop == 0.0 and affected_zero_sample_rate < 1.0:
                        if rng.random() > affected_zero_sample_rate:
                            continue

            else:
                # Negative example: partition was not mutated → stale = fresh, drop = 0
                if neg_sample_rate < 1.0 and rng.random() > neg_sample_rate:
                    continue
                stale_recall = fr.fresh_recall
                recall_drop  = 0.0

            label_rows.append(LabelRow(
                query_idx=fr.query_idx,
                partition_id=cid,
                probe_rank=rank,
                fresh_recall=fr.fresh_recall,
                stale_recall=stale_recall,
                recall_drop=recall_drop,
            ))

    return fresh_results, label_rows, partition_hit_rates
