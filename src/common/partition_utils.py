"""
New helpers that don't exist in the online serving path:
  - load_all_partitions           : loads partitions as {cid: (ids, vecs, version)} dict
  - probe_centroids               : standalone centroid lookup returning (ids, l2_dists)
  - ivf_search                    : search over an arbitrary in-memory partition dict
  - compute_partition_contributions: counts how many top-k results each partition
                                    contributed across a query batch (used for
                                    contribution-based partition targeting in offline training)

Everything else (ObjectStore.load_centroid, read_fvecs, brute_force_topk,
assign_to_centroids, compute_recall) is imported directly from existing modules.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import faiss


def load_all_partitions(store) -> dict[int, tuple]:
    """
    Load every partition from object store into memory.
    Returns {centroid_id: (ids: np.ndarray, vecs: np.ndarray, version: int)}.

    Uses the existing ObjectStore.list_centroid_ids / load_centroid interface.
    """
    cids = store.list_centroid_ids()
    results: dict[int, tuple] = {}
    with ThreadPoolExecutor(max_workers=32) as ex:
        futures = {ex.submit(store.load_centroid, cid): cid for cid in cids}
        for fut in as_completed(futures):
            cid = futures[fut]
            ids, vecs, version = fut.result()
            results[cid] = (ids, vecs, version)
    return results


def probe_centroids(
    centroid_index: faiss.IndexFlatL2, query: np.ndarray, n_probe: int
) -> tuple[list[int], list[float]]:
    """
    Return (probe_ids, l2_distances) for the n_probe nearest centroids,
    ordered by ascending distance.
    """
    q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))
    D, I = centroid_index.search(q, n_probe)
    probe_ids = [int(x) for x in I[0] if x >= 0]
    dists     = [float(d) for d in D[0][: len(probe_ids)]]
    return probe_ids, dists


def compute_partition_contributions(
    query_batch: list[tuple[int, np.ndarray]],
    base_partitions: dict[int, tuple],
    centroid_index: faiss.IndexFlatL2,
    n_probe: int,
    topk: int,
) -> dict[int, int]:
    """
    For each query in the batch, run IVF search on the clean (pre-mutation) index
    and record which partition each top-k result vector came from.

    Returns {partition_id: contribution_count} — how many top-k result vectors
    each partition contributed across the full query batch.

    Used by contribution-based partition selection (top_k_contributor /
    marginal_contributor patterns) so that targeted mutations land on partitions
    that actually contain true neighbors for the labeling query batch.
    """
    # Reverse map: vector_id → partition_id (built once from clean partitions)
    vec_to_partition: dict[int, int] = {}
    for cid, (ids, _, _) in base_partitions.items():
        for vid in ids:
            vec_to_partition[int(vid)] = cid

    counts: Counter = Counter()
    for _, q_vec in query_batch:
        probe_ids, _ = probe_centroids(centroid_index, q_vec, n_probe)
        topk_ids = ivf_search(q_vec, base_partitions, probe_ids, topk)
        for vid in topk_ids:
            cid = vec_to_partition.get(int(vid))
            if cid is not None:
                counts[cid] += 1

    return dict(counts)


def compute_rank_targeted_scores(
    query_batch: list[tuple[int, np.ndarray]],
    centroid_index: faiss.IndexFlatL2,
    n_probe: int,
    rank_range: tuple[int, int],
) -> dict[int, int]:
    """
    For each query, find which partitions appear at the target probe rank positions
    and count how often each partition lands in that rank window across the batch.

    rank_range: (start, end) 1-based inclusive, e.g. (1, 4) for close ranks.
    Returns {partition_id: count} — same structure as compute_partition_contributions,
    compatible with contribution_scores param in build_mutated_snapshot.
    """
    rank_start, rank_end = rank_range
    counts: Counter = Counter()
    for _, q_vec in query_batch:
        probe_ids, _ = probe_centroids(centroid_index, q_vec, n_probe)
        for rank_idx in range(rank_start - 1, min(rank_end, len(probe_ids))):
            counts[probe_ids[rank_idx]] += 1
    return dict(counts)


def ivf_search(
    query: np.ndarray,
    partitions: dict[int, tuple],
    probe_ids: list[int],
    k: int,
) -> list[int]:
    """
    Scan the probe_ids partitions from an in-memory dict and return the
    top-k vector ids by L2 distance to query.

    partitions: {cid: (ids: np.ndarray, vecs: np.ndarray, version: int)}
    """
    parts = [partitions[cid] for cid in probe_ids if cid in partitions]
    if not parts:
        return []
    all_ids  = np.concatenate([p[0] for p in parts])
    all_vecs = np.concatenate([p[1] for p in parts], axis=0)
    if len(all_ids) == 0:
        return []
    dists   = ((all_vecs.astype(np.float32) - query) ** 2).sum(axis=1)
    top_idx = np.argsort(dists)[:k]
    return [int(all_ids[i]) for i in top_idx]
