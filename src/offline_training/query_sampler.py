"""
Generates query batches for offline training data generation.

Three query types:
  random   — uniformly sampled from the full SIFT query set
  hot      — queries whose top-nprobe partitions include the most-populated ones
  boundary — queries sitting near the Voronoi boundary of target partitions
             (one of their two nearest centroids must be a target affected partition)

Returns list of (original_query_idx, query_vector) tuples so callers can
cross-reference with the SIFT ground truth by absolute index.
"""

from __future__ import annotations

import numpy as np
import faiss


def sample_queries(
    all_queries: np.ndarray,
    query_type: str,
    n: int,
    centroid_index: faiss.IndexFlatL2,
    n_probe: int = 20,
    partitions: dict[int, tuple] | None = None,
    rng: np.random.Generator | None = None,
    target_cids: list[int] | None = None,
) -> list[tuple[int, np.ndarray]]:
    """
    Returns up to n (original_query_idx, query_vector) pairs.

    all_queries : (N, D) float32 — full SIFT query file
    query_type  : "random" | "hot" | "boundary"
    partitions  : required for "hot"; ignored otherwise
    target_cids : for "boundary" — restrict to queries whose top-2 centroids
                  include at least one affected partition; falls back to global
                  boundary if no such queries found in the pool
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = min(n, len(all_queries))

    if query_type == "random":
        return _sample_random(all_queries, n, rng)
    elif query_type == "hot":
        return _sample_hot(all_queries, n, centroid_index, n_probe, partitions, rng)
    elif query_type == "boundary":
        return _sample_boundary(all_queries, n, centroid_index, rng, target_cids)
    else:
        raise ValueError(f"Unknown query_type {query_type!r}. Choose: random | hot | boundary")


# ── Implementations ───────────────────────────────────────────────────────────

def _sample_random(
    all_queries: np.ndarray, n: int, rng: np.random.Generator
) -> list[tuple[int, np.ndarray]]:
    indices = rng.choice(len(all_queries), size=n, replace=False)
    return [(int(i), all_queries[i]) for i in indices]


def _sample_hot(
    all_queries: np.ndarray,
    n: int,
    centroid_index: faiss.IndexFlatL2,
    n_probe: int,
    partitions: dict[int, tuple] | None,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    """
    Score each candidate query by the total size of its top-nprobe partitions.
    Queries that naturally probe large (hot) partitions score highest.
    """
    if partitions is None:
        return _sample_random(all_queries, n, rng)

    partition_size = {cid: len(data[0]) for cid, data in partitions.items()}

    pool_size = min(len(all_queries), max(n * 10, 2000))
    pool_idx = rng.choice(len(all_queries), size=pool_size, replace=False)
    pool_q = np.ascontiguousarray(all_queries[pool_idx].astype(np.float32))

    _, I = centroid_index.search(pool_q, n_probe)

    scores = np.array(
        [sum(partition_size.get(int(c), 0) for c in row if c >= 0) for row in I],
        dtype=float,
    )

    top_local = np.argsort(scores)[::-1][: n]
    return [(int(pool_idx[i]), all_queries[pool_idx[i]]) for i in top_local]


def _sample_boundary(
    all_queries: np.ndarray,
    n: int,
    centroid_index: faiss.IndexFlatL2,
    rng: np.random.Generator,
    target_cids: list[int] | None = None,
) -> list[tuple[int, np.ndarray]]:
    """
    Score by d2/d1 where d1, d2 are distances to the two nearest centroids.
    Ratio near 1.0 → query sits on the Voronoi boundary.

    If target_cids is provided, restrict candidates to queries whose nearest
    OR second-nearest centroid is one of the affected partitions — this ensures
    boundary queries actually probe the mutated partitions.  Falls back to
    global boundary sampling if too few target-adjacent queries are found.
    """
    pool_size = min(len(all_queries), max(n * 20, 4000))
    pool_idx = rng.choice(len(all_queries), size=pool_size, replace=False)
    pool_q = np.ascontiguousarray(all_queries[pool_idx].astype(np.float32))

    D, I = centroid_index.search(pool_q, 2)

    eps = 1e-9
    d1 = D[:, 0] + eps
    d2 = D[:, 1] + eps
    margin = d2 / d1   # 1.0 = perfect boundary; higher = clear assignment

    if target_cids is not None:
        target_set = set(target_cids)
        # Keep only queries where at least one of the top-2 centroids is a target
        adjacent = np.array([
            int(I[i, 0]) in target_set or int(I[i, 1]) in target_set
            for i in range(len(pool_q))
        ])
        adj_idx = np.where(adjacent)[0]
        if len(adj_idx) >= n:
            # Sort adjacent candidates by boundary margin, pick tightest n
            sorted_adj = adj_idx[np.argsort(margin[adj_idx])][:n]
            return [(int(pool_idx[i]), all_queries[pool_idx[i]]) for i in sorted_adj]
        # Not enough adjacent — fall through to global boundary sampling

    top_local = np.argsort(margin)[:n]
    return [(int(pool_idx[i]), all_queries[pool_idx[i]]) for i in top_local]
