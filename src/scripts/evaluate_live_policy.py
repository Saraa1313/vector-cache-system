#!/usr/bin/env python3
"""
evaluate_live_policy.py

Compares always_cache, always_fetch, and learned fetch policies across
multiple mutation scenarios using the real query node via gRPC.

Scenarios
---------
  conc_update_30pct       concentrated top-12, 30% updates, high drift (in-dist, heavy)
  conc_mixed_20pct        concentrated top-12, 20% updates + 20% deletes (in-dist)
  rank_mid_update_30pct   mid-rank (9-16) top-8 partitions, 30% updates (OOD)
  conc_update_10pct       concentrated top-12, 10% updates (light staleness)

Each scenario: mutate MinIO, notify query node, evaluate all 3 policies on same
queries, restore MinIO, repeat for next scenario.

Prerequisites:
    python src/query/start_query_node.py \
        --model-path models/sift1m_recall_drop_v15_xgb_clf.ubj

Usage:
    python src/scripts/evaluate_live_policy.py [--n-eval 200] [--n-warmup 300]
"""

import os
import sys
import time
import json
import argparse
from dataclasses import dataclass

import numpy as np
import grpc

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.join(_SRC, "proto"))

import faiss
import proto.vector_search_pb2 as pb2
import proto.vector_search_pb2_grpc as pb2_grpc

from config import DATA_DIR, CENTROIDS_PATH, N_PROBE, TOPK, QUERY_NODE_HOST, GRPC_PORT, CACHE_SIZE
from storage.object_store import ObjectStore
from admin.data_loader import read_fvecs
from common.partition_utils import (
    load_all_partitions, compute_rank_targeted_scores,
)

_DEFAULT_N_PROBE = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE
# Order matters: always_fetch resets _freshness via _init_freshness on every fetch,
# zeroing version_lag for all probed partitions. Run it last so learned sees real staleness.
POLICIES = ["always_cache", "learned", "always_fetch"]


# ── Scenario definition ───────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    label: str            # human-readable
    pattern: str          # "concentrated" | "rank_targeted"
    mutation_type: str    # "update" | "mixed"
    mutation_fraction: float
    n_targets: int
    drift_scale: float
    rank_range: tuple | None = None   # for rank_targeted only


SCENARIOS = [
    # ── Concentrated (top-12 by size) ─────────────────────────────────────────
    Scenario(
        name="conc_update_10pct",
        label="concentrated top-12, 10% updates  [light staleness]",
        pattern="concentrated",
        mutation_type="update",
        mutation_fraction=0.10,
        n_targets=12,
        drift_scale=200.0,
    ),
    Scenario(
        name="conc_update_50pct",
        label="concentrated top-12, 50% updates  [heavy staleness]",
        pattern="concentrated",
        mutation_type="update",
        mutation_fraction=0.50,
        n_targets=12,
        drift_scale=200.0,
    ),
    Scenario(
        name="conc_mixed_40pct",
        label="concentrated top-12, 40% updates + 40% deletes  [aggressive mixed]",
        pattern="concentrated",
        mutation_type="mixed",
        mutation_fraction=0.40,
        n_targets=12,
        drift_scale=200.0,
    ),
    # ── Rank-targeted (tests model across probe-rank spectrum) ─────────────────
    Scenario(
        name="rank_close_update_50pct",
        label="rank-targeted close (1-4), top-8 partitions, 50% updates  [low rank OOD]",
        pattern="rank_targeted",
        mutation_type="update",
        mutation_fraction=0.50,
        n_targets=8,
        drift_scale=200.0,
        rank_range=(1, 4),
    ),
    Scenario(
        name="rank_mid_update_50pct",
        label="rank-targeted mid (9-16), top-8 partitions, 50% updates  [mid rank OOD]",
        pattern="rank_targeted",
        mutation_type="update",
        mutation_fraction=0.50,
        n_targets=8,
        drift_scale=200.0,
        rank_range=(9, 16),
    ),
    Scenario(
        name="rank_late_update_50pct",
        label="rank-targeted late (25-32), top-8 partitions, 50% updates  [high rank OOD]",
        pattern="rank_targeted",
        mutation_type="update",
        mutation_fraction=0.50,
        n_targets=8,
        drift_scale=200.0,
        rank_range=(25, 32),
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _recon_error(centroid_vec: np.ndarray, vecs: np.ndarray) -> float:
    if len(vecs) == 0:
        return 0.0
    diffs = vecs.astype(np.float32) - centroid_vec.astype(np.float32)
    return float(np.mean(np.sqrt(np.sum(diffs ** 2, axis=1))))


def _build_centroid_index(centroids: np.ndarray):
    idx = faiss.IndexFlatL2(centroids.shape[1])
    idx.add(np.ascontiguousarray(centroids.astype(np.float32)))
    return idx


def _recall_at_k(result_ids: list[int], gt_ids: np.ndarray, k: int) -> float:
    return len(set(result_ids[:k]) & set(gt_ids[:k])) / k


def _select_targets(
    scenario: Scenario,
    base_partitions: dict,
    centroid_index,
    n_probe: int,
    all_queries: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    non_empty = [c for c, (ids, _, _) in base_partitions.items() if len(ids) > 0]

    if scenario.pattern == "concentrated":
        return sorted(
            non_empty,
            key=lambda c: len(base_partitions[c][0]),
            reverse=True,
        )[:scenario.n_targets]

    if scenario.pattern == "rank_targeted":
        pool_idx = rng.choice(len(all_queries), size=200, replace=False)
        query_batch = [(int(i), all_queries[i]) for i in pool_idx]
        scores = compute_rank_targeted_scores(
            query_batch, centroid_index, n_probe, scenario.rank_range,
        )
        return sorted(
            non_empty,
            key=lambda c: scores.get(c, 0),
            reverse=True,
        )[:scenario.n_targets]

    raise ValueError(f"Unknown pattern: {scenario.pattern!r}")


def _build_mutations(
    scenario: Scenario,
    target_cids: list[int],
    base_partitions: dict,
    rng: np.random.Generator,
) -> tuple[dict, dict, dict]:
    """
    Returns (originals, mutated, mutation_log).
    mutation_log[cid] = {updates, deletes, new_size, new_version}
    """
    originals: dict[int, tuple] = {}
    mutated:   dict[int, tuple] = {}
    logs:      dict[int, dict]  = {}

    for cid in target_cids:
        ids, vecs, version = base_partitions[cid]
        originals[cid] = (ids, vecs, version)
        new_ids, new_vecs = ids.copy(), vecs.copy()
        n_update = n_delete = 0

        if scenario.mutation_type in ("update", "mixed"):
            n_update = max(1, int(len(ids) * scenario.mutation_fraction))
            chosen = rng.choice(len(ids), size=n_update, replace=False)
            noise  = rng.standard_normal((n_update, vecs.shape[1])).astype(np.float32)
            new_vecs[chosen] += noise * scenario.drift_scale

        if scenario.mutation_type == "mixed":
            remaining = [i for i in range(len(ids)) if i not in set(chosen)]
            n_delete  = max(1, int(len(ids) * scenario.mutation_fraction))
            n_delete  = min(n_delete, len(remaining))
            del_idx   = rng.choice(remaining, size=n_delete, replace=False)
            keep_mask = np.ones(len(ids), dtype=bool)
            keep_mask[del_idx] = False
            new_ids  = new_ids[keep_mask]
            new_vecs = new_vecs[keep_mask]

        mutated[cid] = (new_ids, new_vecs, version + 1)
        logs[cid] = {
            "updates":     n_update,
            "deletes":     n_delete,
            "new_size":    len(new_ids),
            "new_version": version + 1,
        }

    return originals, mutated, logs


def _write_to_minio(store: ObjectStore, mutated: dict) -> None:
    for cid, (ids, vecs, version) in mutated.items():
        store.save_centroid(cid, ids, vecs, version)


def _restore_minio(store: ObjectStore, originals: dict) -> None:
    for cid, (ids, vecs, version) in originals.items():
        store.save_centroid(cid, ids, vecs, version)


def _notify_query_node(
    stub,
    centroids: np.ndarray,
    mutated: dict,
    mutation_log: dict,
) -> None:
    deltas = []
    for cid, (ids, vecs, new_version) in mutated.items():
        log = mutation_log[cid]
        re        = _recon_error(centroids[cid], vecs)
        orig_size = log["new_size"] + log["deletes"]  # pre-mutation size
        deltas.append(pb2.PartitionMetadataDelta(
            partition_id             = cid,
            version_id               = new_version,
            number_of_inserts        = 0,
            number_of_updates        = log["updates"],
            number_of_deletes        = log["deletes"],
            partition_size           = log["new_size"],
            fraction_vectors_touched = (log["updates"] + log["deletes"])
                                       / max(orig_size, 1),
            membership_change_count  = log["deletes"],
            new_centroid             = centroids[cid].tolist(),
            reconstruction_error     = re,
        ))
    stub.NotifyBatchApplied(pb2.BatchAppliedNotification(
        last_seq_id      = 999_999,
        partition_deltas = deltas,
    ))


def _select_eval_queries(
    all_queries: np.ndarray,
    I_all: np.ndarray,
    target_set: set[int],
    hot_cids: set[int],
    n_eval: int,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray]]:
    """
    Pick eval queries that (a) probe at least one target partition and (b) have the
    highest warm_frac = fraction of probes already in hot_cids.

    Eval queries are spatially clustered around the target partitions, so their union
    of probed partitions is small (typically well under CACHE_SIZE). Warming the cache
    with these same queries loads exactly that focused partition neighbourhood, giving
    near-zero cold misses during evaluation and making the staleness effect observable
    on most probed partitions rather than just the target ones.
    """
    candidates = []
    for qi in rng.permutation(len(all_queries)):
        probes = [int(c) for c in I_all[qi] if c >= 0]
        if any(c in target_set for c in probes):
            warm_frac = sum(1 for c in probes if c in hot_cids) / len(probes)
            candidates.append((warm_frac, int(qi)))
    candidates.sort(reverse=True)  # highest warm_frac first
    result = [(qi, all_queries[qi]) for _, qi in candidates[:n_eval]]
    if result:
        fracs = [f for f, _ in candidates[:n_eval]]
        print(f"    warm_frac: mean={np.mean(fracs):.2f}  "
              f"min={np.min(fracs):.2f}  max={np.max(fracs):.2f}")
    return result


def _run_policy(
    stub,
    policy: str,
    eval_queries: list[tuple[int, np.ndarray]],
    gt: dict[int, np.ndarray],
    n_probe: int,
    topk: int,
) -> dict:
    recalls, total_ms_list, fetch_ms_list = [], [], []
    fetch_counts, bytes_list, infer_list  = [], [], []
    cold_counts, policy_counts            = [], []

    for q_idx, q_vec in eval_queries:
        t0   = time.perf_counter()
        resp = stub.Search(pb2.SearchRequest(
            vector       = q_vec.tolist(),
            top_k        = topk,
            n_probe      = n_probe,
            fetch_policy = policy,
        ))
        total_ms = (time.perf_counter() - t0) * 1000

        result_ids = [n.id for n in resp.results]
        recall     = _recall_at_k(result_ids, gt[q_idx], topk)

        recalls.append(recall)
        total_ms_list.append(total_ms)
        fetch_ms_list.append(resp.fetch_ms)
        fetch_counts.append(resp.fetch_count)
        bytes_list.append(resp.bytes_fetched)
        infer_list.append(resp.inference_ms)
        cold_counts.append(resp.cold_fetch_count)
        policy_counts.append(resp.policy_fetch_count)

    n_queries = len(eval_queries)
    return {
        "mean_recall":        float(np.mean(recalls)),
        "p50_total_ms":       float(np.percentile(total_ms_list, 50)),
        "p95_total_ms":       float(np.percentile(total_ms_list, 95)),
        "p50_fetch_ms":       float(np.percentile(fetch_ms_list, 50)),
        "p95_fetch_ms":       float(np.percentile(fetch_ms_list, 95)),
        "mean_fetch_rate":    float(np.mean(fetch_counts)) / n_probe,
        "mean_bytes_mb":      float(np.mean(bytes_list)) / (1024 ** 2),
        "mean_infer_ms":      float(np.mean(infer_list)),
        "total_cold_fetches": int(np.sum(cold_counts)),
        "total_policy_fetches": int(np.sum(policy_counts)),
        "n_queries":          n_queries,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval",    type=int, default=200)
    parser.add_argument("--n-probe",   type=int, default=_DEFAULT_N_PROBE)
    parser.add_argument("--topk",      type=int, default=TOPK)
    parser.add_argument("--host",      default=QUERY_NODE_HOST)
    parser.add_argument("--port",      type=int, default=GRPC_PORT)
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Run only these scenario names (default: all)")
    args = parser.parse_args()

    rng = np.random.default_rng(777)

    active_scenarios = SCENARIOS
    if args.scenarios:
        name_set = set(args.scenarios)
        active_scenarios = [s for s in SCENARIOS if s.name in name_set]

    # ── Load shared resources ─────────────────────────────────────────────────
    print("Loading SIFT queries ...")
    all_queries = read_fvecs(
        os.path.join(DATA_DIR, "sift", "sift_query.fvecs")
    ).astype(np.float32)

    print("Loading centroids ...")
    centroids      = np.load(CENTROIDS_PATH).astype(np.float32)
    centroid_index = _build_centroid_index(centroids)

    print("Loading all partitions from MinIO ...")
    store           = ObjectStore()
    base_partitions = load_all_partitions(store)
    print(f"  {len(base_partitions)} partitions loaded")

    # Pre-compute probe sets for every SIFT query (single batched FAISS search).
    # I_all[qi] = top-n_probe centroid IDs for query qi.
    print(f"Pre-computing probe sets for all {len(all_queries)} SIFT queries ...")
    _, I_all = centroid_index.search(np.ascontiguousarray(all_queries), args.n_probe)

    # hot_cids: top-CACHE_SIZE partitions by total probe frequency across all queries.
    # Eval queries are selected to have high warm_frac (fraction of their probes in
    # hot_cids), so the per-policy warmup with those queries naturally loads the right
    # partitions into the LRU and cold misses are minimised during evaluation.
    probe_freq: dict[int, int] = {}
    for row in I_all:
        for c in row:
            if c >= 0:
                probe_freq[int(c)] = probe_freq.get(int(c), 0) + 1
    hot_cids = set(sorted(probe_freq, key=probe_freq.get, reverse=True)[:CACHE_SIZE])
    print(f"  Hot partition set: {len(hot_cids)} partitions  (cache_size={CACHE_SIZE})")

    # ── Connect to query node ─────────────────────────────────────────────────
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub    = pb2_grpc.VectorSearchStub(channel)
    print(f"Connected to query node at {args.host}:{args.port}")

    # ── Per-scenario evaluation loop ──────────────────────────────────────────
    all_results: list[tuple[Scenario, dict[str, dict]]] = []

    for scenario in active_scenarios:
        print(f"\n{'='*72}")
        print(f"SCENARIO: {scenario.name}")
        print(f"  {scenario.label}")

        # Select target partitions
        target_cids = _select_targets(
            scenario, base_partitions, centroid_index,
            args.n_probe, all_queries, rng,
        )
        target_set = set(target_cids)
        print(f"  target partitions ({len(target_cids)}): {target_cids}")

        # Build mutations in-memory
        originals, mutated, mutation_log = _build_mutations(
            scenario, target_cids, base_partitions, rng,
        )
        for cid in target_cids:
            log = mutation_log[cid]
            print(f"    cid={cid:4d}  size={len(base_partitions[cid][0]):6d}  "
                  f"updates={log['updates']}  deletes={log['deletes']}")

        eval_queries = _select_eval_queries(
            all_queries, I_all, target_set, hot_cids, args.n_eval, rng,
        )
        if len(eval_queries) < args.n_eval:
            print(f"  WARNING: only {len(eval_queries)} queries probe target partitions")
        print(f"  {len(eval_queries)} evaluation queries selected (sorted by warm_frac)")

        # Ground truth: exact brute-force via FAISS (batched — much faster than per-query numpy)
        print("  Computing ground truth (FAISS exact) ...")
        post_mutation = {**base_partitions, **mutated}
        cids_sorted   = sorted(post_mutation.keys())
        all_ids_pm    = np.concatenate([post_mutation[c][0] for c in cids_sorted])
        all_vecs_pm   = np.concatenate([post_mutation[c][1] for c in cids_sorted], axis=0).astype(np.float32)
        _gt_idx = faiss.IndexFlatL2(all_vecs_pm.shape[1])
        _gt_idx.add(np.ascontiguousarray(all_vecs_pm))
        q_mat = np.ascontiguousarray(
            np.stack([q for _, q in eval_queries]).astype(np.float32)
        )
        _, _I = _gt_idx.search(q_mat, args.topk)
        gt: dict[int, np.ndarray] = {
            q_idx: all_ids_pm[_I[i]] for i, (q_idx, _) in enumerate(eval_queries)
        }
        print(f"  Ground truth computed ({len(gt)} queries).")

        # Evaluate each policy with its own independent cache state so
        # one policy's fetches don't warm the cache for the next policy.
        scenario_results: dict[str, dict] = {}
        for policy in POLICIES:
            print(f"  [{policy}] resetting cache state ...")
            # 1. Clear cache
            stub.ClearCache(pb2.ClearCacheRequest())
            # 2. Write originals to MinIO so warmup loads stale versions
            _restore_minio(store, originals)
            # 3. Warm cache using the eval queries (real SIFT queries clustered around
            #    the target partitions). Their union of probed partitions is small and
            #    typically fits entirely within CACHE_SIZE, so cold misses during eval
            #    are near-zero. Target partitions are loaded explicitly last so they
            #    sit at the LRU head and are guaranteed to remain in cache.
            for _, q_vec in eval_queries:
                stub.Search(pb2.SearchRequest(
                    vector=q_vec.tolist(),
                    top_k=args.topk,
                    n_probe=args.n_probe,
                    fetch_policy="always_fetch",
                ))
            for cid in target_cids:
                stub.Search(pb2.SearchRequest(
                    vector=centroids[cid].tolist(),
                    top_k=args.topk,
                    n_probe=args.n_probe,
                    fetch_policy="always_fetch",
                ))
            # 4. Write mutations to MinIO — cache now holds stale, MinIO has fresh
            _write_to_minio(store, mutated)
            # 5. Inject freshness so version_lag=1 for all target partitions
            _notify_query_node(stub, centroids, mutated, mutation_log)

            print(f"  [{policy}] evaluating {len(eval_queries)} queries ...")
            scenario_results[policy] = _run_policy(
                stub, policy, eval_queries, gt, args.n_probe, args.topk,
            )
            r = scenario_results[policy]
            print(f"    recall={r['mean_recall']:.4f}  "
                  f"p50={r['p50_total_ms']:.1f}ms  "
                  f"fetch_rate={r['mean_fetch_rate']:.1%}  "
                  f"MB/q={r['mean_bytes_mb']:.3f}")

        all_results.append((scenario, scenario_results))

        # Final restore after all policies done for this scenario
        print("  Restoring MinIO ...")
        _restore_minio(store, originals)
        stub.ClearCache(pb2.ClearCacheRequest())

    # ── Final comparison table ────────────────────────────────────────────────
    topk_label = f"recall@{args.topk}"

    print(f"\n{'='*110}")
    print(f"FINAL RESULTS  |  n_probe={args.n_probe}  topk={args.topk}  n_eval={args.n_eval}")
    print(f"{'='*110}")

    for scenario, scenario_results in all_results:
        print(f"\n  {scenario.name}  —  {scenario.label}")
        n_q = next(iter(scenario_results.values()))["n_queries"]
        print(f"  {'Policy':<16} {topk_label:<14} {'p50 ms':<10} {'p95 ms':<10} "
              f"{'fetch%':<8} {'cold(total)':<14} {'policy(total)':<16} {'MB/q':<10} {'infer ms'}")
        print(f"  {'-'*110}")
        for policy in POLICIES:
            r = scenario_results[policy]
            print(
                f"  {policy:<16} "
                f"{r['mean_recall']:<14.4f} "
                f"{r['p50_total_ms']:<10.1f} "
                f"{r['p95_total_ms']:<10.1f} "
                f"{r['mean_fetch_rate']:<8.1%} "
                f"{r['total_cold_fetches']:<5d}/{n_q:<8d} "
                f"{r['total_policy_fetches']:<5d}/{n_q:<10d} "
                f"{r['mean_bytes_mb']:<10.3f} "
                f"{r['mean_infer_ms']:.2f}"
            )

    print(f"\n{'='*110}")
    print("Done.")

    # ── Save results ──────────────────────────────────────────────────────────
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir   = os.path.join(_project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path  = os.path.join(results_dir, f"eval_{timestamp}.json")

    output = {
        "timestamp": timestamp,
        "config": {
            "n_probe":  args.n_probe,
            "topk":     args.topk,
            "n_eval":   args.n_eval,
            "policies": POLICIES,
        },
        "scenarios": {
            scenario.name: {
                "label":             scenario.label,
                "pattern":           scenario.pattern,
                "mutation_type":     scenario.mutation_type,
                "mutation_fraction": scenario.mutation_fraction,
                "n_targets":         scenario.n_targets,
                "policies":          scenario_results,
            }
            for scenario, scenario_results in all_results
        },
    }

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
