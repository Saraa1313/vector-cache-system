"""
Dry-run: end-to-end pipeline smoke test on a tiny query batch.

Uses 1 template (uniform_update) and 10 random queries so it completes in
under a minute.  Prints a sample output row and writes a small dataset to
data/training_data_dryrun/.

Run from project root:
    python src/offline_training/run_dry_run.py
"""

import os
import sys
import json
import time

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

# ── Reuse existing system components ─────────────────────────────────────────
from config import DATA_DIR, CENTROIDS_PATH, N_PROBE, TOPK
from storage.object_store import ObjectStore
from admin.data_loader import read_fvecs
from admin.compute_ground_truth import brute_force_topk

# ── New offline-only modules ──────────────────────────────────────────────────
from common.partition_utils import load_all_partitions
from offline_training.snapshot_builder import FIRST_VERSION_TEMPLATES, build_mutated_snapshot
from offline_training.query_sampler import sample_queries
from offline_training.label_generator import compute_all_labels
from offline_training.feature_extractor import extract_features_for_qp
from offline_training.dataset_writer import write_training_dataset

import faiss

N_PROBE_VAL  = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE
N_QUERIES    = 10
OUTPUT_DIR   = os.path.join(os.path.dirname(_SRC), "data", "training_data_dryrun")
DATASET_NAME = "dry_run_sample"


def build_centroid_index(centroids: np.ndarray) -> faiss.IndexFlatL2:
    d = centroids.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(np.ascontiguousarray(centroids.astype(np.float32)))
    return idx


def main() -> None:
    t0 = time.time()
    print("=== DRY RUN: recall-drop training data pipeline ===\n")

    print("Loading SIFT queries ...")
    all_queries = read_fvecs(
        os.path.join(DATA_DIR, "sift", "sift_query.fvecs")
    ).astype(np.float32)

    print("Loading centroids ...")
    centroids      = np.load(CENTROIDS_PATH).astype(np.float32)
    centroid_index = build_centroid_index(centroids)

    print("Loading partitions from MinIO ...")
    store           = ObjectStore()
    base_partitions = load_all_partitions(store)
    print(f"  {len(base_partitions)} partitions loaded")

    # Use the first template only (uniform_update — no near/far, no reference queries needed)
    template = FIRST_VERSION_TEMPLATES[0]
    print(f"\nTemplate: {template.name}")
    snapshot = build_mutated_snapshot(base_partitions, centroids, template)
    print(f"  Affected partitions: {snapshot.affected_cids}")
    for cid in snapshot.affected_cids:
        log  = snapshot.mutation_log[cid]
        size = len(snapshot.stale_partitions[cid][0])
        print(f"    cid={cid}  size={size}  upd={log.updates}  del={log.deletes}")

    # Build post-mutation index arrays for GT
    cids_sorted    = sorted(snapshot.all_partitions)
    all_ids_fresh  = np.concatenate([snapshot.all_partitions[c][0] for c in cids_sorted])
    all_vecs_fresh = np.concatenate([snapshot.all_partitions[c][1] for c in cids_sorted], axis=0)

    # Random queries
    queries = sample_queries(
        all_queries, "random", N_QUERIES, centroid_index, N_PROBE_VAL,
        partitions=base_partitions,
    )
    print(f"\nSampled {len(queries)} random queries")

    # GT per query
    gt_lookup: dict[int, np.ndarray] = {}
    for q_idx, q_vec in queries:
        gt_lookup[q_idx] = brute_force_topk(
            q_vec.reshape(1, -1), all_ids_fresh, all_vecs_fresh, TOPK
        )[0]

    # Label generation
    fresh_results, label_rows = compute_all_labels(
        queries, snapshot, centroid_index, N_PROBE_VAL, TOPK, gt_lookup
    )
    print(f"  {len(label_rows)} label rows from {len(fresh_results)} queries")

    if not label_rows:
        print("\nWARNING: no label rows — none of the sampled queries probed an affected partition.")
        print("Try increasing N_QUERIES or selecting a template with more affected partitions.")
        return

    # Feature extraction
    fr_by_qidx = {fr.query_idx: fr for fr in fresh_results}
    rows = []
    for lr in label_rows:
        fr  = fr_by_qidx[lr.query_idx]
        row = extract_features_for_qp(
            query_idx          = lr.query_idx,
            partition_id       = lr.partition_id,
            probe_rank         = lr.probe_rank,
            n_probe            = N_PROBE_VAL,
            centroid_distances = fr.centroid_distances,
            snapshot           = snapshot,
            fresh_recall       = lr.fresh_recall,
            stale_recall       = lr.stale_recall,
            recall_drop        = lr.recall_drop,
        )
        row["query_type"] = "random"
        rows.append(row)

    # Write
    paths = write_training_dataset(rows, OUTPUT_DIR, DATASET_NAME, {
        "template":   template.name,
        "n_queries":  N_QUERIES,
        "n_probe":    N_PROBE_VAL,
        "topk":       TOPK,
        "elapsed_s":  round(time.time() - t0, 1),
    })

    print("\nOutput files:")
    for k, v in paths.items():
        print(f"  {k:10s}: {v}")

    print("\nSample row (first label row):")
    print(json.dumps(
        {k: v for k, v in rows[0].items()},
        indent=2, default=str,
    ))
    print(f"\nDry run complete in {round(time.time()-t0, 1)}s")


if __name__ == "__main__":
    main()
