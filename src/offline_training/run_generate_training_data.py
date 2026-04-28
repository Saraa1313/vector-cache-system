"""
Offline training-data generation pipeline.

For each template:
  1. Sample the query batch (hot queries always first for contribution scoring)
  2. For contribution-based patterns (top_k_contributor / marginal_contributor):
       compute partition contribution scores from the hot query batch against
       the clean index, then build the mutated snapshot using those scores.
     For other patterns (uniform / concentrated):
       build the snapshot directly (no query-batch dependency).
  3. Sample remaining query types (boundary uses snapshot.affected_cids)
  4. Compute post-mutation brute-force GT
  5. Run stale interventions + label (label_generator)
  6. Extract feature rows (feature_extractor)
  7. Write combined dataset (dataset_writer)

Requires MinIO running with a built index.
Does NOT modify MinIO, DynamoDB, or any online-serving state.

Run from project root:
    python src/offline_training/run_generate_training_data.py [--help]
"""

import os
import sys
import argparse
import time

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np
import faiss

from config import DATA_DIR, CENTROIDS_PATH, N_PROBE, TOPK
from storage.object_store import ObjectStore
from admin.data_loader import read_fvecs
from admin.compute_ground_truth import brute_force_topk

from common.partition_utils import load_all_partitions, compute_partition_contributions, compute_rank_targeted_scores
from offline_training.snapshot_builder import (
    TEMPLATES, build_mutated_snapshot,
    ChainedWorkloadTemplate, build_chained_snapshot,
)
from offline_training.query_sampler import sample_queries
from offline_training.label_generator import compute_all_labels
from offline_training.feature_extractor import extract_features_for_qp
from offline_training.dataset_writer import write_training_dataset

DEFAULT_N_PROBE      = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE
DEFAULT_N_QUERIES    = 200
DEFAULT_OUTPUT_DIR   = os.path.join(os.path.dirname(_SRC), "data", "training_data")
DEFAULT_DATASET_NAME = "sift1m_recall_drop_v1"

_CONTRIBUTION_PATTERNS = {"top_k_contributor", "marginal_contributor"}
_RANK_TARGETED_PATTERNS = {"rank_targeted"}


def _build_centroid_index(centroids: np.ndarray) -> faiss.IndexFlatL2:
    d = centroids.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(np.ascontiguousarray(centroids.astype(np.float32)))
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline recall-drop training data")
    parser.add_argument("--n-queries",       type=int, default=DEFAULT_N_QUERIES,
                        help="Queries per (template, query_type) combination")
    parser.add_argument("--n-probe",         type=int, default=DEFAULT_N_PROBE)
    parser.add_argument("--topk",            type=int, default=TOPK)
    parser.add_argument("--output-dir",      default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name",    default=DEFAULT_DATASET_NAME)
    parser.add_argument("--templates",       nargs="*", default=None,
                        help="Run only these template names (default: all)")
    parser.add_argument("--neg-sample-rate", type=float, default=0.1,
                        help="Fraction of unaffected probed partitions to keep as "
                             "negative examples per query (default: 0.1)")
    parser.add_argument("--no-gt",           action="store_true",
                        help="Skip brute-force GT; recall_drop relative to fresh top-k")
    args = parser.parse_args()

    t_start = time.time()

    # ── Shared resources (loaded once) ────────────────────────────────────────
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

    active_templates = TEMPLATES
    if args.templates:
        name_set = set(args.templates)
        active_templates = [t for t in TEMPLATES if t.name in name_set]
        missing = name_set - {t.name for t in active_templates}
        if missing:
            print(f"WARNING: unknown template names: {missing}")

    total_combos = sum(len(t.query_types) for t in active_templates)
    print(f"  Running {len(active_templates)} templates, "
          f"{total_combos} (template × query_type) combos")

    all_rows: list[dict] = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for template in active_templates:
        active_query_types = list(template.query_types)

        print(f"\n{'='*64}")
        print(f"Template: {template.name}")
        print(f"  type={template.mutation_type}  frac={template.mutation_fraction}"
              f"  drift={template.drift_magnitude}  pattern={template.spatial_pattern}"
              f"  query_types={active_query_types}")

        # ── Step 1: build snapshot ────────────────────────────────────────────
        if isinstance(template, ChainedWorkloadTemplate):
            # Chained: two sub-templates applied sequentially with independent
            # partition sets. Sample hot queries once if any sub-template needs
            # contribution scoring; reuse across sub-templates.
            needs_hot = any(
                st.spatial_pattern in (_CONTRIBUTION_PATTERNS | _RANK_TARGETED_PATTERNS)
                for st in template.sub_templates
            )
            if needs_hot:
                hot_queries = sample_queries(
                    all_queries, "hot", args.n_queries, centroid_index, args.n_probe,
                    partitions=base_partitions,
                    rng=np.random.default_rng(hash(template.name + "hot") % (2 ** 31)),
                )
                print(f"  Computing per-sub-template scores from {len(hot_queries)} hot queries ...")
            else:
                hot_queries = None

            contribution_scores_list = []
            for sub_t in template.sub_templates:
                if sub_t.spatial_pattern in _CONTRIBUTION_PATTERNS:
                    scores = compute_partition_contributions(
                        hot_queries, base_partitions, centroid_index, args.n_probe, args.topk,
                    )
                elif sub_t.spatial_pattern in _RANK_TARGETED_PATTERNS:
                    scores = compute_rank_targeted_scores(
                        hot_queries, centroid_index, args.n_probe, sub_t.rank_range,
                    )
                else:
                    scores = None
                contribution_scores_list.append(scores)

            snapshot = build_chained_snapshot(
                base_partitions, centroids, template, contribution_scores_list,
            )

        elif template.spatial_pattern in _CONTRIBUTION_PATTERNS:
            # Contribution-based: sample hot queries first, score contributions,
            # then build snapshot so targeted partitions align with the labeling batch.
            hot_queries = sample_queries(
                all_queries, "hot", args.n_queries, centroid_index, args.n_probe,
                partitions=base_partitions,
                rng=np.random.default_rng(hash(template.name + "hot") % (2 ** 31)),
            )
            print(f"  Computing contribution scores from {len(hot_queries)} hot queries ...")
            contribution_scores = compute_partition_contributions(
                hot_queries, base_partitions, centroid_index, args.n_probe, args.topk,
            )
            snapshot = build_mutated_snapshot(
                base_partitions, centroids, template,
                contribution_scores=contribution_scores,
            )
        elif template.spatial_pattern in _RANK_TARGETED_PATTERNS:
            # Rank-targeted: sample hot queries, score by rank window appearance count,
            # then build snapshot targeting partitions at the specified probe rank positions.
            hot_queries = sample_queries(
                all_queries, "hot", args.n_queries, centroid_index, args.n_probe,
                partitions=base_partitions,
                rng=np.random.default_rng(hash(template.name + "hot") % (2 ** 31)),
            )
            print(f"  Computing rank-targeted scores (ranks {template.rank_range}) "
                  f"from {len(hot_queries)} hot queries ...")
            rank_scores = compute_rank_targeted_scores(
                hot_queries, centroid_index, args.n_probe, template.rank_range,
            )
            snapshot = build_mutated_snapshot(
                base_partitions, centroids, template,
                contribution_scores=rank_scores,
            )
        else:
            # uniform / concentrated: partition selection is query-independent
            hot_queries = None
            snapshot = build_mutated_snapshot(base_partitions, centroids, template)

        print(f"  Affected partitions ({len(snapshot.affected_cids)}): {snapshot.affected_cids}")
        for cid in snapshot.affected_cids:
            log  = snapshot.mutation_log[cid]
            size = len(snapshot.stale_partitions[cid][0])
            print(f"    cid={cid}  size={size}  upd={log.updates}  del={log.deletes}")

        # ── Step 2: sample query batch ────────────────────────────────────────
        queries_by_type: dict[str, list] = {}
        for qt in active_query_types:
            if qt == "hot" and hot_queries is not None:
                # Already sampled above for contribution scoring — reuse
                queries_by_type["hot"] = hot_queries
            else:
                queries_by_type[qt] = sample_queries(
                    all_queries, qt, args.n_queries, centroid_index, args.n_probe,
                    partitions=base_partitions,
                    rng=np.random.default_rng(hash(template.name + qt) % (2 ** 31)),
                    # boundary queries target the Voronoi boundary of affected partitions
                    target_cids=snapshot.affected_cids if qt == "boundary" else None,
                )

        # ── Step 3: brute-force GT (post-mutation, once per template) ─────────
        if not args.no_gt:
            print("  Assembling post-mutation index for GT ...")
            cids_s         = sorted(snapshot.all_partitions)
            all_ids_fresh  = np.concatenate([snapshot.all_partitions[c][0] for c in cids_s])
            all_vecs_fresh = np.concatenate([snapshot.all_partitions[c][1] for c in cids_s], axis=0)

        unique_queries: dict[int, np.ndarray] = {
            q_idx: q_vec
            for qt in active_query_types
            for q_idx, q_vec in queries_by_type[qt]
        }

        gt_lookup: dict[int, np.ndarray] | None = None
        if not args.no_gt:
            print(f"  Computing GT for {len(unique_queries)} unique queries ...")
            gt_lookup = {
                q_idx: brute_force_topk(
                    q_vec.reshape(1, -1), all_ids_fresh, all_vecs_fresh, args.topk
                )[0]
                for q_idx, q_vec in unique_queries.items()
            }

        # ── Step 4: label + extract features ─────────────────────────────────
        for qt in active_query_types:
            queries = queries_by_type[qt]
            print(f"\n  Query type: {qt}  ({len(queries)} queries)")

            fresh_results, label_rows, partition_hit_rates = compute_all_labels(
                queries, snapshot, centroid_index, args.n_probe, args.topk, gt_lookup,
                neg_sample_rate=args.neg_sample_rate,
                rng=np.random.default_rng(hash(template.name + qt + "neg") % (2 ** 31)),
                filter_non_contributing=(template.spatial_pattern == "marginal_contributor"),
                affected_zero_sample_rate=(0.10 if template.spatial_pattern == "uniform" else 1.0),
            )
            print(f"    {len(label_rows)} label rows  "
                  f"(nonzero recall_drop: "
                  f"{sum(1 for lr in label_rows if lr.recall_drop > 0)})")

            fr_by_qidx = {fr.query_idx: fr for fr in fresh_results}

            for lr in label_rows:
                fr  = fr_by_qidx[lr.query_idx]
                row = extract_features_for_qp(
                    query_idx           = lr.query_idx,
                    partition_id        = lr.partition_id,
                    probe_rank          = lr.probe_rank,
                    n_probe             = args.n_probe,
                    centroid_distances  = fr.centroid_distances,
                    snapshot            = snapshot,
                    fresh_recall        = lr.fresh_recall,
                    stale_recall        = lr.stale_recall,
                    recall_drop         = lr.recall_drop,
                    probe_ids           = fr.probe_ids,
                    partition_hit_rates = partition_hit_rates,
                )
                row["query_type"] = qt
                all_rows.append(row)

    # ── Write dataset ─────────────────────────────────────────────────────────
    elapsed = round(time.time() - t_start, 1)
    print(f"\n{'='*64}")
    print(f"Total rows: {len(all_rows):,}  |  elapsed: {elapsed}s")

    metadata = {
        "n_probe":              args.n_probe,
        "topk":                 args.topk,
        "n_queries_per_combo":  args.n_queries,
        "templates":            [t.name for t in active_templates],
        "template_query_types": {t.name: list(t.query_types) for t in active_templates},
        "neg_sample_rate":      args.neg_sample_rate,
        "use_brute_force_gt":   not args.no_gt,
        "elapsed_seconds":      elapsed,
        "targeting":            "contribution_based",
    }

    paths = write_training_dataset(all_rows, args.output_dir, args.dataset_name, metadata)
    print("\nDataset written:")
    for k, v in paths.items():
        print(f"  {k:10s}: {v}")


if __name__ == "__main__":
    main()
