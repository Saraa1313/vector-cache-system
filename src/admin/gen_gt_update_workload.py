"""
Generate an update workload CSV targeting the ground-truth neighbors of a
specific query set.  Updating these vectors to random values makes the LRU
cache stale: post-mutation queries served from cache return wrong neighbors,
so recall drops against the recomputed ground truth.

Usage:
    python src/admin/gen_gt_update_workload.py \
        --queries src/client/concentrated_queries_top10.csv \
        --topk 5 \
        --out data/gt_update_workload.csv
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import numpy as np
from config import DATA_DIR, TOPK, DIM
from admin.data_loader import read_ivecs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True,
                        help="CSV with query_idx column (e.g. concentrated_queries_top10.csv)")
    parser.add_argument("--topk", type=int, default=TOPK,
                        help="How many GT neighbors per query to include (controls recall drop magnitude)")
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "gt_update_workload.csv"),
                        help="Output CSV: vector_id, dim_0..dim_127")
    args = parser.parse_args()

    gt_path = os.path.join(DATA_DIR, "sift", "sift_groundtruth.ivecs")
    gt = read_ivecs(gt_path)

    with open(args.queries) as f:
        indices = [int(row["query_idx"]) for row in csv.DictReader(f)]

    ids = sorted({int(x) for idx in indices for x in gt[idx][:args.topk]})

    rng = np.random.default_rng(42)
    new_vecs = rng.standard_normal((len(ids), DIM)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vector_id"] + [f"dim_{i}" for i in range(DIM)])
        for vid, vec in zip(ids, new_vecs):
            writer.writerow([vid] + vec.tolist())

    print(f"Queries covered  : {len(indices)}")
    print(f"GT neighbors/q   : {args.topk}")
    print(f"Unique IDs to update: {len(ids)}")
    print(f"Written to: {args.out}")


if __name__ == "__main__":
    main()