import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_DIR, TOPK, NUMBER_OF_QUERIES
from admin.data_loader import read_fvecs
from storage.object_store import ObjectStore


def load_all_vectors(store: ObjectStore) -> tuple[np.ndarray, np.ndarray]:
    centroid_ids = store.list_centroid_ids()
    print(f"Loading {len(centroid_ids)} partitions from MinIO ...", flush=True)

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(store.load_centroid, cid): cid for cid in centroid_ids}
        results = []
        for i, future in enumerate(as_completed(futures)):
            ids, vecs, _ = future.result()
            if len(ids) > 0:
                results.append((ids, vecs))
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(centroid_ids)} partitions loaded", flush=True)

    all_ids = np.concatenate([r[0] for r in results])
    all_vecs = np.concatenate([r[1] for r in results], axis=0)
    print(f"  Total vectors in index: {len(all_ids)}", flush=True)
    return all_ids, all_vecs


def brute_force_topk(queries: np.ndarray, all_ids: np.ndarray,
                     all_vecs: np.ndarray, topk: int) -> np.ndarray:
    nq = len(queries)
    gt = np.zeros((nq, topk), dtype=np.int64)
    print(f"Computing exact top-{topk} for {nq} queries ...", flush=True)
    for i, q in enumerate(queries):
        dists = ((all_vecs - q) ** 2).sum(axis=1)
        top_idx = np.argpartition(dists, topk)[:topk]
        top_idx = top_idx[np.argsort(dists[top_idx])]
        gt[i] = all_ids[top_idx]
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{nq} queries done", flush=True)
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(DATA_DIR, "sift", "current_groundtruth.npy"),
                        help="Output path for ground truth array (.npy)")
    parser.add_argument("--nqueries", type=int, default=NUMBER_OF_QUERIES,
                        help="Number of queries to compute ground truth for (ignored if --queries given)")
    parser.add_argument("--queries", default=None,
                        help="CSV with query_idx column to select specific queries")
    parser.add_argument("--topk", type=int, default=TOPK)
    args = parser.parse_args()

    query_path = os.path.join(DATA_DIR, "sift", "sift_query.fvecs")
    all_queries = read_fvecs(query_path).astype(np.float32)

    if args.queries:
        import csv
        with open(args.queries) as f:
            indices = [int(row["query_idx"]) for row in csv.DictReader(f)]
        queries = all_queries[indices]
        print(f"Using {len(indices)} queries from {args.queries}")
    else:
        queries = all_queries[:args.nqueries]

    store = ObjectStore()
    all_ids, all_vecs = load_all_vectors(store)

    gt = brute_force_topk(queries, all_ids, all_vecs, args.topk)

    np.save(args.out, gt)
    print(f"Ground truth saved → {args.out}  shape={gt.shape}")


if __name__ == "__main__":
    main()
