import os
import sys
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)
sys.path.append(os.path.join(_src, "proto"))

import argparse
import time
import numpy as np
import grpc

import proto.vector_search_pb2 as pb2
import proto.vector_search_pb2_grpc as pb2_grpc

from config import DATA_DIR, QUERY_NODE_HOST, GRPC_PORT, TOPK, NUMBER_OF_QUERIES, N_PROBE
from admin.data_loader import read_fvecs, read_ivecs


def compute_recall(I_pred, gt, k):
    recalls = []
    for pred_row, gt_row in zip(I_pred, gt):
        gt_set = set(int(x) for x in gt_row[:k])
        found = sum(1 for pid in pred_row if pid in gt_set)
        recalls.append(found / k)
    return float(np.mean(recalls))


def run_nprobe(stub, queries, gt, nprobe):
    I_pred = []
    errors = 0
    total_centroid_ms = total_fetch_ms = total_scan_ms = 0.0
    total_cache_hits = total_cache_probes = 0
    t0 = time.time()
    completed_queries = 0

    for q in queries:
        try:
            resp = stub.Search(pb2.SearchRequest(vector=q.tolist(), top_k=TOPK, n_probe=nprobe))
            I_pred.append([n.id for n in resp.results])
            total_centroid_ms   += resp.centroid_search_ms
            total_fetch_ms      += resp.fetch_ms
            total_scan_ms       += resp.scan_ms
            total_cache_hits    += resp.cache_hits
            total_cache_probes  += nprobe
            completed_queries += 1
            if completed_queries % 1000 == 0:
                print(f"  Completed {completed_queries} queries")
        except grpc.RpcError as e:
            print(f"  Error: {e}")
            errors += 1
            I_pred.append([])

    elapsed_ms = (time.time() - t0) * 1000
    n = len(queries)
    recall = compute_recall(I_pred, gt, TOPK)
    hit_pct = 100.0 * total_cache_hits / total_cache_probes if total_cache_probes else 0.0
    print(f"  nprobe={nprobe:4d} | "
          f"avg total={elapsed_ms/n:.2f} ms  "
          f"centroid={total_centroid_ms/n:.2f} ms  "
          f"fetch={total_fetch_ms/n:.2f} ms  "
          f"scan={total_scan_ms/n:.2f} ms  "
          f"cache_hit={hit_pct:.1f}%  "
          f"recall@{TOPK}={recall:.4f}  errors={errors}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default=None,
                        help="Path to ground truth .npy file (overrides sift_groundtruth.ivecs)")
    parser.add_argument("--queries", default=None,
                        help="Path to CSV with query_idx column to select specific queries")
    parser.add_argument("--no-clear-cache", action="store_true",
                        help="Skip ClearCache RPC before each nprobe sweep (use to measure stale-cache recall)")
    args = parser.parse_args()

    import csv
    all_queries = read_fvecs(os.path.join(DATA_DIR, "sift", "sift_query.fvecs"))

    if args.queries:
        with open(args.queries) as f:
            indices = [int(row["query_idx"]) for row in csv.DictReader(f)]
        queries = all_queries[indices]
        print(f"Using {len(indices)} queries from {args.queries}")
    else:
        indices = list(range(NUMBER_OF_QUERIES))
        queries = all_queries[:NUMBER_OF_QUERIES]

    if args.gt:
        gt = np.load(args.gt)   # already aligned with selected queries
        print(f"Using ground truth: {args.gt}")
    else:
        all_gt = read_ivecs(os.path.join(DATA_DIR, "sift", "sift_groundtruth.ivecs"))
        gt = all_gt[indices]
    print(f"Running {len(queries)} queries for nprobe values: {N_PROBE}")

    channel = grpc.insecure_channel(f"{QUERY_NODE_HOST}:{GRPC_PORT}")
    stub = pb2_grpc.VectorSearchStub(channel)

    for nprobe in N_PROBE:
        if not args.no_clear_cache:
            stub.ClearCache(pb2.ClearCacheRequest())
        run_nprobe(stub, queries, gt, nprobe)


if __name__ == "__main__":
    main()
