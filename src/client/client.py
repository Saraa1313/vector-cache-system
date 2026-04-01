import os
import sys
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)
sys.path.append(os.path.join(_src, "proto"))

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
    t0 = time.time()

    for q in queries:
        try:
            resp = stub.Search(pb2.SearchRequest(vector=q.tolist(), top_k=TOPK, n_probe=nprobe))
            I_pred.append([n.id for n in resp.results])
        except grpc.RpcError as e:
            print(f"  Error: {e}")
            errors += 1
            I_pred.append([])

    elapsed = time.time() - t0
    elapsed_ms = elapsed * 1000.0
    recall = compute_recall(I_pred, gt, TOPK)
    print(f"  nprobe={nprobe:4d}  total={elapsed_ms:.1f} ms  avg={elapsed_ms/len(queries):.2f} ms/query  "
          f"qps={len(queries)/elapsed:.1f}  recall@{TOPK}={recall:.4f}  errors={errors}")


def main():
    queries = read_fvecs(os.path.join(DATA_DIR, "sift", "sift_query.fvecs"))[:NUMBER_OF_QUERIES]
    gt = read_ivecs(os.path.join(DATA_DIR, "sift", "sift_groundtruth.ivecs"))[:NUMBER_OF_QUERIES]
    print(f"Running {NUMBER_OF_QUERIES} queries for nprobe values: {N_PROBE}")

    channel = grpc.insecure_channel(f"{QUERY_NODE_HOST}:{GRPC_PORT}")
    stub = pb2_grpc.VectorSearchStub(channel)

    for nprobe in N_PROBE:
        run_nprobe(stub, queries, gt, nprobe)


if __name__ == "__main__":
    main()
