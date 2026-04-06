import os
import sys
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)
sys.path.append(os.path.join(_src, "proto"))

import argparse
import csv
import time
import numpy as np
import grpc

import proto.vector_search_pb2 as pb2
import proto.vector_search_pb2_grpc as pb2_grpc

from config import DIM, QUERY_NODE_HOST, GRPC_PORT


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", metavar="PATH", help="CSV file with vector_id and dim_0..dim_127")
    group.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                       help="Update all IDs in [START, END] with random vectors")
    group.add_argument("--ids", nargs="+", type=int, metavar="ID",
                       help="Specific IDs to update with random vectors")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of updates to send")
    args = parser.parse_args()

    if args.csv:
        ids = []
        new_vectors = []
        with open(args.csv) as f:
            for row in csv.DictReader(f):
                ids.append(int(row["vector_id"]))
                new_vectors.append([float(row[f"dim_{i}"]) for i in range(DIM)])
        new_vectors = np.array(new_vectors, dtype=np.float32)
    elif args.range:
        start, end = args.range
        ids = list(range(start, end + 1))
        new_vectors = np.random.randn(len(ids), DIM).astype(np.float32)
    else:
        ids = args.ids
        new_vectors = np.random.randn(len(ids), DIM).astype(np.float32)

    if args.limit is not None:
        ids = ids[:args.limit]
        new_vectors = new_vectors[:args.limit]

    channel = grpc.insecure_channel(f"{QUERY_NODE_HOST}:{GRPC_PORT}")
    stub = pb2_grpc.VectorSearchStub(channel)

    print(f"Sending {len(ids)} updates ...")
    errors = 0
    t0 = time.time()

    for vid, vec in zip(ids, new_vectors):
        try:
            stub.Update(pb2.UpdateRequest(id=vid, new_vector=vec.tolist()))
        except grpc.RpcError as e:
            print(f"  Error updating {vid}: {e}")
            errors += 1

    elapsed_ms = (time.time() - t0) * 1000
    n = len(ids) - errors
    print(f"  Updated {n} vectors in {elapsed_ms:.1f} ms  "
          f"avg={elapsed_ms/max(len(ids),1):.2f} ms/update  "
          f"errors={errors}")


if __name__ == "__main__":
    main()
