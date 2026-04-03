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

from config import DIM, QUERY_NODE_HOST, GRPC_PORT


def main():
    if len(sys.argv) != 2:
        print("Usage: python insert_client.py <num_inserts>")
        sys.exit(1)
    n = int(sys.argv[1])
    vectors = np.random.randn(n, DIM).astype(np.float32)

    channel = grpc.insecure_channel(f"{QUERY_NODE_HOST}:{GRPC_PORT}")
    stub = pb2_grpc.VectorSearchStub(channel)

    print(f"Sending {n} inserts ...")
    assigned_ids = []
    errors = 0
    t0 = time.time()

    for vec in vectors:
        try:
            resp = stub.Insert(pb2.InsertRequest(vector=vec.tolist()))
            assigned_ids.append(resp.id)
        except grpc.RpcError as e:
            print(f"  Error: {e}")
            errors += 1

    elapsed_ms = (time.time() - t0) * 1000
    n = len(assigned_ids)
    print(f"  Inserted {n} vectors in {elapsed_ms:.1f} ms  "
          f"avg={elapsed_ms/max(n,1):.2f} ms/insert  "
          f"errors={errors}")
    if assigned_ids:
        print(f"  ID range: {min(assigned_ids)} – {max(assigned_ids)}")


if __name__ == "__main__":
    main()
