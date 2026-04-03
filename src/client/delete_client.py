import os
import sys
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)
sys.path.append(os.path.join(_src, "proto"))

import time
import grpc

import proto.vector_search_pb2 as pb2
import proto.vector_search_pb2_grpc as pb2_grpc

from config import QUERY_NODE_HOST, GRPC_PORT


def main():
    if len(sys.argv) < 2:
        print("Usage: python delete_client.py <id1> [id2 ...]")
        print("       python delete_client.py <start> <end>     # delete range [start, end]")
        print("       python delete_client.py --file <path>     # delete IDs listed in file")
        sys.exit(1)

    if sys.argv[1] == "--file":
        with open(sys.argv[2]) as f:
            ids = [int(line.strip()) for line in f if line.strip()]
    elif len(sys.argv) == 3:
        start, end = int(sys.argv[1]), int(sys.argv[2])
        ids = list(range(start, end + 1))
    else:
        ids = [int(x) for x in sys.argv[1:]]

    channel = grpc.insecure_channel(f"{QUERY_NODE_HOST}:{GRPC_PORT}")
    stub = pb2_grpc.VectorSearchStub(channel)

    print(f"Sending {len(ids)} deletes ...")
    errors = 0
    t0 = time.time()

    for vid in ids:
        try:
            stub.Delete(pb2.DeleteRequest(id=vid))
        except grpc.RpcError as e:
            print(f"  Error deleting {vid}: {e}")
            errors += 1

    elapsed_ms = (time.time() - t0) * 1000
    n = len(ids) - errors
    print(f"  Deleted {n} vectors in {elapsed_ms:.1f} ms  "
          f"avg={elapsed_ms/max(len(ids),1):.2f} ms/delete  "
          f"errors={errors}")


if __name__ == "__main__":
    main()
