import os
import sys
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)
sys.path.append(os.path.join(_src, "proto"))

import grpc
from concurrent import futures

import proto.vector_search_pb2_grpc as pb2_grpc
from config import QUERY_NODE_HOST, GRPC_PORT
from query.query_node import QueryNode
from query.servicer import VectorSearchServicer


def serve(model_path: str | None = None, use_learned_policy: bool = False):
    node = QueryNode(model_path=model_path, use_learned_policy=use_learned_policy)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_VectorSearchServicer_to_server(VectorSearchServicer(node), server)
    server.add_insecure_port(f"0.0.0.0:{GRPC_PORT}")
    server.start()
    print(f"Query node listening on 0.0.0.0:{GRPC_PORT}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Path to .ubj XGBoost model (enables shadow/learned policy)")
    parser.add_argument("--use-learned-policy", action="store_true",
                        help="Act on learned policy decisions (default: shadow mode only)")
    args = parser.parse_args()
    serve(model_path=args.model_path, use_learned_policy=args.use_learned_policy)
