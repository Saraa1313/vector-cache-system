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


def serve():
    node = QueryNode()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_VectorSearchServicer_to_server(VectorSearchServicer(node), server)
    server.add_insecure_port(f"0.0.0.0:{GRPC_PORT}")
    server.start()
    print(f"Query node listening on 0.0.0.0:{GRPC_PORT}", flush=True)
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
