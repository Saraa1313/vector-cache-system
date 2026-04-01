import sys
import os
_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_src)
sys.path.append(os.path.join(_src, "proto"))

import numpy as np
import proto.vector_search_pb2 as pb2
import proto.vector_search_pb2_grpc as pb2_grpc

from config import TOPK, N_PROBE

_DEFAULT_N_PROBE = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE


class VectorSearchServicer(pb2_grpc.VectorSearchServicer):
    def __init__(self, node):
        self.node = node

    def Search(self, request, context):
        query = np.array(request.vector, dtype=np.float32)
        topk = request.top_k or TOPK
        n_probe = request.n_probe or _DEFAULT_N_PROBE
        hits = self.node.search(query, topk=topk, n_probe=n_probe)
        neighbors = [pb2.Neighbor(id=h["id"], distance=h["distance"]) for h in hits]
        return pb2.SearchResponse(results=neighbors)

    def Insert(self, request, context):
        vector = np.array(request.vector, dtype=np.float32)
        new_id = self.node.insert(vector)
        return pb2.InsertResponse(id=new_id)

    def Delete(self, request, context):
        success = self.node.delete(request.id)
        return pb2.DeleteResponse(success=success)

    def Update(self, request, context):
        vector = np.array(request.new_vector, dtype=np.float32)
        success = self.node.update(request.id, vector)
        return pb2.UpdateResponse(success=success)
