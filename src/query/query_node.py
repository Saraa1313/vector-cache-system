import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CACHE_DIR, TOPK


class QueryNode:
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager

    def _load_all_cached_vectors(self):
        all_ids = []
        all_vecs = []

        for fname in sorted(os.listdir(CACHE_DIR)):
            if fname.endswith(".npz") and self.cache_manager.has_partition(fname):
                ids, vecs = self.cache_manager.load_partition(fname)
                all_ids.append(ids)
                all_vecs.append(vecs)

        if not all_ids:
            return np.array([]), np.empty((0, 0), dtype=np.float32)

        return np.concatenate(all_ids), np.concatenate(all_vecs)

    def search(self, query, topk=TOPK):
        ids, vecs = self._load_all_cached_vectors()
        if len(ids) == 0:
            return []

        dists = np.linalg.norm(vecs - query[None, :], axis=1)
        order = np.argsort(dists)[:topk]

        return [
            {"id": int(ids[i]), "distance": float(dists[i])}
            for i in order
        ]