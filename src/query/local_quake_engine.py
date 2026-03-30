import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admin.quake_backend import QuakeBackend
from query.cache_manager import CacheManager


class LocalQuakeEngine:
    def __init__(self):
        self.cache = CacheManager()
        self.backend = QuakeBackend()
        self.loaded_snapshot_version = None
        self.num_loaded_vectors = 0

    def rebuild_from_cache(self):
        cache_meta = self.cache.read_cache_metadata()

        if "_snapshot" not in cache_meta:
            raise RuntimeError("No cached snapshot metadata found. Warm cache first.")

        ids_all = []
        vecs_all = []

        for fname, info in cache_meta.items():
            if fname == "_snapshot":
                continue

            ids, vecs = self.cache.load_partition(fname)
            ids_all.append(ids)
            vecs_all.append(vecs)

        if not ids_all:
            raise RuntimeError("No cached partitions found.")

        ids_all = np.concatenate(ids_all).astype(np.int64)
        vecs_all = np.concatenate(vecs_all).astype(np.float32)

        # nlist can be read from snapshot metadata if present
        nlist = cache_meta["_snapshot"].get("nlist", 4)

        self.backend = QuakeBackend()
        self.backend.build(vecs_all, ids_all, nlist=nlist)

        self.loaded_snapshot_version = cache_meta["_snapshot"]["snapshot_version"]
        self.num_loaded_vectors = len(ids_all)

    def maybe_rebuild(self):
        cache_meta = self.cache.read_cache_metadata()
        cached_snapshot = cache_meta.get("_snapshot", {}).get("snapshot_version")

        if cached_snapshot is None:
            raise RuntimeError("Cache has no snapshot metadata.")

        if self.loaded_snapshot_version != cached_snapshot:
            self.rebuild_from_cache()

    def search(self, queries: np.ndarray, k: int = 5, nprobe: int = 2):
        self.maybe_rebuild()
        return self.backend.search(queries, k=k, nprobe=nprobe)

