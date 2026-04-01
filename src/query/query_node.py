import threading
import time
import numpy as np
import faiss

from config import N_PROBE, CENTROIDS_PATH, CACHE_SIZE
from query.lru_cache import LRUCache

_DEFAULT_N_PROBE = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE
from storage.object_store import ObjectStore


class QueryNode:
    def __init__(self):
        self.centroids = np.load(CENTROIDS_PATH) 
        self.store = ObjectStore()
        self._mut_lock = threading.Lock()

        print("Scanning MinIO to build id→centroid map", flush=True)
        self.id_to_centroid: dict[int, int] = {}
        max_id = -1
        for cid in self.store.list_centroid_ids():
            ids, _ = self.store.load_centroid(cid)
            for vid in ids:
                self.id_to_centroid[int(vid)] = cid
                if int(vid) > max_id:
                    max_id = int(vid)
        self.next_id = max_id + 1
        print(f"  {len(self.id_to_centroid)} vectors across {len(self.centroids)} centroids", flush=True)

        # Faiss index over centroids for fast nearest-centroid lookup
        d = self.centroids.shape[1]
        self._centroid_index = faiss.IndexFlatL2(d)
        self._centroid_index.add(np.ascontiguousarray(self.centroids))

        self._cache = LRUCache(CACHE_SIZE)

    def _nearest_centroid(self, vector: np.ndarray) -> int:
        q = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        _, I = self._centroid_index.search(q, 1)
        return int(I[0, 0])

    def search(self, query: np.ndarray, topk: int, n_probe: int = _DEFAULT_N_PROBE):
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))

        t0 = time.perf_counter()
        _, I = self._centroid_index.search(q, n_probe)
        probe_ids = I[0]
        centroid_search_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        candidate_ids, candidate_vecs = [], []
        for cid in probe_ids:
            entry = self._cache.get(int(cid))
            if entry is None:
                entry = self.store.load_centroid(int(cid))
                self._cache.put(int(cid), entry)
            c_ids, c_vecs = entry
            candidate_ids.append(c_ids)
            candidate_vecs.append(c_vecs)
        fetch_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        candidate_ids = np.concatenate(candidate_ids)
        candidate_vecs = np.concatenate(candidate_vecs, axis=0)
        dists = ((candidate_vecs - query) ** 2).sum(axis=1)
        top_idx = np.argsort(dists)[:topk]
        scan_ms = (time.perf_counter() - t2) * 1000

        results = [{"id": int(candidate_ids[i]), "distance": float(dists[i])} for i in top_idx]
        return results, centroid_search_ms, fetch_ms, scan_ms


    def insert(self, vector: np.ndarray) -> int:
        cid = self._nearest_centroid(vector)
        with self._mut_lock:
            ids, vecs = self.store.load_centroid(cid)
            new_id = self.next_id
            ids = np.append(ids, np.int64(new_id))
            vecs = np.vstack([vecs, vector.reshape(1, -1)])
            self.store.save_centroid(cid, ids, vecs)
            self._cache.put(cid, (ids, vecs))
            self.id_to_centroid[new_id] = cid
            self.next_id += 1
        return new_id


    def delete(self, vector_id: int) -> bool:
        cid = self.id_to_centroid.get(vector_id)
        if cid is None:
            return False
        with self._mut_lock:
            ids, vecs = self.store.load_centroid(cid)
            mask = ids != vector_id
            self.store.save_centroid(cid, ids[mask], vecs[mask])
            self._cache.put(cid, (ids[mask], vecs[mask]))
            del self.id_to_centroid[vector_id]
        return True

    def update(self, vector_id: int, new_vector: np.ndarray) -> bool:
        old_cid = self.id_to_centroid.get(vector_id)
        if old_cid is None:
            return False
        new_cid = self._nearest_centroid(new_vector)
        with self._mut_lock:
            if old_cid == new_cid:
                ids, vecs = self.store.load_centroid(old_cid)
                vecs[ids == vector_id] = new_vector
                self.store.save_centroid(old_cid, ids, vecs)
                self._cache.put(old_cid, (ids, vecs))
            else:
                # Remove from old centroid
                ids, vecs = self.store.load_centroid(old_cid)
                mask = ids != vector_id
                self.store.save_centroid(old_cid, ids[mask], vecs[mask])
                self._cache.put(old_cid, (ids[mask], vecs[mask]))
                # Add to new centroid
                ids, vecs = self.store.load_centroid(new_cid)
                ids = np.append(ids, np.int64(vector_id))
                vecs = np.vstack([vecs, new_vector.reshape(1, -1)])
                self.store.save_centroid(new_cid, ids, vecs)
                self._cache.put(new_cid, (ids, vecs))
                self.id_to_centroid[vector_id] = new_cid
        return True
