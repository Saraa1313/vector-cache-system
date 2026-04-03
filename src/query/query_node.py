import threading
import time
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import N_PROBE, CENTROIDS_PATH
from storage.object_store import ObjectStore
from storage.wal import WALClient

_DEFAULT_N_PROBE = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE


class QueryNode:
    def __init__(self):
        self.centroids = np.load(CENTROIDS_PATH)
        self.store = ObjectStore()
        self._wal = WALClient()
        self._seq = 0
        self._seq_lock = threading.Lock()

        d = self.centroids.shape[1]
        self._centroid_index = faiss.IndexFlatL2(d)
        self._centroid_index.add(np.ascontiguousarray(self.centroids))
        print("Query node ready", flush=True)

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
        with ThreadPoolExecutor(max_workers=len(probe_ids)) as executor:
            futures = {executor.submit(self.store.load_centroid, int(cid)): cid
                       for cid in probe_ids}
            fetched = {cid: future.result() for future, cid in
                       ((f, futures[f]) for f in as_completed(futures))}
        candidate_ids = [fetched[cid][0] for cid in probe_ids]
        candidate_vecs = [fetched[cid][1] for cid in probe_ids]
        fetch_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        candidate_ids = np.concatenate(candidate_ids)
        candidate_vecs = np.concatenate(candidate_vecs, axis=0)
        dists = ((candidate_vecs - query) ** 2).sum(axis=1)
        top_idx = np.argsort(dists)[:topk]
        scan_ms = (time.perf_counter() - t2) * 1000

        results = [{"id": int(candidate_ids[i]), "distance": float(dists[i])} for i in top_idx]
        return results, centroid_search_ms, fetch_ms, scan_ms

    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def insert(self, vector: np.ndarray) -> int:
        new_id = self._wal.next_vector_id()
        self._wal.write(self._next_seq(), "add", vector=vector, vector_id=new_id)
        return new_id

    def delete(self, vector_id: int) -> bool:
        self._wal.write(self._next_seq(), "delete", vector_id=vector_id)
        return True

    def update(self, vector_id: int, new_vector: np.ndarray) -> bool:
        self._wal.write(self._next_seq(), "update", vector=new_vector, vector_id=vector_id)
        return True
