import threading
import time
import numpy as np
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import N_PROBE, CENTROIDS_PATH, CACHE_SIZE
from query.lru_cache import LRUCache
from query.recall_policy import RecallAwarePolicy, PolicyConfig
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

        self._cache = LRUCache(CACHE_SIZE)
        self._freshness: dict[int, dict] = {}
        self._freshness_lock = threading.Lock()
        self._policy = RecallAwarePolicy(PolicyConfig(n_probe=_DEFAULT_N_PROBE))
        print("Query node ready", flush=True)

    def _nearest_centroid(self, vector: np.ndarray) -> int:
        q = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        _, I = self._centroid_index.search(q, 1)
        return int(I[0, 0])

    def search(self, query: np.ndarray, topk: int, n_probe: int = _DEFAULT_N_PROBE):
        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))

        t0 = time.perf_counter()
        _, I = self._centroid_index.search(q, n_probe)
        probe_ids = [int(cid) for cid in I[0] if cid >= 0]
        centroid_search_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        entries: dict[int, tuple] = {}
        fetch_ids = []  # partitions to fetch from MinIO (not cached, or policy says stale)

        with self._freshness_lock:
            for rank, cid in enumerate(probe_ids, start=1):
                cid = int(cid)
                entry = self._cache.get(cid)
                if entry is None:
                    # Not in cache — must fetch
                    fetch_ids.append(cid)
                else:
                    f = self._freshness.get(cid)
                    if f is None:
                        # In cache but no freshness entry — use cache conservatively
                        entries[cid] = entry
                    else:
                        fetch, _ = self._policy.should_fetch(f, probe_rank=rank)
                        if fetch:
                            fetch_ids.append(cid)
                        else:
                            entries[cid] = entry

        cache_hits = len(probe_ids) - len(fetch_ids)

        if fetch_ids:
            with ThreadPoolExecutor(max_workers=len(fetch_ids)) as executor:
                futures = {executor.submit(self.store.load_centroid, cid): cid
                           for cid in fetch_ids}
                for future in as_completed(futures):
                    cid = futures[future]
                    ids, vecs, version = future.result()
                    self._cache.put(cid, (ids, vecs, version))
                    entries[cid] = (ids, vecs, version)
                    self._init_freshness(cid, version, partition_size=len(ids))

        candidate_ids = [entries[int(cid)][0] for cid in probe_ids]
        candidate_vecs = [entries[int(cid)][1] for cid in probe_ids]
        fetch_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        candidate_ids = np.concatenate(candidate_ids)
        candidate_vecs = np.concatenate(candidate_vecs, axis=0)
        dists = ((candidate_vecs - query) ** 2).sum(axis=1)
        top_idx = np.argsort(dists)[:topk]
        scan_ms = (time.perf_counter() - t2) * 1000

        results = [{"id": int(candidate_ids[i]), "distance": float(dists[i])} for i in top_idx]
        return results, centroid_search_ms, fetch_ms, scan_ms, cache_hits

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

    def _init_freshness(self, cid: int, cached_version: int, partition_size: int) -> None:
        with self._freshness_lock:
            self._freshness[cid] = {
                "cached_version":                       cached_version,
                "latest_known_version":                 cached_version,
                "cumulative_inserts_since_cache":       0,
                "cumulative_updates_since_cache":       0,
                "cumulative_deletes_since_cache":       0,
                "cumulative_membership_changes_since_cache": 0,
                "latest_partition_size":                partition_size,
                "latest_fraction_vectors_touched":      None,
                "latest_reconstruction_error":          None,
                "latest_centroid":                      None,
                "last_metadata_update_time":            None,
            }

    def on_batch_applied(self, partition_deltas: list, last_seq_id: int) -> None:
        with self._freshness_lock:
            for delta in partition_deltas:
                cid = delta.partition_id
                if cid not in self._freshness:
                    # partition not cached — nothing to track yet
                    continue
                f = self._freshness[cid]
                if delta.version_id <= f["latest_known_version"]:
                    continue  # duplicate or out-of-order notification
                # accumulate deltas since cached version
                f["cumulative_inserts_since_cache"]             += delta.number_of_inserts
                f["cumulative_updates_since_cache"]             += delta.number_of_updates
                f["cumulative_deletes_since_cache"]             += delta.number_of_deletes
                f["cumulative_membership_changes_since_cache"]  += delta.membership_change_count
                # overwrite latest snapshot fields
                f["latest_known_version"]           = delta.version_id
                f["latest_partition_size"]          = delta.partition_size
                f["latest_fraction_vectors_touched"]= delta.fraction_vectors_touched
                f["latest_reconstruction_error"]    = delta.reconstruction_error
                f["latest_centroid"]                = list(delta.new_centroid)
                f["last_metadata_update_time"]      = time.time()
        print(f"Query node received batch applied: seq={last_seq_id} partitions={[d.partition_id for d in partition_deltas]}", flush=True)
        with self._freshness_lock:
            for cid, f in sorted(self._freshness.items()):
                print(f"  [freshness] cid={cid}"
                      f"  cached_ver={f['cached_version']}"
                      f"  latest_ver={f['latest_known_version']}"
                      f"  cum_ins={f['cumulative_inserts_since_cache']}"
                      f"  cum_upd={f['cumulative_updates_since_cache']}"
                      f"  cum_del={f['cumulative_deletes_since_cache']}"
                      f"  cum_mem={f['cumulative_membership_changes_since_cache']}"
                      f"  size={f['latest_partition_size']}"
                      f"  re={f['latest_reconstruction_error']}", flush=True)

    def get_cached_version(self, partition_id: int) -> int | None:
        entry = self._cache.get(partition_id)
        return entry[2] if entry is not None else None

    def clear_cache(self) -> None:
        self._cache = LRUCache(CACHE_SIZE)
        with self._freshness_lock:
            self._freshness.clear()

    def update(self, vector_id: int, new_vector: np.ndarray) -> bool:
        self._wal.write(self._next_seq(), "update", vector=new_vector, vector_id=vector_id)
        return True
