import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

# xgboost must be imported before faiss — both use native BLAS/OpenMP libraries
# and faiss claiming the thread pool first causes XGBoost load_model to segfault.
try:
    import xgboost as _xgb_preload  # noqa: F401
except ImportError:
    pass

import faiss

from config import N_PROBE, CENTROIDS_PATH, CACHE_SIZE
from query.lru_cache import LRUCache
from query.recall_policy import RecallAwarePolicy, LearnedPolicy, PolicyConfig
from storage.object_store import ObjectStore
from storage.wal import WALClient

_DEFAULT_N_PROBE = N_PROBE[0] if isinstance(N_PROBE, list) else N_PROBE
_VALID_POLICIES = {"heuristic", "learned", "always_fetch", "always_cache", "fetch_on_lag"}


@dataclass
class PartitionDecision:
    partition_id: int
    probe_rank: int
    in_cache: bool
    decision: str                    # "fetch" or "cache"
    learned_prediction: float | None # model P(unsafe), None if not learned policy
    fetch_latency_ms: float
    bytes_fetched: int


@dataclass
class QueryResult:
    top_k: list
    total_ms: float
    centroid_search_ms: float
    fetch_ms: float
    inference_ms: float
    scan_ms: float
    cache_hits: int
    fetch_count: int
    bytes_fetched: int
    cold_fetch_count: int = 0   # partition was not in cache at all
    policy_fetch_count: int = 0 # partition was cached but policy decided to refresh
    partition_log: list = field(default_factory=list)


class QueryNode:
    def __init__(self, model_path: str | None = None, use_learned_policy: bool = False):
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
        self._hit_rate_ema: dict[int, float] = {}  # per-partition EMA of top-k contribution
        self._hit_rate_alpha = 0.1
        self._policy = RecallAwarePolicy(PolicyConfig(n_probe=_DEFAULT_N_PROBE))

        self._learned_policy: LearnedPolicy | None = None
        if model_path is not None:
            self._learned_policy = LearnedPolicy(
                model_path, n_probe=_DEFAULT_N_PROBE,
                is_classifier=True,
            )
        self._use_learned_policy = use_learned_policy and (self._learned_policy is not None)
        print("Query node ready", flush=True)

    def _nearest_centroid(self, vector: np.ndarray) -> int:
        q = np.ascontiguousarray(vector.reshape(1, -1).astype(np.float32))
        _, I = self._centroid_index.search(q, 1)
        return int(I[0, 0])

    def search(
        self,
        query: np.ndarray,
        topk: int,
        n_probe: int = _DEFAULT_N_PROBE,
        fetch_policy: str = "heuristic",
    ) -> QueryResult:
        if fetch_policy not in _VALID_POLICIES:
            fetch_policy = "heuristic"

        q = np.ascontiguousarray(query.reshape(1, -1).astype(np.float32))

        t0 = time.perf_counter()
        D, I = self._centroid_index.search(q, n_probe)
        probe_ids = [int(cid) for cid in I[0] if cid >= 0]
        centroid_dists = [float(D[0][i]) for i in range(len(probe_ids))]
        centroid_search_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        entries: dict[int, tuple] = {}
        fetch_ids: list[int] = []
        inference_ms = 0.0
        partition_log: list[PartitionDecision] = []

        with self._freshness_lock:
            total_probe_size = sum(
                self._freshness[c]["latest_partition_size"]
                for c in probe_ids
                if c in self._freshness
            )
            for rank, cid in enumerate(probe_ids, start=1):
                cid = int(cid)
                entry = self._cache.get(cid)
                in_cache = entry is not None
                learned_pred: float | None = None

                if not in_cache:
                    # Not in cache — must fetch regardless of policy
                    decision = "fetch"
                    fetch_ids.append(cid)
                elif fetch_policy == "always_fetch":
                    decision = "fetch"
                    fetch_ids.append(cid)
                elif fetch_policy == "always_cache":
                    decision = "cache"
                    entries[cid] = entry
                elif fetch_policy == "fetch_on_lag":
                    f = self._freshness.get(cid)
                    if f is not None and f["latest_known_version"] > f["cached_version"]:
                        decision = "fetch"
                        fetch_ids.append(cid)
                    else:
                        decision = "cache"
                        entries[cid] = entry
                else:
                    # "learned" or "heuristic" — consult policy
                    f = self._freshness.get(cid)
                    if f is None:
                        decision = "cache"
                        entries[cid] = entry
                    else:
                        candidate_fraction = (
                            f["latest_partition_size"] / max(total_probe_size, 1)
                        )
                        if fetch_policy == "learned" and self._learned_policy is not None:
                            t_inf = time.perf_counter()
                            fetch, diag = self._learned_policy.should_fetch(
                                f, probe_rank=rank,
                                candidate_fraction=candidate_fraction,
                                centroid_distances=centroid_dists,
                            )
                            inference_ms += (time.perf_counter() - t_inf) * 1000
                            learned_pred = diag.get("predicted_recall_drop")
                        else:
                            fetch, _ = self._policy.should_fetch(
                                f, probe_rank=rank,
                                candidate_fraction=candidate_fraction,
                                centroid_distances=centroid_dists,
                            )
                        if fetch:
                            decision = "fetch"
                            fetch_ids.append(cid)
                        else:
                            decision = "cache"
                            entries[cid] = entry

                partition_log.append(PartitionDecision(
                    partition_id=cid,
                    probe_rank=rank,
                    in_cache=in_cache,
                    decision=decision,
                    learned_prediction=learned_pred,
                    fetch_latency_ms=0.0,  # filled in after fetch
                    bytes_fetched=0,       # filled in after fetch
                ))

        cache_hits = len(probe_ids) - len(fetch_ids)
        total_bytes_fetched = 0
        fetch_timings: dict[int, tuple[float, int]] = {}  # cid -> (latency_ms, bytes)

        if fetch_ids:
            with ThreadPoolExecutor(max_workers=len(fetch_ids)) as executor:
                futures = {executor.submit(self.store.load_centroid, cid): cid
                           for cid in fetch_ids}
                for future in as_completed(futures):
                    cid = futures[future]
                    t_fetch = time.perf_counter()
                    ids, vecs, version = future.result()
                    fetch_lat = (time.perf_counter() - t_fetch) * 1000
                    nbytes = int(ids.nbytes + vecs.nbytes)
                    total_bytes_fetched += nbytes
                    fetch_timings[cid] = (fetch_lat, nbytes)
                    self._cache.put(cid, (ids, vecs, version))
                    entries[cid] = (ids, vecs, version)
                    cached_re = self._recon_error(cid, vecs)
                    self._init_freshness(cid, version, partition_size=len(ids),
                                        cached_reconstruction_error=cached_re)

        # Backfill per-partition fetch timings into the log
        for pd in partition_log:
            if pd.partition_id in fetch_timings:
                pd.fetch_latency_ms, pd.bytes_fetched = fetch_timings[pd.partition_id]

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
        total_ms = (time.perf_counter() - t0) * 1000

        # Update per-partition hit rate EMA
        result_id_set = {r["id"] for r in results}
        contributing_cids: set[int] = set()
        for cid in probe_ids:
            cid = int(cid)
            if cid in entries:
                for vid in entries[cid][0]:
                    if int(vid) in result_id_set:
                        contributing_cids.add(cid)
                        break
        alpha = self._hit_rate_alpha
        with self._freshness_lock:
            for cid in probe_ids:
                cid = int(cid)
                hit = 1.0 if cid in contributing_cids else 0.0
                ema = self._hit_rate_ema.get(cid, 0.0)
                self._hit_rate_ema[cid] = (1 - alpha) * ema + alpha * hit
                if cid in self._freshness:
                    self._freshness[cid]["historical_hit_rate"] = round(self._hit_rate_ema[cid], 4)

        cold_fetch_count   = sum(1 for pd in partition_log if not pd.in_cache and pd.decision == "fetch")
        policy_fetch_count = sum(1 for pd in partition_log if pd.in_cache  and pd.decision == "fetch")

        return QueryResult(
            top_k=results,
            total_ms=total_ms,
            centroid_search_ms=centroid_search_ms,
            fetch_ms=fetch_ms,
            inference_ms=inference_ms,
            scan_ms=scan_ms,
            cache_hits=cache_hits,
            fetch_count=len(fetch_ids),
            bytes_fetched=total_bytes_fetched,
            cold_fetch_count=cold_fetch_count,
            policy_fetch_count=policy_fetch_count,
            partition_log=partition_log,
        )

    def _recon_error(self, cid: int, vecs: np.ndarray) -> float:
        """Mean L2 distance from partition vectors to their centroid."""
        if len(vecs) == 0:
            return 0.0
        centroid = self.centroids[cid].astype(np.float32)
        diffs = vecs.astype(np.float32) - centroid
        return float(np.mean(np.sqrt(np.sum(diffs ** 2, axis=1))))

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

    def _init_freshness(self, cid: int, cached_version: int, partition_size: int,
                        cached_reconstruction_error: float = 0.0) -> None:
        with self._freshness_lock:
            self._freshness[cid] = {
                "cached_version":                       cached_version,
                "latest_known_version":                 cached_version,
                "cumulative_inserts_since_cache":       0,
                "cumulative_updates_since_cache":       0,
                "cumulative_deletes_since_cache":       0,
                "cumulative_membership_changes_since_cache": 0,
                "cached_partition_size":                partition_size,
                "latest_partition_size":                partition_size,
                "latest_fraction_vectors_touched":      None,
                "latest_reconstruction_error":          None,
                "cached_reconstruction_error":          cached_reconstruction_error,
                "latest_centroid":                      None,
                "last_metadata_update_time":            None,
                "historical_hit_rate":                  self._hit_rate_ema.get(cid, 0.0),
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
