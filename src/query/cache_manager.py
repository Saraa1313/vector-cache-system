import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CACHE_DIR, CACHE_META_PATH, REMOTE_META_SNAPSHOT_PATH
from storage.object_store import ObjectStore


class CacheManager:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(CACHE_META_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(REMOTE_META_SNAPSHOT_PATH), exist_ok=True)
        self.store = ObjectStore()

    def _cache_path(self, fname: str) -> str:
        return os.path.join(CACHE_DIR, fname)

    def load_partition(self, fname: str):
        arr = np.load(self._cache_path(fname))
        return arr["ids"], arr["vectors"]

    def save_partition(self, fname: str, ids, vectors):
        np.savez(
            self._cache_path(fname),
            ids=np.array(ids, dtype=np.int64),
            vectors=np.array(vectors, dtype=np.float32),
        )

    def has_partition(self, fname: str) -> bool:
        return os.path.exists(self._cache_path(fname))

    def refresh_partition(self, fname: str, step=0):
        if fname == "_snapshot":
            return

        ids, vecs = self.store.load_partition(fname)
        self.save_partition(fname, ids, vecs)

        remote_meta = self.store.read_metadata()
        cache_meta = self.read_cache_metadata()
        info = remote_meta[fname]

        # keep global snapshot metadata in local cache metadata
        if "_snapshot" in remote_meta:
            cache_meta["_snapshot"] = {
                "snapshot_version": remote_meta["_snapshot"]["snapshot_version"],
                "last_update_step": remote_meta["_snapshot"].get("last_update_step"),
                "backend": remote_meta["_snapshot"].get("backend"),
                "nlist": remote_meta["_snapshot"].get("nlist"),
            }

        cache_meta[fname] = {
            "cached_version": info["version"],
            "snapshot_version": info.get("snapshot_version"),
            "num_vectors": info["num_vectors"],
            "partition_id": info.get("partition_id"),
            "last_refresh_step": step,
        }
        self.write_cache_metadata(cache_meta)

    def refresh_all(self, step=0):
        remote_meta = self.store.read_metadata()
        cache_meta = self.read_cache_metadata()

        # copy global snapshot metadata once
        if "_snapshot" in remote_meta:
            cache_meta["_snapshot"] = {
                "snapshot_version": remote_meta["_snapshot"]["snapshot_version"],
                "last_update_step": remote_meta["_snapshot"].get("last_update_step"),
                "backend": remote_meta["_snapshot"].get("backend"),
                "nlist": remote_meta["_snapshot"].get("nlist"),
            }
            self.write_cache_metadata(cache_meta)

        for fname in remote_meta:
            if fname == "_snapshot":
                continue
            self.refresh_partition(fname, step=step)

    def read_cache_metadata(self):
        if not os.path.exists(CACHE_META_PATH):
            return {}
        with open(CACHE_META_PATH, "r") as f:
            return json.load(f)

    def write_cache_metadata(self, metadata: dict):
        with open(CACHE_META_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_cache_metadata(self, metadata: dict):
        self.write_cache_metadata(metadata)

    def snapshot_remote_metadata(self):
        remote_meta = self.store.read_metadata()
        with open(REMOTE_META_SNAPSHOT_PATH, "w") as f:
            json.dump(remote_meta, f, indent=2)

    def compare_versions(self):
        remote_meta = self.store.read_metadata()
        cache_meta = self.read_cache_metadata()

        remote_snapshot = remote_meta.get("_snapshot", {}).get("snapshot_version", -1)
        cached_snapshot = cache_meta.get("_snapshot", {}).get("snapshot_version", -1)

        comparison = {
            "snapshot": {
                "remote_snapshot_version": remote_snapshot,
                "cached_snapshot_version": cached_snapshot,
                "is_stale": cached_snapshot < remote_snapshot,
            }
        }

        for fname, rinfo in remote_meta.items():
            if fname == "_snapshot":
                continue

            cached_version = cache_meta.get(fname, {}).get("cached_version", -1)
            comparison[fname] = {
                "remote_version": rinfo["version"],
                "cached_version": cached_version,
                "is_stale": cached_version < rinfo["version"],
            }

        return comparison