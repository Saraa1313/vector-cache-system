import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.object_store import ObjectStore
from query.cache_manager import CacheManager


def warm_cache():
    store = ObjectStore()
    cache = CacheManager()

    remote_meta = store.read_metadata()

    cache_meta = {
        "_snapshot": {
            "snapshot_version": remote_meta["_snapshot"]["snapshot_version"]
        }
    }

    for fname, info in remote_meta.items():
        if fname == "_snapshot":
            continue

        ids, vecs = store.load_partition(fname)
        cache.save_partition(fname, ids, vecs)

        cache_meta[fname] = {
            "cached_version": info["version"],
            "snapshot_version": info["snapshot_version"],
            "partition_id": info["partition_id"],
            "num_vectors": info["num_vectors"],
        }

    cache.save_cache_metadata(cache_meta)
    print(
        f"Warmed cache with snapshot version "
        f"{cache_meta['_snapshot']['snapshot_version']}"
    )


if __name__ == "__main__":
    warm_cache()