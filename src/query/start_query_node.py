import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from storage.object_store import ObjectStore
from query.cache_manager import CacheManager
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import QUERY_NODE_HOST, QUERY_NODE_PORT, DIM
from query.cache_manager import CacheManager
from query.query_node import QueryNode

app = Flask(__name__)
store = ObjectStore()
cache = CacheManager()
node = QueryNode(cache)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/cache_status")
def cache_status():
    remote_meta = store.read_metadata()
    cache_meta = cache.read_cache_metadata()

    remote_snapshot = remote_meta.get("_snapshot", {}).get("snapshot_version", -1)
    cached_snapshot = cache_meta.get("_snapshot", {}).get("snapshot_version", -1)

    status = {
        "snapshot": {
            "remote_snapshot_version": remote_snapshot,
            "cached_snapshot_version": cached_snapshot,
            "is_stale": cached_snapshot < remote_snapshot,
        }
    }

    for fname, info in remote_meta.items():
        if fname == "_snapshot":
            continue

        cached_info = cache_meta.get(fname, {})
        cached_version = cached_info.get("cached_version", -1)
        remote_version = info.get("version", -1)

        status[fname] = {
            "cached_version": cached_version,
            "remote_version": remote_version,
            "is_stale": cached_version < remote_version,
        }

    return jsonify(status)


@app.post("/rebuild_local_index")
def rebuild_local_index():
    # Force refresh all partitions from the authoritative object store.
    cache.refresh_all(step=0)
    cache.snapshot_remote_metadata()
    return jsonify({"status": "ok", "message": "local cache rebuilt"})


@app.post("/search")
def search():
    payload = request.get_json(force=True)
    query = payload.get("query")

    if query is None:
        return jsonify({"error": "Missing 'query' field"}), 400

    query = np.array(query, dtype=np.float32)
    if query.shape[0] != DIM:
        return jsonify({"error": f"Expected query dimension {DIM}, got {query.shape[0]}"}), 400

    topk = int(payload.get("topk", 5))
    results = node.search(query, topk=topk)
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host=QUERY_NODE_HOST, port=QUERY_NODE_PORT, debug=False)