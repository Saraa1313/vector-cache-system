import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admin.quake_state import QuakeState
from storage.object_store import ObjectStore
from config import NUM_PARTITIONS


def export_snapshot_to_minio():
    state = QuakeState()
    ids, vectors, meta = state.load()

    store = ObjectStore()

    remote_meta = {
        "_snapshot": {
            "snapshot_version": meta["version"],
            "last_update_step": meta["last_update_step"],
            "backend": "quake_snapshot",
            "nlist": meta["nlist"],
        }
    }

    total = len(ids)
    chunk = total // NUM_PARTITIONS
    start = 0

    for p in range(NUM_PARTITIONS):
        end = start + chunk if p < NUM_PARTITIONS - 1 else total

        part_ids = ids[start:end]
        part_vecs = vectors[start:end]

        fname = f"part_{p:04d}.npz"
        store.save_partition(fname, part_ids, part_vecs)

        remote_meta[fname] = {
            "version": meta["version"],
            "snapshot_version": meta["version"],
            "num_vectors": int(len(part_ids)),
            "last_update_step": meta["last_update_step"],
            "partition_id": p,
            "backend": "quake_snapshot",
        }

        start = end

    store.write_metadata(remote_meta)
    print(f"Exported snapshot version {meta['version']} to MinIO.")


if __name__ == "__main__":
    export_snapshot_to_minio()