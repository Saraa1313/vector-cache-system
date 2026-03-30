import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NUM_PARTITIONS, VECS_PER_PARTITION, DIM
from storage.object_store import ObjectStore


def build_authoritative_index():
    store = ObjectStore()
    metadata = {}

    next_id = 0
    for p in range(NUM_PARTITIONS):
        ids = []
        vecs = []

        # Toy partition generation for now
        # Later replace with Quake-exported partitions
        for _ in range(VECS_PER_PARTITION):
            ids.append(next_id)
            vecs.append(np.random.randn(DIM).astype(np.float32))
            next_id += 1

        fname = f"part_{p:04d}.npz"
        store.save_partition(fname, ids, vecs)

        metadata[fname] = {
            "version": 1,
            "num_vectors": len(ids),
            "last_update_step": 0,
            "partition_id": p,
        }

    store.write_metadata(metadata)
    print("Authoritative index uploaded to MinIO.")
    print(f"Partitions: {len(metadata)}")


if __name__ == "__main__":
    build_authoritative_index()