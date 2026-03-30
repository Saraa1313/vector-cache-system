import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DIM, NUM_PARTITIONS, VECS_PER_PARTITION
from admin.quake_backend import QuakeBackend
from admin.quake_state import QuakeState
from admin.export_snapshot_to_minio import export_snapshot_to_minio


def build_authoritative_index_quake():
    total = NUM_PARTITIONS * VECS_PER_PARTITION
    ids = np.arange(total, dtype=np.int64)
    vectors = np.random.randn(total, DIM).astype("float32")

    q = QuakeBackend()
    q.build(vectors, ids, nlist=NUM_PARTITIONS)

    state = QuakeState()
    state.save(
        ids,
        vectors,
        {
            "version": 1,
            "last_update_step": 0,
            "nlist": NUM_PARTITIONS,
            "backend": "quake"
        }
    )

    export_snapshot_to_minio()
    print("Built Quake authority and exported snapshot.")
    

if __name__ == "__main__":
    build_authoritative_index_quake()