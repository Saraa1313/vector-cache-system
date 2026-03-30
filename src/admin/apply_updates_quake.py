import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admin.quake_backend import QuakeBackend
from admin.quake_state import QuakeState
from admin.export_snapshot_to_minio import export_snapshot_to_minio


def apply_updates_quake(mutation_fraction=0.1, drift_scale=0.5):
    state = QuakeState()
    ids, vectors, meta = state.load()

    n = len(vectors)
    m = max(1, int(n * mutation_fraction))
    chosen = np.random.choice(n, m, replace=False)

    vectors[chosen] += np.random.randn(m, vectors.shape[1]).astype("float32") * drift_scale

    q = QuakeBackend()
    q.build(vectors, ids, nlist=meta["nlist"])
    q.maintenance()

    meta["version"] += 1
    meta["last_update_step"] += 1
    meta["mutation_fraction"] = mutation_fraction
    meta["drift_scale"] = drift_scale

    state.save(ids, vectors, meta)
    export_snapshot_to_minio()

    print("Applied updates to Quake authority and exported new snapshot.")


if __name__ == "__main__":
    apply_updates_quake()