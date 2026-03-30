import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.object_store import ObjectStore


def apply_updates(mutation_fraction=0.1, drift_scale=0.5, step=1):
    store = ObjectStore()
    remote_meta = store.read_metadata()

    if not remote_meta:
        raise RuntimeError("Remote metadata is empty. Build the authoritative index first.")

    for fname, info in remote_meta.items():
        ids, vecs = store.load_partition(fname)

        n = len(vecs)
        m = max(1, int(n * mutation_fraction))
        chosen = np.random.choice(n, m, replace=False)

        noise = np.random.randn(m, vecs.shape[1]).astype(np.float32) * drift_scale
        vecs[chosen] += noise

        store.save_partition(fname, ids, vecs)
        info["version"] += 1
        info["last_update_step"] = step

    store.write_metadata(remote_meta)
    print("Applied updates to authoritative index in MinIO only.")


if __name__ == "__main__":
    apply_updates()