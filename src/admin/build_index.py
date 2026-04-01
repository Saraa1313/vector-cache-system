import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np


from config import N_CENTROIDS, CENTROIDS_PATH, DATA_DIR
from admin.data_loader import read_fvecs
from admin.ivf import train_coarse_quantizer, assign_to_centroids
from storage.object_store import ObjectStore


def main(base_path: str):
    print(f"Loading vectors from {base_path} ...")
    vectors = read_fvecs(base_path)
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    n, d = vectors.shape
    ids = np.arange(n, dtype=np.int64)
    print(f"  {n} vectors, dim={d}")

    print(f"Training k-means with {N_CENTROIDS} centroids ...")
    centroids = train_coarse_quantizer(vectors, d, N_CENTROIDS)

    print("Assigning vectors to centroids ...")
    assignments = assign_to_centroids(vectors, centroids)

    print("Uploading centroid objects to MinIO ...")
    store = ObjectStore()
    for cid in range(N_CENTROIDS):
        mask = assignments == cid
        store.save_centroid(cid, ids[mask], vectors[mask])
        print(f"  centroid {cid:04d}: {mask.sum()} vectors")

    print(f"Saving centroids locally → {CENTROIDS_PATH}")
    np.save(CENTROIDS_PATH, centroids)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default=os.path.join(DATA_DIR, "sift", "sift_base.fvecs"),
        help="Path to base vectors file (.fvecs)",
    )
    args = parser.parse_args()
    main(args.base)
