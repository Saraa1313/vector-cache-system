import os
import json
import numpy as np


class QuakeState:
    def __init__(self, state_dir="state"):
        self.state_dir = state_dir
        os.makedirs(self.state_dir, exist_ok=True)
        self.id_path = os.path.join(self.state_dir, "ids.npy")
        self.vec_path = os.path.join(self.state_dir, "vectors.npy")
        self.meta_path = os.path.join(self.state_dir, "meta.json")

    def save(self, ids: np.ndarray, vectors: np.ndarray, meta: dict):
        np.save(self.id_path, ids)
        np.save(self.vec_path, vectors)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load(self):
        ids = np.load(self.id_path)
        vectors = np.load(self.vec_path)
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        return ids, vectors, meta