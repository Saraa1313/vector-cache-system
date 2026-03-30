import numpy as np
import torch
import quake


class QuakeBackend:
    def __init__(self):
        self.index = quake.QuakeIndex()

    def build(self, vectors: np.ndarray, ids: np.ndarray, nlist: int):
        xb = torch.tensor(vectors, dtype=torch.float32)
        ids_t = torch.tensor(ids, dtype=torch.int64)

        params = quake.IndexBuildParams()
        params.nlist = nlist
        self.index.build(xb, ids_t, params)

    def add(self, vectors: np.ndarray, ids: np.ndarray):
        xb = torch.tensor(vectors, dtype=torch.float32)
        ids_t = torch.tensor(ids, dtype=torch.int64)
        self.index.add(xb, ids_t)

    def remove(self, ids: np.ndarray):
        ids_t = torch.tensor(ids, dtype=torch.int64)
        self.index.remove(ids_t)

    def maintenance(self):
        return self.index.maintenance()